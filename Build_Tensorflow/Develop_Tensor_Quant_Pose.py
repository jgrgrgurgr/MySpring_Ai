import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = (256, 256)  # MoveNet Thunder 입력 크기
BATCH_SIZE = 64
EPOCHS = 30
POSE_CLASSES = None

# TensorFlow 및 GPU 확인
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# GPU 메모리 관리
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")
    except RuntimeError as e:
        logger.error(f"GPU setup error: {e}")

# MoveNet 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = movenet.signatures['serving_default']
print("MoveNet input signature:", movenet.structured_input_signature)

# 이미지 전처리 함수
def load_and_preprocess_image(image_path, label):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE, preserve_aspect_ratio=True)
        image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        image = tf.cast(image, tf.float32) / 255.0  # MoveNet Thunder는 float32 정규화 입력을 기대
        image = tf.expand_dims(image, 0)  # [1, 256, 256, 3]
        return image, label
    except tf.errors.InvalidArgumentError as e:
        logger.warning(f"Error loading image: {image_path}, Error: {e}")
        return tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32), label

# 키포인트 추출 함수
def extract_landmarks(image):
    logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    outputs = movenet(image)
    keypoints = outputs['output_0']  # (1, 1, 17, 3)
    keypoints = tf.squeeze(keypoints, axis=[0, 1])  # (17, 3)
    keypoints_xy = keypoints[:, :2]  # (17, 2)
    flattened = tf.reshape(keypoints_xy, [-1])  # (34,)
    return flattened

# 포즈 데이터셋 경로 설정
poses_dir = '/Users/harold0812/Desktop/2025 2G/Pose_Ext'

# 이미지 경로 수집 함수
def get_image_paths_and_labels(directory):
    global POSE_CLASSES
    pose_classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if len(pose_classes) != 2:
        raise ValueError(f"Exactly 2 pose classes expected, found {len(pose_classes)}: {pose_classes}")
    POSE_CLASSES = pose_classes
    image_paths = []
    labels = []
    for label, class_name in enumerate(pose_classes):
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(label)
    return image_paths, labels

# 키포인트 미리 추출
def preprocess_dataset(image_paths, labels):
    keypoints_list = []
    valid_labels = []
    for img_path, label in zip(image_paths, labels):
        image, _ = load_and_preprocess_image(img_path, label)
        if tf.reduce_sum(image) > 0:
            try:
                keypoints = extract_landmarks(image)
                keypoints_list.append(keypoints.numpy())
                valid_labels.append(label)
            except Exception as e:
                logger.warning(f"Failed to extract keypoints for {img_path}: {e}")
        else:
            logger.warning(f"Skipping empty or invalid image: {img_path}")
    keypoints_array = np.array(keypoints_list)
    logger.info(f"Processed keypoints shape: {keypoints_array.shape}")
    return keypoints_array, np.array(valid_labels)

# 데이터 로드
image_paths, labels = get_image_paths_and_labels(poses_dir)
logger.info(f"Detected pose classes: {POSE_CLASSES}")

# 훈련/테스트 데이터 분할
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# 키포인트 추출
logger.info("Extracting keypoints for training data...")
train_keypoints, train_labels = preprocess_dataset(train_image_paths, train_labels)
logger.info("Extracting keypoints for test data...")
test_keypoints, test_labels = preprocess_dataset(test_image_paths, test_labels)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = dict(enumerate(class_weights))
logger.info(f"Class weights: {class_weight_dict}")

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_keypoints, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_keypoints, test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = max(1, len(train_keypoints) // BATCH_SIZE)

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(34,)),  # 17개 키포인트 * 2
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(POSE_CLASSES), activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 조기 중단 콜백
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# 모델 훈련
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    callbacks=[early_stopping],
    class_weight=class_weight_dict
)

# TFLite 변환을 위한 대표 데이터셋
def representative_dataset():
    for keypoints in train_keypoints[:100]:
        yield [tf.expand_dims(keypoints.astype(np.float32), 0)]

# TFLite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32  # 입력 타입을 float32로 유지
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# TFLite 모델 저장
with open('pose_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 레이블 파일 저장
with open('pose_labels.txt', 'w', encoding='utf-8') as f:
    for class_name in POSE_CLASSES:
        f.write(class_name + '\n')

# 훈련 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 분류 보고서
test_predictions = model.predict(test_keypoints)
predicted_classes = np.argmax(test_predictions, axis=1)
print("\nClassification Report:")
print(classification_report(test_labels, predicted_classes, target_names=POSE_CLASSES))