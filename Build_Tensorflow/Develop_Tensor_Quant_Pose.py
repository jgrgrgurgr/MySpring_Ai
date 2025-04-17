import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

# TensorFlow 및 GPU 확인
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# MoveNet 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = movenet.signatures['serving_default']

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [256, 256])  # MoveNet Thunder 입력 크기
        image = tf.cast(image, tf.int32)  # int32 타입으로 변환 (0~255 범위 유지)
        image = tf.expand_dims(image, 0)  # 배치 차원 추가
        return image
    except tf.errors.InvalidArgumentError:
        print(f"Error loading image: {image_path}")
        return None

# 키포인트 추출 함수
def extract_landmarks(image):
    outputs = movenet(image)
    keypoints = outputs['output_0']  # 예상 shape: (1, 1, 17, 3)
    print("Raw keypoints shape:", keypoints.shape)  # 디버깅용 출력
    
    keypoints = tf.squeeze(keypoints, axis=[0, 1])  # 배치 및 단일 포즈 차원 제거 -> (17, 3)
    print("Squeezed keypoints shape:", keypoints.shape)  # 디버깅용 출력
    
    keypoints_xy = keypoints[:, :2]  # x, y 좌표만 추출 -> (17, 2)
    print("Keypoints_xy shape:", keypoints_xy.shape)  # 디버깅용 출력
    
    flattened = tf.reshape(keypoints_xy, [-1])  # 평평한 벡터로 변환 -> (34,)
    print("Flattened shape:", flattened.shape)  # 디버깅용 출력
    return flattened

# 포즈 데이터셋 경로 설정
poses_dir = '/Users/harold0812/Desktop/2025 2G/AI project (myspring)/Myspring_AI_Develop/Img_Hub'

# 이미지 경로 수집 함수
def get_image_paths_and_labels(directory):
    pose_classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if len(pose_classes) != 2:
        raise ValueError(f"Exactly 2 pose classes are expected, but found {len(pose_classes)}: {pose_classes}")
    image_paths = []
    labels = []
    for label, class_name in enumerate(pose_classes):
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_paths.append(os.path.join(class_dir, file))
                labels.append(label)
    return image_paths, labels, pose_classes

# 데이터셋 로드
image_paths, labels, pose_classes = get_image_paths_and_labels(poses_dir)
print(f"Detected pose classes: {pose_classes}")

# 키포인트 미리 추출
def preprocess_dataset(image_paths, labels):
    keypoints_list = []
    valid_labels = []
    for img_path, label in zip(image_paths, labels):
        image = load_and_preprocess_image(img_path)
        if image is not None:
            keypoints = extract_landmarks(image)
            keypoints_list.append(keypoints.numpy())  # Tensor를 NumPy 배열로 변환
            valid_labels.append(label)
    keypoints_array = np.array(keypoints_list)
    print("Final keypoints array shape:", keypoints_array.shape)  # 디버깅용 출력
    return keypoints_array, np.array(valid_labels)

# 훈련/테스트 데이터 분할
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# 키포인트 추출
print("Extracting keypoints for training data...")
train_keypoints, train_labels = preprocess_dataset(train_image_paths, train_labels)
print("Extracting keypoints for test data...")
test_keypoints, test_labels = preprocess_dataset(test_image_paths, test_labels)

# 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((train_keypoints, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(64)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_keypoints, test_labels))
test_dataset = test_dataset.batch(64)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# 모델 정의
num_classes = len(pose_classes)  # 2
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(34,)),  # 17개 키포인트 * 2
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 조기 중단 콜백
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 모델 훈련
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

# TFLite 변환을 위한 대표 데이터셋
def representative_dataset():
    for keypoints in train_keypoints[:100]:
        yield [tf.expand_dims(keypoints, 0)]

# TFLite 모델로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# TFLite 모델 저장
with open('pose_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 레이블 파일 저장
with open('pose_labels.txt', 'w') as f:
    for class_name in pose_classes:
        f.write(class_name + '\n')

# 훈련 결과 시각화
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()