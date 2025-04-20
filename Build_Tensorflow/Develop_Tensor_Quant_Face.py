import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

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

# 이미지 유효성 검사 함수
def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image detected: {file_path}, Error: {e}")
        return False

# 이미지 전처리 함수
def load_and_preprocess_image(image_path, label):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE, preserve_aspect_ratio=True)
        image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    except tf.errors.InvalidArgumentError as e:
        logger.warning(f"Error loading image: {image_path}, Error: {e}")
        return tf.zeros(IMAGE_SIZE + (3,), dtype=tf.float32), label

def load_and_preprocess_image_wrapper(image_path, label):
    [image, label] = tf.py_function(
        func=load_and_preprocess_image,
        inp=[image_path, label],
        Tout=[tf.float32, tf.int32]
    )
    image.set_shape(IMAGE_SIZE + (3,))
    label.set_shape(())
    return image, label

# 데이터 경로
stroke_image_dir = '/Users/harold0812/Desktop/2025 2G/AI project (myspring)/Myspring_AI_Develop/Img_Hub/질환군'
normal_image_dir = '/Users/harold0812/Desktop/2025 2G/AI project (myspring)/Myspring_AI_Develop/Img_Hub/비질환군'

def get_image_paths(directory):
    logger.info(f"Scanning directory: {directory}")
    image_paths = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg')):
            file_path = os.path.join(directory, file)
            if is_valid_image(file_path):
                image_paths.append(file_path)
            else:
                logger.warning(f"Skipping invalid image: {file_path}")
    return image_paths

# 데이터 로드
stroke_image_paths = get_image_paths(stroke_image_dir)
normal_image_paths = get_image_paths(normal_image_dir)
image_paths = stroke_image_paths + normal_image_paths
labels = [1] * len(stroke_image_paths) + [0] * len(normal_image_paths)

train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

# 클래스 가중치
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# 데이터 증강
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# 오버샘플링
stroke_paths = [p for p, l in zip(train_image_paths, train_labels) if l == 1]
normal_paths = [p for p, l in zip(train_image_paths, train_labels) if l == 0]
repeat_factor = len(normal_paths) // len(stroke_paths)
augmented_stroke_paths = stroke_paths * repeat_factor + stroke_paths[:len(normal_paths) % len(stroke_paths)]

balanced_image_paths = normal_paths + augmented_stroke_paths
balanced_labels = [0] * len(normal_paths) + [1] * len(augmented_stroke_paths)

balanced_dataset = tf.data.Dataset.from_tensor_slices((balanced_image_paths, balanced_labels))
train_dataset = balanced_dataset.shuffle(1000).map(
    load_and_preprocess_image_wrapper,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).filter(lambda x, y: tf.reduce_sum(x) > 0).map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).batch(BATCH_SIZE).repeat().prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(
    load_and_preprocess_image_wrapper,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
).filter(lambda x, y: tf.reduce_sum(x) > 0).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

steps_per_epoch = max(1, len(balanced_image_paths) // BATCH_SIZE)

# 모델 생성
base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset, callbacks=[early_stopping],
                    class_weight=class_weight_dict)

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
fine_tune_history = model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, steps_per_epoch=steps_per_epoch,
                              validation_data=test_dataset, callbacks=[early_stopping],
                              class_weight=class_weight_dict)

# Representative dataset for TFLite
def representative_dataset():
    for image_path in train_image_paths[:100]:
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE, preserve_aspect_ratio=True)
            image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
            image = tf.cast(image, tf.float32) / 255.0
            yield [tf.expand_dims(image, 0)]
        except tf.errors.InvalidArgumentError as e:
            logger.warning(f"Error in representative dataset: {image_path}, Error: {e}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

with open('label.txt', 'w', encoding='utf-8') as f:
    f.write("비질환군\n질환군\n")

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + fine_tune_history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Classification report
test_images = []
test_labels_list = []
for images, labels in test_dataset.unbatch():
    test_images.append(images.numpy())
    test_labels_list.append(labels.numpy())
test_images = np.array(test_images)
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
print("\nClassification Report:")
print(classification_report(test_labels_list, predicted_classes, target_names=['비질환군', '질환군']))