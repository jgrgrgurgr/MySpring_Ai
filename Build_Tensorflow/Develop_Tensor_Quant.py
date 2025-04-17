import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_and_preprocess_image(image_path, label):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)  # 다양한 형식 지원
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  # 훈련 시 FLOAT32로 정규화
        return image, label
    except tf.errors.InvalidArgumentError:
        print(f"Error loading image: {image_path}")
        return None, label

stroke_image_dir = '/Users/harold0812/Desktop/2025 2G/이미지_증강(임시)/질환군(증강 완료)'
normal_image_dir = '/Users/harold0812/Downloads/비질환군'

def get_image_paths(directory):
    image_paths = []
    for file in os.listdir(directory):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            image_paths.append(os.path.join(directory, file))
    return image_paths

stroke_image_paths = get_image_paths(stroke_image_dir)
normal_image_paths = get_image_paths(normal_image_dir)

image_paths = stroke_image_paths + normal_image_paths
labels = [1] * len(stroke_image_paths) + [0] * len(normal_image_paths)

train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE  # 병렬 처리
)
train_dataset = train_dataset.filter(lambda image, label: image is not None)  # None 필터링
train_dataset = train_dataset.shuffle(buffer_size=1000)  # 데이터 셔플
train_dataset = train_dataset.batch(64)  # 배치 크기 64 (필요 시 128로 조정 가능)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # 사전 가져오기

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
test_dataset = test_dataset.filter(lambda image, label: image is not None)
test_dataset = test_dataset.batch(64)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # 초기 학습 시 기본 모델 동결

model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # 데이터 증강
    tf.keras.layers.RandomRotation(0.2),       # 데이터 증강
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),              # 과적합 방지
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 낮은 학습률
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 조기 중단 콜백
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 모델 학습
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

def representative_dataset():
    for image_path in train_image_paths[:100]:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  # FLOAT32로 변환 및 정규화
        yield [tf.expand_dims(image, 0)]  # 배치 차원 추가

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # 추론 시 입력은 UINT8
converter.inference_output_type = tf.uint8  # 추론 시 출력은 UINT8
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

with open('label.txt', 'w') as f:
    f.write("질환군\n비질환군")

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