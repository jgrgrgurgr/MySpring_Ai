import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os

# MoveNet 모델 로컬 캐싱 설정
MODEL_URL = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "movenet_thunder")

# MoveNet 모델을 로컬에 저장 (최초 실행 시 다운로드)
def cache_movenet_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading and caching MoveNet model to {LOCAL_MODEL_PATH}...")
        model = hub.KerasLayer(MODEL_URL)
        tf.saved_model.save(model, LOCAL_MODEL_PATH)
        print("MoveNet model cached successfully.")
    else:
        print(f"MoveNet model already cached at {LOCAL_MODEL_PATH}.")

# MoveNet 모델 로드 (로컬에서)
def load_movenet_model():
    cache_movenet_model()
    model = tf.saved_model.load(LOCAL_MODEL_PATH)
    return model.signatures['serving_default']

# 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '3')
tf.config.set_visible_devices([], 'GPU')

# MoveNet 모델 로드
movenet = load_movenet_model()

class StrokePredictor:
    def __init__(self, model_path, label_path, temperature=1.0):
        self.temperature = temperature
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(label_path, "r", encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, image):
        # MoveNet으로 17개 키포인트 추출 후 (x, y)만 flatten해서 모델 입력
        image = image.convert("RGB")
        image = ImageOps.fit(image, (256, 256), Image.LANCZOS)
        image_array = np.asarray(image, dtype=np.uint8)
        image_array = tf.cast(image_array, tf.int32)
        input_data = tf.expand_dims(image_array, axis=0)
        outputs = movenet(input_data)
        keypoints = outputs['output_0']
        keypoints = tf.squeeze(keypoints, axis=[0, 1])
        keypoints_xy = keypoints[:, :2]
        flattened = tf.reshape(keypoints_xy, [-1])
        input_data = np.expand_dims(flattened.numpy(), axis=0).astype(np.float32)
        return input_data

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image):
        input_data = self.preprocess(image)
        # 입력 텐서 형식 검사 및 변환
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = (input_data * 255).astype(np.uint8)
        else:
            input_data = input_data.astype(self.input_details[0]['dtype'])

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prediction = self.softmax(output / self.temperature)
        return {
            "label_scores": {self.labels[i]: float(prediction[i]) for i in range(len(self.labels))},
            "severity_score": float(prediction[1]) if len(prediction) > 1 else float(prediction[0])
        }