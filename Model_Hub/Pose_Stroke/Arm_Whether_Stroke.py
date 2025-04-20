import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import traceback

app = Flask(__name__)

# MoveNet 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = movenet.signatures['serving_default']

os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '3')
tf.config.set_visible_devices([], 'GPU')

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

predictor = StrokePredictor(
    os.path.join(os.path.dirname(__file__), "pose_model.tflite"),
    os.path.join(os.path.dirname(__file__), "pose_label.txt"),
    temperature=1.0 # 필요시 조정
)

@app.route('/pose/ai_send', methods=['POST'])
def ai_send():
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '' or not file:
                return jsonify({"status": "error", "message": "Invalid image file"}), 400
            image_data = file.read()
        elif request.is_json:
            data = request.get_json()
            if 'image' in data:
                try:
                    image_data = base64.b64decode(data['image'])
                except Exception as e:
                    return jsonify({"status": "error", "message": "Invalid base64 encoding"}), 400
            else:
                return jsonify({"status": "error", "message": "No image data provided in JSON"}), 400
        else:
            return jsonify({"status": "error", "message": "Invalid request: no image file or JSON data provided"}), 400
        
        try:
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({"status": "error", "message": "Failed to open image: " + str(e)}), 400
        
        result = predictor.predict(image)
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)