import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import json
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import traceback

app = Flask(__name__)

# MoveNet 모델 로드 (전역 변수로 사용)
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = movenet.signatures['serving_default']
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')  # GPU 가속 비활성화

class StrokePredictor:
    def __init__(self, model_path, temperature=1.0):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.temperature = temperature
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)
        
        labels_path = os.path.join(os.path.split(__file__)[0], "pose_labels.txt")
        with open(labels_path, "r", encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features_from_image(self, image):
        """MoveNet을 사용해 이미지에서 키포인트 추출 후 TFLite 모델 입력 준비"""
        try:
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
            input_scale, input_zero_point = self.input_details[0]['quantization']
            if input_scale > 0:
                input_data = ((input_data / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data
        except Exception as e:
            print(f"Error in extract_features_from_image: {e}")
            raise

    def predict_image_from_image(self, image):
        """PIL 이미지 객체에서 포즈 예측, severity_score만 반환"""
        try:
            current_features = self.extract_features_from_image(image)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            
            probabilities = output_data_float[0]
            predicted_class_idx = np.argmax(probabilities)
            pose_probability = float(probabilities[predicted_class_idx])
            
            # severity_score만 반환하도록 수정 (두 번째 코드 방식 적용)
            severity_score = int(round(pose_probability * 100))
            return {
                "severity_score": severity_score
            }
        except Exception as e:
            return {
                "filename": "unknown",
                "error": str(e)
            }

predictor = StrokePredictor(os.path.join(os.path.split(__file__)[0], "pose_model.tflite"), temperature=0.1)

@app.route('/pose/ai_send', methods=['POST'])
def ai_send():
    try:
        print("Request received:", request.method, request.headers)
        print("Files in request:", request.files)
        print("JSON in request:", request.is_json, request.get_json(silent=True))

        if 'image' in request.files:
            file = request.files['image']
            print("File received:", file, "Filename:", file.filename)
            if file.filename == '' or not file:
                return jsonify({"status": "error", "message": "Invalid image file"}), 400
            image_data = file.read()
        elif request.is_json:
            data = request.get_json()
            print("JSON data:", data)
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
        
        result = predictor.predict_image_from_image(image)
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 500
        else:
            # 두 번째 코드처럼 severity_score만 반환
            return jsonify({"status": "success", "result": result["severity_score"]}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)