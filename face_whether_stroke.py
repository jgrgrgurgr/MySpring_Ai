import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import json
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)

class StrokePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

        # 레이블 파일 로드
        labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")
        with open(labels_path, "r", encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image) / 255.0
            input_scale, input_zero_point = self.input_details[0]['quantization']
            if input_scale > 0:
                uint8_image_array = ((image_array / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
            else:
                uint8_image_array = (image_array * 255).astype(np.uint8)
            input_data = np.expand_dims(uint8_image_array, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data
        except Exception as e:
            print(f"Error in extract_features for {image_path}: {e}")
            raise

    def predict_image(self, image_path):
        try:
            current_features = self.extract_features(image_path)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:
                original_probabilities = output_data_float[0]
            else:
                original_probabilities = tf.nn.softmax(output_data_float[0]).numpy()
            stroke_probability = float(original_probabilities[0])
            non_stroke_probability = float(original_probabilities[1])
            severity_score = stroke_probability
            class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
            return {
                "filename": os.path.basename(image_path),
                "class": class_name,
                "stroke_probability": stroke_probability,
                "non_stroke_probability": non_stroke_probability,
                "severity_score": severity_score
            }
        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }

    def extract_features_from_image(self, image):
        try:
            image = image.convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image) / 255.0
            input_scale, input_zero_point = self.input_details[0]['quantization']
            if input_scale > 0:
                uint8_image_array = ((image_array / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
            else:
                uint8_image_array = (image_array * 255).astype(np.uint8)
            input_data = np.expand_dims(uint8_image_array, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data
        except Exception as e:
            print(f"Error in extract_features_from_image: {e}")
            raise

    def predict_image_from_image(self, image):
        try:
            current_features = self.extract_features_from_image(image)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:
                original_probabilities = output_data_float[0]
            else:
                original_probabilities = tf.nn.softmax(output_data_float[0]).numpy()
            stroke_probability = float(original_probabilities[0])
            non_stroke_probability = float(original_probabilities[1])
            severity_score = stroke_probability
            class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
            return {
                "filename": "unknown",
                "class": class_name,
                "stroke_probability": stroke_probability,
                "non_stroke_probability": non_stroke_probability,
                "severity_score": severity_score
            }
        except Exception as e:
            return {
                "filename": "unknown",
                "error": str(e)
            }

def process_folder(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    results = []
    predictor = StrokePredictor(os.path.join(os.path.dirname(__file__), "model.tflite"))

    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                result = predictor.predict_image(image_path)
                results.append(result)

                if "error" in result:
                    print(f"파일 {result['filename']}: 처리 중 오류 발생 - {result['error']}")
                else:
                    print(f"파일 경로: {image_path}")
                    print(f"분류 결과: {result['class']}")
                    print(f"질환 (Stroke) 확률: {round(result['stroke_probability'] * 100):d}%")
                    print(f"비질환 (Non-Stroke) 확률: {round(result['non_stroke_probability'] * 100):d}%")
                    print(f"심각도 점수: {round(result['severity_score'] * 100):d}%")
                    print("-" * 50)
        return results
    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {str(e)}")
        return None

def send_to_backend(results, backend_url):
    try:
        severity_scores = [round(result["severity_score"] * 100) for result in results]
        data = json.dumps(severity_scores, ensure_ascii=False)
        response = requests.post(backend_url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
        response.raise_for_status()
        print(f"데이터가 백엔드에 성공적으로 전송됨. 상태 코드: {response.status_code}")
        if response.text:
            print(f"백엔드 응답: {response.text}")
        return response
    except requests.exceptions.RequestException as e:
        print(f"백엔드 전송 중 오류 발생: {e}")
        return None

def main():
    folder_path = input("이미지 폴더의 경로를 입력하세요: ")
    print(f"처리할 폴더: {folder_path}")
    print("=" * 50)
    results = process_folder(folder_path)
    if results:
        print("\n처리 완료 통계:")
        print(f"총 처리된 이미지: {len(results)}개")
        successful = sum(1 for r in results if "error" not in r)
        print(f"성공: {successful}개")
        print(f"실패: {len(results) - successful}개")

        backend_url = ""
        send_prompt = input("\n예측 결과를 백엔드에 전송하시겠습니까? (yes/no): ")
        if send_prompt.lower() == "yes":
            send_to_backend(results, backend_url)
        else:
            print("백엔드에 데이터를 전송하지 않습니다.")
    else:
        print("처리 결과가 없거나 오류가 발생했습니다.")

# Flask 서버 실행
predictor = StrokePredictor(os.path.join(os.path.dirname(__file__), "model.tflite"))

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_data()
        base64_data = data.decode('utf-8')
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        
        result = predictor.predict_image_from_image(image)
        
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 500
        else:
            results_list = [result]
            response = send_to_backend(results_list, "http://192.168.9.29:8080/face/receive")
            if response:
                return jsonify({"status": "success"}), 200
            else:
                return jsonify({"status": "error", "message": "Failed to send to backend"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)