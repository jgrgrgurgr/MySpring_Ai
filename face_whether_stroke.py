# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import os
# import requests
# import json
# from flask import Flask, request, jsonify
# import base64
# from io import BytesIO
# import traceback

# app = Flask(__name__)

# class StrokePredictor:
#     def __init__(self, model_path, temperature=1.0):
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#         self.temperature = temperature
#         print("Input details:", self.input_details)
#         print("Output details:", self.output_details)

#         labels_path = os.path.join(os.path.split(__file__)[0], "labels.txt")
#         with open(labels_path, "r", encoding='utf-8') as f:
#             self.class_names = [line.strip() for line in f.readlines()]
#         print(f"Loaded labels: {self.class_names}")

#     def extract_features(self, image_path):
#         try:
#             image = Image.open(image_path).convert("RGB")
#             size = (224, 224)
#             image = ImageOps.fit(image, size, Image.LANCZOS)
#             image_array = np.asarray(image) / 255.0
#             input_scale, input_zero_point = self.input_details[0]['quantization']
#             if input_scale > 0:
#                 uint8_image_array = ((image_array / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
#             else:
#                 uint8_image_array = (image_array * 255).astype(np.uint8)
#             input_data = np.expand_dims(uint8_image_array, axis=0)
#             self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#             self.interpreter.invoke()
#             output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
#             return output_data
#         except Exception as e:
#             print(f"Error in extract_features for {image_path}: {e}")
#             raise

#     def predict_image(self, image_path):
#         try:
#             current_features = self.extract_features(image_path)
#             output_scale, output_zero_point = self.output_details[0]['quantization']
#             output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
#             if np.abs(np.sum(output_data_float[0]) - 1.0) < 0.1:
#                 logits = np.log(output_data_float[0] + 1e-10)  # Prevent log(0)
#             else:
#                 logits = output_data_float[0]
#             logits_scaled = logits / self.temperature
#             original_probabilities = tf.nn.softmax(logits_scaled).numpy()
#             stroke_probability = float(original_probabilities[0])
#             non_stroke_probability = float(original_probabilities[1])
#             severity_score = non_stroke_probability  # Changed to base on non-stroke probability
#             class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
#             return {
#                 "filename": os.path.basename(image_path),
#                 "class": class_name,
#                 "stroke_probability": stroke_probability,
#                 "non_stroke_probability": non_stroke_probability,
#                 "severity_score": severity_score
#             }
#         except Exception as e:
#             return {
#                 "filename": os.path.basename(image_path),
#                 "error": str(e)
#             }

#     def extract_features_from_image(self, image):
#         try:
#             image = image.convert("RGB")
#             size = (224, 224)
#             image = ImageOps.fit(image, size, Image.LANCZOS)
#             image_array = np.asarray(image) / 255.0
#             input_scale, input_zero_point = self.input_details[0]['quantization']
#             if input_scale > 0:
#                 uint8_image_array = ((image_array / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
#             else:
#                 uint8_image_array = (image_array * 255).astype(np.uint8)
#             input_data = np.expand_dims(uint8_image_array, axis=0)
#             self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
#             self.interpreter.invoke()
#             output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
#             return output_data
#         except Exception as e:
#             print(f"Error in extract_features_from_image: {e}")
#             raise

#     def predict_image_from_image(self, image):
#         try:
#             current_features = self.extract_features_from_image(image)
#             output_scale, output_zero_point = self.output_details[0]['quantization']
#             output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
#             if np.abs(np.sum(output_data_float[0]) - 1.0) < 0.1:
#                 logits = np.log(output_data_float[0] + 1e-10)
#             else:
#                 logits = output_data_float[0]
#             logits_scaled = logits / self.temperature
#             original_probabilities = tf.nn.softmax(logits_scaled).numpy()
#             stroke_probability = float(original_probabilities[0])
#             non_stroke_probability = float(original_probabilities[1])
#             severity_score = int(round(stroke_probability * 100))  # Changed to base on non-stroke probability
#             class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
#             return {
#                 "severity_score": severity_score
#             }
#         except Exception as e:
#             return {
#                 "filename": "unknown",
#                 "error": str(e)
#             }

# predictor = StrokePredictor(os.path.join(os.path.split(__file__)[0], "model.tflite"), temperature=0.5)

# @app.route('/api/ai_send', methods=['POST'])
# def ai_send():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"status": "error", "message": "No file provided"}), 400
#         file = request.files['file']
#         if file.filename == '' or not file:
#             return jsonify({"status": "error", "message": "Invalid image file"}), 400
#         image_data = file.read()
#         image = Image.open(BytesIO(image_data))
#         result = predictor.predict_image_from_image(image)
#         if "error" in result:
#             return jsonify({"status": "error", "message": result["error"]}), 500
#         else:
#             return jsonify({"status": "success", "result": result["severity_score"]}), 200
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

import tensorflow as tf
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

class StrokePredictor:
    def __init__(self, model_path, temperature=1.0):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.temperature = temperature
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)
        
        labels_path = os.path.join(os.path.split(__file__)[0], "label.txt")
        with open(labels_path, "r", encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features(self, image_path):
        """파일 경로에서 이미지 특징 추출"""
        try:
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image, dtype=np.float32) / 255.0  # 학습 시와 동일하게 정규화
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

    def extract_features_from_image(self, image):
        """PIL 이미지 객체에서 특징 추출"""
        try:
            image = image.convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image, dtype=np.float32) / 255.0  # 학습 시와 동일하게 정규화
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

    def predict_image(self, image_path):
        """파일 경로에서 이미지 예측"""
        try:
            current_features = self.extract_features(image_path)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale  # 역양자화
            stroke_probability = float(output_data_float[0])  # sigmoid 출력 (0~1)
            non_stroke_probability = 1.0 - stroke_probability
            severity_score = stroke_probability  # 두 번째 코드의 정의 유지
            class_name = self.class_names[0] if stroke_probability > 0.5 else self.class_names[1]  # "질환군" or "비질환군"
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

    def predict_image_from_image(self, image):
        """PIL 이미지 객체에서 예측"""
        try:
            current_features = self.extract_features_from_image(image)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale  # 역양자화
            stroke_probability = float(output_data_float[0])  # sigmoid 출력 (0~1)
            non_stroke_probability = 1.0 - stroke_probability
            severity_score = int(round(abs(stroke_probability) * 100))  # 첫 번째 코드의 정의 유지
            class_name = self.class_names[0] if stroke_probability > 0.5 else self.class_names[1]  # "질환군" or "비질환군"
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

predictor = StrokePredictor(os.path.join(os.path.split(__file__)[0], "model.tflite"), temperature=0.1)

@app.route('/api/ai_send', methods=['POST'])
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
        
        result = predictor.predict_image_from_image(image)
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 500
        else:
            return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)