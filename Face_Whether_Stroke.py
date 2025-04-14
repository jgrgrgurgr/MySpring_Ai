import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import traceback
import logging
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max content length to 16MB

class StrokePredictor:
    def __init__(self, model_path, temperature=1.0):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.temperature = temperature
            logger.info("Model loaded successfully. Input details: %s, Output details: %s", self.input_details, self.output_details)
        except Exception as e:
            logger.error("Failed to load model: %s", str(e))
            raise RuntimeError(f"Model loading failed: {str(e)}")

        labels_path = os.path.join(os.path.dirname(__file__), "label.txt")
        try:
            with open(labels_path, "r", encoding='utf-8') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info("Loaded labels: %s", self.class_names)
            if len(self.class_names) < 2:
                raise ValueError("Label file must contain at least 2 classes (stroke, non-stroke)")
        except Exception as e:
            logger.error("Failed to load labels: %s", str(e))
            raise RuntimeError(f"Label file loading failed: {str(e)}")

    def extract_features_from_image(self, image):
        try:
            image = image.convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
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
            logger.error("Error in feature extraction: %s", str(e))
            raise

    def predict_image_from_image(self, image):
        try:
            current_features = self.extract_features_from_image(image)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            logger.debug("Raw model output: %s", output_data_float)

            if output_data_float.shape[1] != len(self.class_names):
                raise ValueError(f"Model output size ({output_data_float.shape[1]}) does not match number of classes ({len(self.class_names)})")
            
            if np.abs(np.sum(output_data_float[0]) - 1.0) < 0.1:
                logits = np.log(output_data_float[0] + 1e-10)
            else:
                logits = output_data_float[0]
            logits_scaled = logits / self.temperature
            original_probabilities = tf.nn.softmax(logits_scaled).numpy()
            logger.debug("Probabilities: %s", original_probabilities)

            # 수정: 모델 출력 순서에 맞게 확률 재할당
            stroke_probability = float(original_probabilities[1])  # 뇌졸중 확률 (1)
            non_stroke_probability = float(original_probabilities[0])  # 정상 확률 (0)
            severity_score = int(round(non_stroke_probability * 100))
            class_name = self.class_names[np.argmax(original_probabilities)]
            
            return {
                #"filename": "unknown",
                "class": class_name,
                #"stroke_probability": stroke_probability,
                # "non_stroke_probability": non_stroke_probability,
                "severity_score": severity_score
            }
        except Exception as e:
            logger.error("Error in prediction: %s", str(e))
            return {"filename": "unknown", "error": str(e)}

try:
    predictor = StrokePredictor(os.path.join(os.path.dirname(__file__), "model.tflite"), temperature=0.5)
except Exception as e:
    logger.critical("Failed to initialize predictor: %s", str(e))
    raise

@app.route('/api/ai_send', methods=['POST'])
def ai_send():
    try:
        logger.info("Received request: %s", request.headers)
        
        # 멀티파트 폼 데이터 처리
        if request.content_type and request.content_type.startswith('multipart/form-data'):
            if not request.files:
                logger.warning("No files provided in multipart form-data")
                return jsonify({"status": "error", "message": "No file provided in multipart form-data"}), 400
            file = list(request.files.values())[0]
            if not file or file.filename == '':
                logger.warning("No valid file provided")
                return jsonify({"status": "error", "message": "No valid image file provided"}), 400
            image_data = file.read()
            logger.info("Received file: %s, size: %d bytes", file.filename, len(image_data))
        
        # JSON 처리
        elif request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                logger.warning("No image data in JSON")
                return jsonify({"status": "error", "message": "No image data provided in JSON"}), 400
            image_data = base64.b64decode(data['image'])
            logger.info("Received Base64 image, size: %d bytes", len(image_data))
        
        # 원시 이진 데이터 처리
        else:
            if not request.data:
                logger.warning("No raw data provided")
                return jsonify({"status": "error", "message": "No raw data provided"}), 400
            image_data = request.data
            logger.info("Received raw data, size: %d bytes", len(image_data))

        image = Image.open(BytesIO(image_data))
        result = predictor.predict_image_from_image(image)
        if "error" in result:
            logger.error("Prediction failed: %s", result["error"])
            return jsonify({"status": "error", "message": result["error"]}), 500
        logger.info("Prediction result: %s", result)
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        logger.error("Unexpected error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
    
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
        
#         labels_path = os.path.join(os.path.split(__file__)[0], "label.txt")
#         with open(labels_path, "r", encoding='utf-8') as f:
#             self.class_names = [line.strip() for line in f.readlines()]
#         print(f"Loaded labels: {self.class_names}")

#     def extract_features(self, image_path):
#         """파일 경로에서 이미지 특징 추출"""
#         try:
#             image = Image.open(image_path).convert("RGB")
#             size = (224, 224)
#             image = ImageOps.fit(image, size, Image.LANCZOS)
#             image_array = np.asarray(image, dtype=np.float32) / 255.0  # 학습 시와 동일하게 정규화
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

#     def extract_features_from_image(self, image):
#         """PIL 이미지 객체에서 특징 추출"""
#         try:
#             image = image.convert("RGB")
#             size = (224, 224)
#             image = ImageOps.fit(image, size, Image.LANCZOS)
#             image_array = np.asarray(image, dtype=np.float32) / 255.0  # 학습 시와 동일하게 정규화
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

#     def predict_image(self, image_path):
#         """파일 경로에서 이미지 예측"""
#         try:
#             current_features = self.extract_features(image_path)
#             output_scale, output_zero_point = self.output_details[0]['quantization']
#             output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale  # 역양자화
#             stroke_probability = float(output_data_float[0])  # sigmoid 출력 (0~1)
#             non_stroke_probability = 1.0 - stroke_probability
#             severity_score = stroke_probability  # 두 번째 코드의 정의 유지
#             class_name = self.class_names[0] if stroke_probability > 0.5 else self.class_names[1]  # "질환군" or "비질환군"
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

#     def predict_image_from_image(self, image):
#         """PIL 이미지 객체에서 예측"""
#         try:
#             current_features = self.extract_features_from_image(image)
#             output_scale, output_zero_point = self.output_details[0]['quantization']
#             output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale  # 역양자화
#             stroke_probability = float(output_data_float[0])  # sigmoid 출력 (0~1)
#             non_stroke_probability = 1.0 - stroke_probability
#             severity_score = int(round(abs(stroke_probability) * 100))  # 첫 번째 코드의 정의 유지
#             class_name = self.class_names[0] if stroke_probability > 0.5 else self.class_names[1]  # "질환군" or "비질환군"
#             return {
#                 "filename": "unknown",
#                 "class": class_name,
#                 "stroke_probability": stroke_probability,
#                 "non_stroke_probability": non_stroke_probability,
#                 "severity_score": severity_score
#             }
#         except Exception as e:
#             return {
#                 "filename": "unknown",
#                 "error": str(e)
#             }

# predictor = StrokePredictor(os.path.join(os.path.split(__file__)[0], "model.tflite"), temperature=0.1)

# @app.route('/api/ai_send', methods=['POST'])
# def ai_send():
#     try:
#         if 'image' in request.files:
#             file = request.files['image']
#             if file.filename == '' or not file:
#                 return jsonify({"status": "error", "message": "Invalid image file"}), 400
#             image_data = file.read()
#         elif request.is_json:
#             data = request.get_json()
#             if 'image' in data:
#                 try:
#                     image_data = base64.b64decode(data['image'])
#                 except Exception as e:
#                     return jsonify({"status": "error", "message": "Invalid base64 encoding"}), 400
#             else:
#                 return jsonify({"status": "error", "message": "No image data provided in JSON"}), 400
#         else:
#             return jsonify({"status": "error", "message": "Invalid request: no image file or JSON data provided"}), 400
        
#         try:
#             image = Image.open(BytesIO(image_data))
#         except Exception as e:
#             return jsonify({"status": "error", "message": "Failed to open image: " + str(e)}), 400
        
#         result = predictor.predict_image_from_image(image)
#         if "error" in result:
#             return jsonify({"status": "error", "message": result["error"]}), 500
#         else:
#             return jsonify({"status": "success", "result": result}), 200
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)