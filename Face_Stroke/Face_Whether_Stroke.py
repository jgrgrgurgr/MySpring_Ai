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

tf.config.threading.set_inter_op_parallelism_threads(2)  # CPU 스레드 제한
tf.config.threading.set_intra_op_parallelism_threads(2)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set max content length to 16MB

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU') # GPU 사용 비활성화

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
            severity_score = int(round(stroke_probability * 100))
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

@app.route('/face/ai_send', methods=['POST'])
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