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

tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

class StrokePredictor:
    def __init__(self, model_path, label_path, temperature=1.0):
        self.temperature = temperature
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            with open(label_path, "r", encoding='utf-8') as f:
                self.labels = [line.strip() for line in f.readlines()]
            logger.info("Model and labels loaded. Classes: %s", self.labels)
        except Exception as e:
            logger.error("Failed to load model or labels: %s", str(e))
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return np.expand_dims(image, axis=0)

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
        return {self.labels[i]: float(prediction[i]) for i in range(len(self.labels))}

try:
    predictor = StrokePredictor(
        os.path.join(os.path.dirname(__file__), "model.tflite"),
        os.path.join(os.path.dirname(__file__), "label.txt"),
        temperature=1.0  # 필요하면 조절
    )
except Exception as e:
    logger.critical("Failed to initialize predictor: %s", str(e))
    raise

@app.route('/face/ai_send', methods=['POST'])
def ai_send():
    try:
        logger.info("Received request: %s", request.headers)
        # 멀티파트 폼 데이터
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
        result = predictor.predict(image)
        logger.info("Prediction result: %s", result)
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        logger.error("Unexpected error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
