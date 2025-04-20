# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

# 수정된 임포트: 상대 경로 대신 절대 경로 사용
from Face_Stroke.Face_Whether_Stroke import StrokePredictor as ImageStrokePredictor
from Pose_Stroke.Arm_Whether_Stroke import StrokePredictor as PoseStrokePredictor

app = Flask(__name__)

# 로깅 설정
logging.basicConfig(
    level=logging.ERROR,
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

FACE_MODEL_PATH = os.path.join(BASE_DIR, "Face_Stroke/model.tflite")
FACE_LABEL_PATH = os.path.join(BASE_DIR, "Face_Stroke/label.txt")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "Pose_Stroke/pose_model.tflite")
POSE_LABEL_PATH = os.path.join(BASE_DIR, "Pose_Stroke/pose_labels.txt")

try:
    check_file_exists(FACE_MODEL_PATH)
    check_file_exists(FACE_LABEL_PATH)
    check_file_exists(POSE_MODEL_PATH)
    check_file_exists(POSE_LABEL_PATH)
except FileNotFoundError as e:
    logging.error(str(e))
    raise

# 서버 시작 시 모델 1회 로드
image_model = ImageStrokePredictor(FACE_MODEL_PATH, FACE_LABEL_PATH, temperature=0.5)
pose_model = PoseStrokePredictor(POSE_MODEL_PATH, POSE_LABEL_PATH, temperature=0.1)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

def load_image_from_request(req):
    try:
        if 'image' in req.files:
            img = Image.open(req.files['image'])
        elif req.is_json:
            data = req.get_json()
            img = Image.open(BytesIO(base64.b64decode(data['image'])))
        else:
            img = Image.open(BytesIO(req.data))
        
        if img.format not in ['JPEG', 'PNG']:
            raise ValueError(f"Unsupported image format: {img.format}. Only JPEG and PNG are supported.")
        
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {str(e)}")

@app.route("/Face_Stroke/ai_send", methods=["POST"])
def face_predict():
    try:
        image = load_image_from_request(request)
        result = image_model.predict(image)
        # 이미지 객체 명시적 제거
        del image
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"status": "error"}), 500

@app.route("/Pose_Stroke/ai_send", methods=["POST"])
def pose_predict():
    try:
        image = load_image_from_request(request)
        result = pose_model.predict(image)
        return jsonify({"status": "success", "result": result["severity_score"]})
    except Exception as e:
        logging.error(f"Error in pose_predict: {str(e)}")
        return jsonify({"status": "error", "message": "내부 서버 오류가 발생했습니다."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)