from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
import os
import logging
from flask_cors import CORS  # CORS 처리를 위한 라이브러리 추가

# GPU 비활성화 및 TensorFlow 로그 최소화
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

# TensorFlow 메모리 사용량 제한 (필요 시 활성화)
# tf.config.set_logical_device_configuration(
#     tf.config.list_physical_devices('CPU')[0],
#     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]  # 1GB 제한
# )

# 상대 경로 대신 절대 경로 사용 (기존 코드 유지)
from Face_Stroke.Face_Whether_Stroke import StrokePredictor as ImageStrokePredictor
from Pose_Stroke.Arm_Whether_Stroke import StrokePredictor as PoseStrokePredictor

app = Flask(__name__)

# CORS 설정 간소화
CORS(app, resources={r"/*": {"origins": "*"}})

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # DEBUG로 변경 가능
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# 모델 파일 경로
FACE_MODEL_PATH = os.path.join(BASE_DIR, "Face_Stroke/model.tflite")
FACE_LABEL_PATH = os.path.join(BASE_DIR, "Face_Stroke/label.txt")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "Pose_Stroke/pose_model.tflite")
POSE_LABEL_PATH = os.path.join(BASE_DIR, "Pose_Stroke/pose_label.txt")

# 파일 존재 여부 확인
try:
    check_file_exists(FACE_MODEL_PATH)
    check_file_exists(FACE_LABEL_PATH)
    check_file_exists(POSE_MODEL_PATH)
    check_file_exists(POSE_LABEL_PATH)
except FileNotFoundError as e:
    logging.error(str(e))
    raise

# 서버 시작 시 모델 1회 로드 (기존 코드 유지)
image_model = ImageStrokePredictor(FACE_MODEL_PATH, FACE_LABEL_PATH, temperature=0.5)
pose_model = PoseStrokePredictor(POSE_MODEL_PATH, POSE_LABEL_PATH, temperature=0.1)

def load_image_from_request(req):
    try:
        if 'image' in req.files:
            img = Image.open(req.files['image'])
        elif req.is_json:
            data = req.get_json()
            img = Image.open(BytesIO(base64.b64decode(data['image'])))
        else:
            img = Image.open(BytesIO(req.data))
        
        # 지원 이미지 형식 확인
        if img.format not in ['JPEG', 'PNG']:
            raise ValueError(f"Unsupported image format: {img.format}. Only JPEG and PNG are supported.")
        
        # 메모리 절약을 위해 이미지 리사이징 (필요 시)
        # 모델 입력 크기가 224x224이므로 불필요한 경우 생략 가능
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {str(e)}")
    finally:
        # 요청 데이터 정리
        if 'image' in req.files:
            req.files['image'].close()

@app.route("/Face_Stroke/ai_send", methods=["POST"])
def face_predict():
    try:
        image = load_image_from_request(request)
        result = image_model.predict(image)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        logging.error(f"Error in face_predict: {str(e)}")
        return jsonify({"status": "error", "message": "내부 서버 오류가 발생했습니다."}), 500
    finally:
        # 이미지 객체 명시적 제거
        if 'image' in locals():
            image.close()
            del image

@app.route("/Pose_Stroke/ai_send", methods=["POST"])
def pose_predict():
    try:
        image = load_image_from_request(request)
        result = pose_model.predict(image)
        return jsonify({"status": "success", "result": result["severity_score"]})
    except Exception as e:
        logging.error(f"Error in pose_predict: {str(e)}")
        return jsonify({"status": "error", "message": "내부 서버 오류가 발생했습니다."}), 500
    finally:
        # 이미지 객체 명시적 제거
        if 'image' in locals():
            image.close()
            del image