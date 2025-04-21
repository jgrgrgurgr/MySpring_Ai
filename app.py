from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
import os
import logging
from flask_cors import CORS

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# GPU 비활성화 및 TensorFlow 로그 최소화
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

@app.route('/healthz')
def healthz():
    return jsonify(status="ok"), 200

# CORS 설정 간소화
CORS(app, resources={r"/*": {"origins": "*"}})

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# 모델 파일 경로
FACE_MODEL_PATH = os.path.join(BASE_DIR, "Model_Hub", "Face_Stroke", "model.tflite")
FACE_LABEL_PATH = os.path.join(BASE_DIR, "Model_Hub", "Face_Stroke", "label.txt")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "Model_Hub", "Pose_Stroke", "pose_model.tflite")
POSE_LABEL_PATH = os.path.join(BASE_DIR, "Model_Hub", "Pose_Stroke", "pose_label.txt")

# 파일 존재 여부 확인
try:
    check_file_exists(FACE_MODEL_PATH)
    check_file_exists(FACE_LABEL_PATH)
    check_file_exists(POSE_MODEL_PATH)
    check_file_exists(POSE_LABEL_PATH)
except FileNotFoundError as e:
    logging.error(str(e))
    raise

# 모델 초기화 (지연 로딩을 위해 None으로 설정)
image_model = None
pose_model = None

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
        
        # 이미지 리사이징
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {str(e)}")
    finally:
        if 'image' in req.files:
            req.files['image'].close()

@app.route("/Face/ai_send", methods=["POST"])
def face_predict():
    global image_model
    try:
        # 모델이 로드되지 않았다면 로드
        if image_model is None:
            try:
                # 모델 관련 import를 함수 내부에서 처리
                from Model_Hub.Face_Stroke.Face_Whether_Stroke import StrokePredictor as ImageStrokePredictor
                logging.info("Loading face stroke model...")
                image_model = ImageStrokePredictor(FACE_MODEL_PATH, FACE_LABEL_PATH, temperature=0.5)
                logging.info("Face stroke model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load face model: {str(e)}")
                return jsonify({"status": "error", "message": "Failed to load face model."}), 500

        image = load_image_from_request(request)
        result = image_model.predict(image)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        logging.error(f"Error in face_predict: {str(e)}")
        return jsonify({"status": "error", "message": "내부 서버 오류가 발생했습니다."}), 500
    finally:
        if 'image' in locals():
            image.close()
            del image

@app.route("/ai_send", methods=["POST"])
def pose_predict():
    global pose_model
    try:
        # 모델이 로드되지 않았다면 로드
        if pose_model is None:
            try:
                # 모델 관련 import를 함수 내부에서 처리
                from Model_Hub.Pose_Stroke.Arm_Whether_Stroke import StrokePredictor as PoseStrokePredictor
                logging.info("Loading pose stroke model…")
                pose_model = PoseStrokePredictor(POSE_MODEL_PATH, POSE_LABEL_PATH, temperature=0.1)
                logging.info("Pose stroke model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load pose model: {str(e)}")
                return jsonify({"status": "error", "message": "Failed to load pose model."}), 500

        image = load_image_from_request(request)
        result = pose_model.predict(image)
        return jsonify({"status": "success", "result": result["severity_score"]})
    except Exception as e:
        logging.error(f"Error in pose_predict: {str(e)}")
        return jsonify({"status": "error", "message": "내부 서버 오류가 발생했습니다."}), 500
    finally:
        if 'image' in locals():
            image.close()
            del image

if __name__ == '__main__':
    app.run(debug=True)