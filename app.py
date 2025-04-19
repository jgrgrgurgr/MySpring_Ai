from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import os
import traceback

from image_model import ImageStrokePredictor
from pose_model import PoseStrokePredictor

app = Flask(__name__)

# CORS 헤더를 모든 응답에 추가
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    return response

image_model = ImageStrokePredictor("model.tflite", "label.txt", temperature=0.5)
pose_model = PoseStrokePredictor("pose_model.tflite", "pose_labels.txt", temperature=0.1)

def load_image_from_request(req):
    try:
        if 'image' in req.files:
            return Image.open(req.files['image'])
        elif req.is_json:
            data = req.get_json()
            return Image.open(BytesIO(base64.b64decode(data['image'])))
        else:
            return Image.open(BytesIO(req.data))
    except Exception as e:
        raise ValueError(f"Invalid image input: {e}")

@app.route("/Face_Stroke/ai_send", methods=["POST"])
def face_predict():
    try:
        image = load_image_from_request(request)
        result = image_model.predict(image)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/Pose_Stroke/ai_send", methods=["POST"])
def pose_predict():
    try:
        image = load_image_from_request(request)
        result = pose_model.predict(image)
        return jsonify({"status": "success", "result": result["severity_score"]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)