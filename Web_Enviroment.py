from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import requests
from Face_Whether_Stroke import StrokePredictor

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

BASE_IMAGE_PATH = os.path.join(app.root_path, "Annotated stroke and non stroke Dataset/Stroke/img_0023.jpg") # 기준이 되는 이미지 파일 경로

predictor = StrokePredictor(BASE_IMAGE_PATH)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/")
def home():
    return render_template('Upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('Upload.html')
    elif request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            result = predictor.predict_image(image_path)
            result['filename'] = filename

            try:
                result = predictor.predict_image(image_path)
                
                # 예측 결과를 Spring 백엔드로 전송
                send_data_to_spring_backend(result)
                
                result['confidence'] = f"{result['confidence'] * 100:.1f}%"
                result['severity_score'] = f"{result['severity_score'] * 100:.1f}%"
                
                return render_template('Result.html', result=result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)
        else:
            return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400


def send_data_to_spring_backend(result):
    url = "http://localhost:8080/receive-data"  # Spring 백엔드의 엔드포인트 URL
    headers = {'Content-Type': 'application/json'}
    
    data = {
        "filename": result.get("filename"),
        "className": result.get("class"),
        "confidence": result.get("confidence"),
        "severityScore": result.get("severity_score")
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        print("데이터 전송 성공:", response.text)
    except requests.exceptions.RequestException as e:
        print("데이터 전송 실패:", e)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)