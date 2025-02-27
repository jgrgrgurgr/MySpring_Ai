from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
from Face_Whether_Stroke import StrokePredictor

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

BASE_IMAGE_PATH = os.path.join(app.root_path, "Annotated stroke and non stroke Dataset/Stroke/img_0023.jpg")  # 기준이 되는 이미지 파일명

predictor = StrokePredictor(BASE_IMAGE_PATH)

# 업로드 폴더 생성
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

            try:
                # 이미지 예측
                result = predictor.predict_image(image_path)

                # 신뢰도와 변화 정도를 백분율로 변환하고 소수점 한 자리까지로 포맷팅
                result['confidence'] = f"{result['confidence'] * 100:.1f}%"
                result['severity_score'] = f"{result['severity_score'] * 100:.1f}%"

                # 결과를 템플릿에 전달
                return render_template('Result.html', result=result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                # 이미지를 삭제하지 않습니다.
                pass
        else:
            return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)