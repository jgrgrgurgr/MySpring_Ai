from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Upload.html')

# 업로드 폴더와 모델 경로 설정
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = os.path.join(app.root_path, 'model.tflite')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 레이블 파일 로드 (첫 번째 코드와 동일)
labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")
with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"Loaded labels: {class_names}")

@app.route("/predict", methods=["POST"])
def predict():
    # 파일 업로드 확인
    if 'file' not in request.files:
        return jsonify({"error": "File not uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # 파일 저장
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # TensorFlow Lite 인터프리터 초기화
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        image = Image.open(file_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)

        input_scale, input_zero_point = input_details[0]['quantization']
        if input_scale == 0.0:
            input_scale, input_zero_point = 1.0 / 255.0, 0
        uint8_image_array = (image_array / input_scale + input_zero_point).clip(0, 255).astype(np.uint8)
        input_data = np.expand_dims(uint8_image_array, axis=0)

        # 텐서 설정 및 추론 실행
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        output_scale, output_zero_point = output_details[0]['quantization']
        if output_scale == 0.0:
            output_scale, output_zero_point = 1.0 / 255.0, -128
        output_data_float = (output_data.astype(np.float32) - output_zero_point) * output_scale

        # 이진 분류 모델 출력 처리
        if output_data_float.shape[-1] == 2:
            probabilities = output_data_float[0]
        else:
            probabilities = tf.nn.softmax(output_data_float).numpy()[0]
            if len(probabilities) < 2:
                raise ValueError(f"모델 출력이 이진 분류 형식이 아님. 출력 형상: {output_data_float.shape}")

        stroke_probability = float(probabilities[1])
        non_stroke_probability = float(probabilities[0])

        class_name = class_names[0] if stroke_probability > non_stroke_probability else class_names[1]
        severity_score = stroke_probability

        # 결과 생성
        result = {
            "filename": filename,
            "class": class_name,
            "stroke_probability": f"{stroke_probability * 100:.1f}%",
            "non_stroke_probability": f"{non_stroke_probability * 100:.1f}%",
            "severity_score": f"{severity_score * 100:.1f}%",
            "image_path": os.path.join('uploads', filename)
        }

        return render_template('Result.html', result=result)
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)