from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Upload.html')

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = os.path.join(app.root_path, 'model.tflite')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "File not uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(file_path).resize((224, 224))
        img_array = np.asarray(img)
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]['quantization']
            if input_scale == 0.0:
                input_scale, input_zero_point = 1.0 / 255.0, 0
            img = (img_array / input_scale + input_zero_point).clip(0, 255).astype(np.uint8)
        else:
            img = img_array.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if output_data.shape[-1] == 2:
            probabilities = output_data[0]
            stroke_probability = float(probabilities[0])
            non_stroke_probability = float(probabilities[1])
        elif output_data.size == 1:
            stroke_probability = float(output_data[0])
            non_stroke_probability = 1 - stroke_probability
        else:
            raise ValueError("Unexpected output shape")

        classification = "stroke" if stroke_probability > non_stroke_probability else "non_stroke"
        severity_score = stroke_probability if classification == "stroke" else non_stroke_probability

        result = {
            "filename": filename,
            "class": classification,
            "stroke_probability": stroke_probability,
            "non_stroke_probability": non_stroke_probability,
            "severity_score": severity_score
        }

        result['stroke_probability'] = f"{result['stroke_probability'] * 100:.1f}%"
        result['non_stroke_probability'] = f"{result['non_stroke_probability'] * 100:.1f}%"
        result['severity_score'] = f"{result['severity_score'] * 100:.1f}%"
        result['image_path'] = os.path.join('uploads', filename)

        return render_template('Result.html', result=result)
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)