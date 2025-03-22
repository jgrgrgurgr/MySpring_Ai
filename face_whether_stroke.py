import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import json
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import traceback

app = Flask(__name__)

class StrokePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

        labels_path = os.path.join(os.path.split(__file__)[0], "labels.txt")
        with open(labels_path, "r", encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image) / 255.0
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
            print(f"Error in extract_features for {image_path}: {e}")
            raise

    def predict_image(self, image_path):
        try:
            current_features = self.extract_features(image_path)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:
                original_probabilities = output_data_float[0]
            else:
                original_probabilities = tf.nn.softmax(output_data_float[0]).numpy()
            stroke_probability = float(original_probabilities[0])
            non_stroke_probability = float(original_probabilities[1])
            severity_score = stroke_probability
            class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
            return {
                "filename": os.path.basename(image_path),
                "class": class_name,
                "stroke_probability": stroke_probability,
                "non_stroke_probability": non_stroke_probability,
                "severity_score": severity_score
            }
        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }

    def extract_features_from_image(self, image):
        try:
            image = image.convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image) / 255.0
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
            print(f"Error in extract_features_from_image: {e}")
            raise

    def predict_image_from_image(self, image):
        try:
            current_features = self.extract_features_from_image(image)
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:
                original_probabilities = output_data_float[0]
            else:
                original_probabilities = tf.nn.softmax(output_data_float[0]).numpy()
            stroke_probability = float(original_probabilities[0])
            non_stroke_probability = float(original_probabilities[1])
            severity_score = stroke_probability
            class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
            return {
                "filename": "unknown",
                "class": class_name,
                "stroke_probability": stroke_probability,
                "non_stroke_probability": non_stroke_probability,
                "severity_score": severity_score
            }
        except Exception as e:
            return {
                "filename": "unknown",
                "error": str(e)
            }

def process_folder(folder_path):
    global predictor
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    results = []
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                result = predictor.predict_image(image_path)
                results.append(result)
                if "error" in result:
                    print(f"File {result['filename']}: Error during processing - {result['error']}")
                else:
                    print(f"File path: {image_path}")
                    print(f"Classification result: {result['class']}")
                    print(f"Stroke probability: {round(result['stroke_probability'] * 100)}%")
                    print(f"Non-stroke probability: {round(result['non_stroke_probability'] * 100)}%")
                    print(f"Severity score: {round(result['severity_score'] * 100)}%")
                    print("-" * 50)
        return results
    except Exception as e:
        print(f"Error processing folder: {str(e)}")
        return None

def send_to_backend(results, backend_url):
    try:
        severity_scores = [round(result["severity_score"] * 100) for result in results]
        data = json.dumps({"scores": severity_scores}, ensure_ascii=False)
        print("Sending data:", data)
        response = requests.post(backend_url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
        print("Response status:", response.status_code)
        print("Response text:", response.text)
        response.raise_for_status()
        print("Data sent to backend successfully. Status code:", response.status_code)
        if response.text:
            print("backend response:", response.text)
        return response
    except requests.exceptions.RequestException as e:
        print("Error sending to backend:", e)
        if isinstance(e, requests.exceptions.HTTPError):
            print("HTTP error:", e.response.status_code, e.response.text)
        return None

def main():
    folder_path = input("Image folder path: ")
    print("Processing folder:", folder_path)
    print("=" * 50)
    results = process_folder(folder_path)
    if results:
        print("\nProcessing complete statistics:")
        print("Total processed images:", len(results))
        successful = sum(1 for r in results if "error" not in r)
        print("Success:", successful)
        print("Failure:", len(results) - successful)
        backend_url = input("Enter backend URL to send results (or leave empty to skip): ")
        if backend_url:
            send_to_backend(results, backend_url)
        else:
            print("Not sending to backend.")
    else:
        print("No results or error occurred.")

predictor = StrokePredictor(os.path.join(os.path.split(__file__)[0], "model.tflite"))

# @app.route('/api/ai_send', methods=['POST'])
# def ai_send():
#     try:
#         # Enhanced debugging output
#         print("Headers:", dict(request.headers))
#         print("Form Data:", dict(request.form))
#         print("Files:", dict(request.files))
#         print("Content-Type:", request.content_type)
#         print("All request files keys:", list(request.files.keys()))

#         # 파일이 있는지 확인
#         if 'image' not in request.files:
#             print("Error: 'image' key not found in request.files")
#             return jsonify({"status": "error", "message": "No image file provided"}), 400
        
#         file = request.files['image']
#         if file.filename == '' or not file:
#             print("Error: Invalid or empty file received")
#             return jsonify({"status": "error", "message": "Invalid image file"}), 400
        
#         # 확장자 확인
#         filename = file.filename
#         basename, extension = os.path.splitext(filename)
#         if extension.lower() not in [".jpg", ".jpeg"]:
#             print(f"Error: Invalid file extension {extension}, must be .jpg or .jpeg")
#             return jsonify({"status": "error", "message": "File must be a JPEG image"}), 400
        
#         # JPEG 파일인지 확인
#         image_data = file.read()
#         if not image_data.startswith(b'\xFF\xD8'):
#             print("Error: File is not a valid JPEG image")
#             return jsonify({"status": "error", "message": "Not a valid JPEG image"}), 400
        
#         image = Image.open(BytesIO(image_data))
#         result = predictor.predict_image_from_image(image)
#         if "error" in result:
#             print(f"Prediction error: {result['error']}")
#             return jsonify({"status": "error", "message": result["error"]}), 500
#         else:
#             results_list = [result]
#             response = send_to_backend(results_list, "http://192.168.9.29:8080/face/receive")
#             if response:
#                 print("Backend response successful")
#                 return jsonify({"status": "success"}), 200
#             else:
#                 print("Failed to send to backend")
#                 return jsonify({"status": "error", "message": "Failed to send to backend"}), 500
#     except Exception as e:
#         traceback.print_exc()
#         print(f"Unexpected error: {str(e)}")
#         return jsonify({"status": "error", "message": f"Invalid image file: {str(e)}"}), 500

@app.route('/api/ai_send', methods=['POST'])
def ai_send():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "No image file provided"}), 400
        file = request.files['image']
        if file.filename == '' or not file:
            return jsonify({"status": "error", "message": "Invalid image file"}), 400
        image_data = file.read()
        image = Image.open(BytesIO(image_data))
        result = predictor.predict_image_from_image(image)
        if "error" in result:
            print(f"Prediction error: {result['error']}")
            return jsonify({"status": "error", "message": result["error"]}), 500
        else:
            return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        traceback.print_exc()
        print(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": f"Invalid image file: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)