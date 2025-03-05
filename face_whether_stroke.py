import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import json

class StrokePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

        # 레이블 파일 로드
        labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")
        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features(self, image_path):
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image)
            print(f"Image loaded: {image_path}, shape: {image_array.shape}")

            # 양자화된 모델에 맞춘 입력 처리
            input_scale, input_zero_point = self.input_details[0]['quantization']
            print(f"Input scale: {input_scale}, Zero point: {input_zero_point}")
            if input_scale == 0.0:
                # 양자화되지 않은 모델의 경우 기본값 사용
                input_scale, input_zero_point = 1.0 / 255.0, 0
            uint8_image_array = (image_array / input_scale + input_zero_point).clip(0, 255).astype(np.uint8)
            input_data = np.expand_dims(uint8_image_array, axis=0)
            print(f"Input data prepared: shape: {input_data.shape}, dtype: {input_data.dtype}")

            # 텐서 설정 및 추론 실행
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(f"Output data: {output_data}")
            return output_data
        except Exception as e:
            print(f"Error in extract_features for {image_path}: {e}")
            raise
        
    def predict_image(self, image_path):
        try:
            current_features = self.extract_features(image_path)

            # 양자화된 출력 처리
            output_scale, output_zero_point = self.output_details[0]['quantization']
            if output_scale == 0.0:
                output_scale, output_zero_point = 1.0 / 255.0, -128
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            print(f"Output shape: {output_data_float.shape}, values: {output_data_float}")

            # 이진 분류 모델 출력 처리
            if output_data_float.shape[-1] == 2:
                probabilities = output_data_float[0]
            else:
                probabilities = tf.nn.softmax(output_data_float).numpy()[0]
                if len(probabilities) < 2:
                    raise ValueError(f"모델 출력이 이진 분류 형식이 아님. 출력 형상: {output_data_float.shape}")

            stroke_probability = float(probabilities[1])
            non_stroke_probability = float(probabilities[0])
            class_name = self.class_names[0] if stroke_probability > non_stroke_probability else self.class_names[1]
            severity_score = stroke_probability
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

def process_folder(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    results = []
    predictor = StrokePredictor(os.path.join(os.path.dirname(__file__), "model.tflite"))

    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                result = predictor.predict_image(image_path)
                results.append(result)

                if "error" in result:
                    print(f"파일 {result['filename']}: 처리 중 오류 발생 - {result['error']}")
                else:
                    print(f"파일 경로: {image_path}")
                    print(f"분류 결과: {result['class']}")
                    print(f"질환 (Stroke) 확률: {result['stroke_probability'] * 100:.1f}%")
                    print(f"비질환 (Non-Stroke) 확률: {result['non_stroke_probability'] * 100:.1f}%")
                    print(f"심각도 점수: {result['severity_score'] * 100:.1f}%")
                    print("-" * 50)
        return results
    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {str(e)}")
        return None

def send_to_backend(results, backend_url):
    try:
        # 결과를 JSON 형식으로 변환
        data = json.dumps(results)
        response = requests.post(backend_url, json=data)
        response.raise_for_status()
        print(f"데이터가 백엔드에 성공적으로 전송됨. 상태 코드: {response.status_code}")
        if response.text:
            print(f"백엔드 응답: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"백엔드 전송 중 오류 발생: {e}")

def send_to_backend(results, backend_url):
    try:
        data = json.dumps(results, ensure_ascii=False)  # UTF-8 인코딩 보장
        response = requests.post(backend_url, json=data)
        response.raise_for_status()
        print(f"데이터가 백엔드에 성공적으로 전송됨. 상태 코드: {response.status_code}")
        if response.text:
            try:
                response_data = json.loads(response.text)
                print("백엔드에서 반환된 데이터:")
                for item in response_data:
                    if "severity_score" in item:
                        percentage = item["severity_score"] * 100
                        print(f"파일 {item['filename']}의 심각도 점수: {percentage:.1f}%")
                    else:
                        print(f"파일 {item['filename']}: 심각도 점수 없음")
            except json.JSONDecodeError:
                print("백엔드 응답을 JSON으로 파싱할 수 없습니다.")
    except requests.exceptions.RequestException as e:
        print(f"백엔드 전송 중 오류 발생: {e}")

def main():
    folder_path = input("이미지 폴더의 경로를 입력하세요: ")
    print(f"처리할 폴더: {folder_path}")
    print("=" * 50)
    results = process_folder(folder_path)
    if results:
        print("\n처리 완료 통계:")
        print(f"총 처리된 이미지: {len(results)}개")
        successful = sum(1 for r in results if "error" not in r)
        print(f"성공: {successful}개")
        print(f"실패: {len(results) - successful}개")

        backend_url = "http://localhost:8000"  # 여기에 링크 삽입
        send_prompt = input("\n예측 결과를 백엔드에 전송하시겠습니까? (yes/no): ")
        if send_prompt.lower() == "yes":
            send_to_backend(results, backend_url)
        else:
            print("백엔드에 데이터를 전송하지 않습니다.")
    else:
        print("처리 결과가 없거나 오류가 발생했습니다.")

if __name__ == "__main__":
    main()