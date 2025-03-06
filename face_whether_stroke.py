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
        with open(labels_path, "r", encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded labels: {self.class_names}")

    def extract_features(self, image_path):
        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.LANCZOS)
            image_array = np.asarray(image) / 255.0  # 0~1로 정규화
            print(f"Image loaded: {image_path}, shape: {image_array.shape}")

            # 양자화된 입력 처리
            input_scale, input_zero_point = self.input_details[0]['quantization']
            print(f"Input scale: {input_scale}, Zero point: {input_zero_point}")
            if input_scale > 0:  # 양자화가 적용된 경우
                uint8_image_array = ((image_array / input_scale) + input_zero_point).clip(0, 255).astype(np.uint8)
            else:  # 양자화가 없는 경우
                uint8_image_array = (image_array * 255).astype(np.uint8)
            input_data = np.expand_dims(uint8_image_array, axis=0)
            print(f"Input data prepared: shape: {input_data.shape}, dtype: {input_data.dtype}")

            # 모델 추론 실행
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
            print(f"Output scale: {output_scale}, Zero point: {output_zero_point}")
            output_data_float = (current_features.astype(np.float32) - output_zero_point) * output_scale
            print(f"Raw output (after dequantization): {output_data_float}")
            print(f"Sum of raw output: {np.sum(output_data_float)}")

            # 모델 출력이 로짓인지 확인 후 처리
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:  # 합이 1에 가까우면 확률로 간주
                original_probabilities = output_data_float[0]
            else:  # 로짓이라면 softmax 적용
                original_probabilities = tf.nn.softmax(output_data_float[0]).numpy()
            print(f"Original probabilities: {original_probabilities}")
            print(f"Sum of original probabilities: {np.sum(original_probabilities)}")

            # 확률 조정을 위한 Softmax 온도 적용 (T=2.0으로 설정)
            T = 2.0  # 온도 파라미터: 더 자연스러운 비율을 위해 조정 가능
            if np.abs(np.sum(output_data_float) - 1.0) < 0.1:
                # 확률인 경우, 로짓으로 변환 후 온도 적용
                p0, p1 = original_probabilities
                if p1 == 0:
                    logit0 = float('inf')
                else:
                    logit0 = np.log(p0 / p1)
                logit1 = 0
                new_logit0 = logit0 / T
                new_logit1 = 0 / T
                adjusted_probabilities = [np.exp(new_logit0) / (np.exp(new_logit0) + np.exp(new_logit1)), 
                                      np.exp(new_logit1) / (np.exp(new_logit0) + np.exp(new_logit1))]
            else:
                # 로짓인 경우, 온도 적용 후 softmax
                logits = output_data_float[0]
                adjusted_probabilities = tf.nn.softmax(logits / T).numpy()

            print(f"Adjusted probabilities (with T={T}): {adjusted_probabilities}")
            print(f"Sum of adjusted probabilities: {np.sum(adjusted_probabilities)}")

            if len(original_probabilities) != 2:
                raise ValueError(f"모델 출력이 이진 분류 형식이 아님. 출력 형상: {output_data_float.shape}")

            # 확률 계산 (출력용으로 조정된 확률 사용)
            stroke_probability = float(adjusted_probabilities[0])  # 질환군 확률
            non_stroke_probability = float(adjusted_probabilities[1])  # 비질환군 확률

            # 클래스 결정 (원래 확률을 사용해 진단 결과 유지)
            class_name = self.class_names[1] if original_probabilities[1] > original_probabilities[0] else self.class_names[0]
            severity_score = stroke_probability  # 출력용 확률 사용

            # 디버깅 로그
            print(f"Stroke probability: {stroke_probability:.4f}, Non-stroke probability: {non_stroke_probability:.4f}, Class: {class_name}")

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

# 나머지 함수(process_folder, send_to_backend, main)는 동일하게 유지
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
                    print(f"질환 (Stroke) 확률: {round(result['stroke_probability'] * 100):d}%")
                    print(f"비질환 (Non-Stroke) 확률: {round(result['non_stroke_probability'] * 100):d}%")
                    print(f"심각도 점수: {round(result['severity_score'] * 100):d}%")
                    print("-" * 50)
        return results
    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {str(e)}")
        return None

def send_to_backend(results, backend_url):
    try:
        # 심각도 점수만 추출하고 소수점 없이 반올림
        severity_scores = [round(result["severity_score"] * 100) for result in results]
        # JSON으로 변환
        data = json.dumps(severity_scores, ensure_ascii=False)
        response = requests.post(backend_url, data=data, headers={"Content-Type": "application/json; charset=utf-8"})
        response.raise_for_status()
        print(f"데이터가 백엔드에 성공적으로 전송됨. 상태 코드: {response.status_code}")
        if response.text:
            print(f"백엔드 응답: {response.text}")
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

        backend_url = "http://localhost:8000"
        send_prompt = input("\n예측 결과를 백엔드에 전송하시겠습니까? (yes/no): ")
        if send_prompt.lower() == "yes":
            send_to_backend(results, backend_url)
        else:
            print("백엔드에 데이터를 전송하지 않습니다.")
    else:
        print("처리 결과가 없거나 오류가 발생했습니다.")

if __name__ == "__main__":
    main()