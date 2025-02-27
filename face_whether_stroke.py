# Face_Whether_Stroke.py

from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

class StrokePredictor:
    def __init__(self, base_image_path):
        self.base_image_path = base_image_path

        model_path = os.path.join(os.path.dirname(__file__), "keras_Model.h5")
        labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")

        self.model = load_model(model_path, compile=False)
        self.class_names = open(labels_path, "r").readlines()

    def extract_features(self, image_path):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)

        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        features = self.model.predict(data)
        return features

    def predict_image(self, image_path):
        try:
            current_features = self.extract_features(image_path)

            # 가장 높은 확률을 가진 클래스 선택
            index = np.argmax(current_features)
            class_name = self.class_names[index].strip()
            confidence_score = current_features[0][index]

            # 심각성 점수를 신뢰도 기반으로 정의
            severity_score = 1.0 - confidence_score

            return {
                "filename": os.path.basename(image_path),
                "class": class_name,
                "confidence": float(confidence_score),
                "severity_score": float(severity_score)
            }

        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }

def process_folder(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    results = []
    predictor = StrokePredictor()

    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                result = predictor.predict_image(image_path)
                results.append(result)

                if "error" in result:
                    print(f"파일 {result['filename']}: 처리 중 오류 발생 - {result['error']}")
                else:
                    confidence_percentage = result['confidence'] * 100
                    severity_percentage = result['severity_score'] * 100

                    print(f"파일: {result['filename']}")
                    print(f"클래스: {result['class']}")
                    print(f"신뢰도: {confidence_percentage:.1f}%")
                    print(f"심각성 점수: {severity_percentage:.1f}%")
                    print("-" * 50)

        return results

    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {str(e)}")
        return None

def main():
    folder_path = input("이미지 폴더의 경로를 입력하세요: ") # 이미지 데이터 경로

    print(f"처리할 폴더: {folder_path}")
    print("=" * 50)

    results = process_folder(folder_path)

    if results:
        print("\n처리 완료 통계:")
        print(f"총 처리된 이미지: {len(results)}개")
        successful = sum(1 for r in results if "error" not in r)
        print(f"성공: {successful}개")
        print(f"실패: {len(results) - successful}개")

if __name__ == "__main__":
    main()