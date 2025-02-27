from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

class StrokePredictor:
    def __init__(self, base_image_path):
        model_path = os.path.join(os.path.dirname(__file__), "keras_Model.h5")
        labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")

        base_model = load_model(model_path, compile=False)
        self.model = base_model
        self.class_names = open(labels_path, "r").readlines()

        self.base_features = self.extract_features(base_image_path)

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

            difference = np.abs(self.base_features - current_features)
            severity_score = np.mean(difference)

            index = np.argmax(current_features)
            class_name = self.class_names[index][2:]
            confidence_score = current_features[0][index]

            return {
                "filename": os.path.basename(image_path),
                "class": class_name.strip(),
                "confidence": float(confidence_score),
                "severity_score": float(severity_score)
            }

        except Exception as e:
            return {
                "filename": os.path.basename(image_path),
                "error": str(e)
            }


def process_folder(folder_path, base_image_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    results = []
    predictor = StrokePredictor(base_image_path)

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
                    print(f"변화 정도(심각성 점수): {severity_percentage:.1f}%")
                    print("-" * 50)

        return results

    except Exception as e:
        print(f"폴더 처리 중 오류 발생: {str(e)}")
        return None


def main():
    folder_path = "/Users/harold0812/Desktop/2025 2G/AI project (myspring)/Myspring_AI_Develop/Annotated stroke and non stroke Dataset/NonStroke"  # 저장한 이미지 데이터 경로
    base_image_path = "/Users/harold0812/Desktop/2025 2G/AI project (myspring)/Myspring_AI_Develop/Annotated stroke and non stroke Dataset/Stroke/img_0023.jpg"  # 판단을 결정할 이미지 데이터 경로

    print(f"처리할 폴더: {folder_path}")
    print(f"베이스 이미지: {base_image_path}")
    print("=" * 50)

    results = process_folder(folder_path, base_image_path)

    if results:
        print("\n처리 완료 통계:")
        print(f"총 처리된 이미지: {len(results)}개")
        successful = sum(1 for r in results if "error" not in r)
        print(f"성공: {successful}개")
        print(f"실패: {len(results) - successful}개")

if __name__ == "__main__":
    main()