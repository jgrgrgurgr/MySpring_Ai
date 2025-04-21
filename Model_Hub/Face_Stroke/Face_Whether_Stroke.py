import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

class StrokePredictor:
    def __init__(self, model_path, label_path, temperature=1.0):
        self.temperature = temperature
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(label_path, "r", encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, image):
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = image.astype(np.float32)
        return np.expand_dims(image, axis=0)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image):
        input_data = self.preprocess(image)
        if self.input_details[0]['dtype'] == np.uint8:
            input_data = (input_data * 255).astype(np.uint8)
        else:
            input_data = input_data.astype(self.input_details[0]['dtype'])

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prediction = self.softmax(output / self.temperature)
        return {self.labels[i]: float(prediction[i]) for i in range(len(self.labels))}