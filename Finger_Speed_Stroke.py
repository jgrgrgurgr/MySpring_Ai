import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


tapping_intervals = np.array([...])
labels = np.array([...])


tapping_intervals = tapping_intervals / np.max(tapping_intervals)


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(tapping_intervals.shape[1],)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(tapping_intervals, labels, epochs=15, batch_size=16)


test_loss, test_acc = model.evaluate(tapping_intervals, labels)
print('테스트 정확도:', test_acc)