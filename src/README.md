train_model.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# Load the trained model
model = keras.models.load_model('cifar_model.keras')

# Load data for calibration
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_test = y_test.flatten()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# FP32 accuracy (baseline)
loss, fp32_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"FP32 Accuracy: {fp32_acc:.4f}")

# Convert to FP32 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
with open('model_fp32.tflite', 'wb') as f:
    f.write(tflite_fp32)

# Convert to INT8 TFLite (PTQ)
def representative_dataset():
    for i in range(200):
        yield [x_train[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
tflite_int8 = converter_int8.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_int8)

# Compare sizes
fp32_size = os.path.getsize('model_fp32.tflite')
int8_size = os.path.getsize('model_int8.tflite')
print(f"FP32 size: {fp32_size} bytes")
print(f"INT8 size: {int8_size} bytes")
print(f"Compression: {fp32_size/int8_size:.2f}x")

# Evaluate INT8 accuracy
interpreter = tf.lite.Interpreter(model_path='model_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
for i in range(len(x_test)):
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    if output.argmax() == y_test[i]:
        correct += 1

int8_acc = correct / len(x_test)
print(f"INT8 Accuracy: {int8_acc:.4f}")

# Save results to a file for README
with open('results.txt', 'w') as f:
    f.write(f"FP32 Accuracy: {fp32_acc:.4f}\n")
    f.write(f"INT8 Accuracy: {int8_acc:.4f}\n")
    f.write(f"FP32 Size: {fp32_size} bytes\n")
    f.write(f"INT8 Size: {int8_size} bytes\n")
    f.write(f"Compression: {fp32_size/int8_size:.2f}x\n")
print("Results saved to results.txt")
