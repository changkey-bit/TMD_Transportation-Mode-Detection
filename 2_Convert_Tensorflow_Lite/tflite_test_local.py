#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFLite 모델 로컬 실행 및 평가 스크립트
- CSV 형식의 테스트 데이터를 로드
- 지정된 .tflite 모델에 입력하여 예측 수행
- 정확도, 분류 리포트, 혼동 행렬 출력 및 저장

Usage:
    python tflite_evaluation.py \
        --model_path <model.tflite> \
        --x_test_csv <x_test.csv> \
        --y_test_csv <y_test.csv> \
        [--output_cm <confusion.png>]
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools


def plot_confusion_matrix(cm, classes, save_path, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    fmt = '.2f' if cm.dtype == float else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


def evaluate_tflite(model_path, x_test, y_true, class_names, output_cm):
    # Interpreter 로딩
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 데이터 형태 조정
    X = x_test.to_numpy(dtype=np.float32)
    num_samples, seq_len_cols = X.shape
    # assume last two columns are extras if more than 12 features
    feature_cols = seq_len_cols
    seq_len = 300
    features = feature_cols // seq_len
    X = X.reshape(num_samples, seq_len, features)

    y_pred = []
    for sample in X:
        # 샘플별 입력 분할 (sensor_num segments)
        # 입력 텐서별 순서: match converter order
        for detail in input_details:
            idx = detail['index']
            # calculate segment index from detail order
            seg = idx  # assume correspondence
            data = sample[:, seg*3:(seg+1)*3].reshape(1, seq_len, 3)
            interpreter.set_tensor(detail['index'], data)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(out, axis=-1)[0])

    y_pred = np.array(y_pred)
    y_true = y_true.to_numpy().ravel()

    # 정확도 계산
    accuracy = (y_pred == y_true).mean() * 100
    print(f"TFLite Model Accuracy: {accuracy:.2f}%")

    # 리포트
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plot_confusion_matrix(cm, class_names, output_cm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to .tflite model')
    parser.add_argument('--x_test_csv', required=True, help='Path to X test CSV')
    parser.add_argument('--y_test_csv', required=True, help='Path to y test CSV')
    parser.add_argument('--output_cm', default='confusion_matrix.png', help='Output path for confusion matrix image')
    args = parser.parse_args()

    print(f"Current working dir: {os.getcwd()}")
    x_test = pd.read_csv(args.x_test_csv)
    y_test = pd.read_csv(args.y_test_csv)
    class_names = ["Still", "Walking", "Manual", "Electric", "Bus", "Subway", "Car"]

    evaluate_tflite(args.model_path, x_test, y_test, class_names, args.output_cm)

if __name__ == '__main__':
    main()
