#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 TMD 모델 학습 스크립트
- 모델: multi_cnn, multi_dnn, multi_lstm
- 데이터 로딩, 전처리, 학습, 평가, confusion matrix 출력 포함
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools

# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, save_path, normalize=True,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)

# ─────────────────────────────────────────────────────────────────────────────
def step_decay(epoch, init_lr=1e-3, drop=0.1, epochs_drop=10.0):
    return init_lr * (drop ** np.floor(epoch / epochs_drop))

# ─────────────────────────────────────────────────────────────────────────────
def load_data(base_path, sensor_num, slice_time, num_classes):
    # CSV 로드 및 배열 변환
    x_train = pd.read_csv(os.path.join(base_path, 'user_x_train.csv')).values
    x_test  = pd.read_csv(os.path.join(base_path, 'user_x_test.csv')).values
    y_train = pd.read_csv(os.path.join(base_path, 'user_y_train.csv')).values.ravel()
    y_test  = pd.read_csv(os.path.join(base_path, 'user_y_test.csv')).values.ravel()
    # reshape to (-1, seq_len, features)
    seq_len = 60 * slice_time
    feat = 3 * sensor_num
    x_train = x_train.reshape(-1, seq_len, feat)
    x_test  = x_test.reshape(-1, seq_len, feat)
    # 센서별 분할
    x_train_sets = [x_train[:, :, i*3:(i+1)*3] for i in range(sensor_num)]
    x_test_sets  = [x_test[:,  :, i*3:(i+1)*3] for i in range(sensor_num)]
    # one-hot
    y_train_oh = to_categorical(y_train, num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes)
    return x_train_sets, x_test_sets, y_train_oh, y_test_oh, y_test

# ─────────────────────────────────────────────────────────────────────────────
def build_model(model_type, sensor_num, slice_time, num_classes):
    seq_len = 60 * slice_time
    inputs = [layers.Input(shape=(seq_len,3)) for _ in range(sensor_num)]
    if model_type == 'cnn':
        processed = []
        for inp in inputs:
            x = layers.Conv1D(32,3,padding='same',activation='relu')(inp)
            x = layers.MaxPool1D(2)(x)
            x = layers.Conv1D(64,3,padding='same',activation='relu')(x)
            x = layers.MaxPool1D(2)(x)
            x = layers.Conv1D(128,3,padding='same',activation='relu')(x)
            x = layers.MaxPool1D(2)(x)
            processed.append(layers.Flatten()(x))
        x = layers.concatenate(processed)
    elif model_type == 'dnn':
        processed = []
        for inp in inputs:
            x = layers.Flatten()(inp)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(128,activation='relu')(x)
            processed.append(x)
        x = layers.concatenate(processed)
    elif model_type == 'lstm':
        processed = []
        for inp in inputs:
            x = layers.LSTM(128, return_sequences=True)(inp)
            x = layers.LSTM(128)(x)
            processed.append(x)
        x = layers.concatenate(processed)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=out)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cnn','dnn','lstm'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--slice_time', type=int, default=5)
    parser.add_argument('--sensor_num', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--output', type=str, default='./model.h5')
    parser.add_argument('--cm_path', type=str, default='./confusion.png')
    args = parser.parse_args()

    x_train, x_test, y_train, y_test_oh, y_test = load_data(
        args.data_dir, args.sensor_num, args.slice_time, args.num_classes)

    model = build_model(args.model, args.sensor_num, args.slice_time, args.num_classes)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    lr_cb = LearningRateScheduler(lambda e: step_decay(e), verbose=1)
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[lr_cb]
    )
    loss, acc = model.evaluate(x_test, y_test_oh)
    print(f"Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    # 저장
    model.save(args.output)

    # 예측 및 보고서
    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=-1)
    target_names = ["S","W","M","E","B","Sw","C"] if args.model=='cnn' else None
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plot_confusion_matrix(cm, target_names or list(range(args.num_classes)), args.cm_path)

if __name__ == '__main__':
    main()
