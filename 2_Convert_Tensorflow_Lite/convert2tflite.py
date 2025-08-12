#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow Lite 변환 및 양자화 스크립트
- SavedModel 또는 .h5 모델을 TFLite(.tflite)로 변환
- 선택적 Dynamic Range Quantization 수행

Usage:
    python convert_to_tflite.py \
        --input_dir <keras_model_dir> \
        --output_dir <tflite_output_dir> \
        --models multi_dnn multi_cnn multi_lstm multi_DenseNet \
        [--quantize]
"""
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from glob import glob

def convert_model(model_path: str, output_dir: str, quantize: bool=False):
    """
    Keras 모델(.h5 or SavedModel 폴더)을 TFLite로 변환.
    quantize=True일 경우 Dynamic Range Quantization 적용.
    """
    # Load model
    if os.path.isdir(model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras.models.load_model(model_path))

    # Optional quantization
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    suffix = '_quan' if quantize else '_h5'
    tflite_path = os.path.join(output_dir, f"{model_name}{suffix}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite: {tflite_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Keras models to TFLite')
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='Keras model directory or .h5 files directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save .tflite models')
    parser.add_argument('--models',     nargs='+', required=True,
                        help='List of model names (without extension) to convert')
    parser.add_argument('--quantize',   action='store_true',
                        help='Apply dynamic range quantization')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Gather model paths
    for model_name in args.models:
        # Check for .h5 first, then directory
        h5_path = os.path.join(args.input_dir, f"{model_name}.h5")
        sm_path = os.path.join(args.input_dir, model_name)
        if os.path.isfile(h5_path):
            convert_model(h5_path, args.output_dir, args.quantize)
        elif os.path.isdir(sm_path):
            convert_model(sm_path, args.output_dir, args.quantize)
        else:
            print(f"Warning: '{model_name}' not found as .h5 or SavedModel in {args.input_dir}")

if __name__ == '__main__':
    main()
