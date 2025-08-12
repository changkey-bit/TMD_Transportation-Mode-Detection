#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
통합 데이터 전처리 파이프라인
1) CSV 수집 및 복사
2) 전처리 (컬럼 추출 → 선형 보간 → 시간 컷)
3) NPY 저장
4) 전체 데이터 병합 후 평균/표준편차 계산
5) 정규화 후 윈도우 단위로 저장
6) 사용자 독립 / 기간 독립 / 랜덤 분할 및 파일 저장
"""

import os
import shutil
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
CLASS_LIST = ["still", "walking", "manualChar", "powerChar", "bus", "metro", "car"]

# 1) CSV 수집 및 복사 ----------------------------------------------------------
def collect_sensor_csv(raw_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for participant in raw_dir.iterdir():
        if not participant.is_dir(): continue
        for csv in participant.glob("*SensorData.csv"):
            dst = out_dir / f"{csv.stem}_{participant.name}.csv"
            shutil.copy(csv, dst)

# 2) 전처리 함수 ---------------------------------------------------------------
def extract_columns(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df.iloc[:, [0]]
    df2 = df.iloc[:, 7:16]
    df3 = df.iloc[:, 18:21]
    df_final = pd.concat([df1, df2, df3], axis=1)
    df_final["Mode"] = CLASS_LIST.index(df["Mode"].iloc[0])
    return df_final

def linear_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    time_index = np.linspace(0, 660, 39600)
    interp_df = pd.DataFrame({
        "Time": time_index,
        "Mode": df["Mode"].iloc[0],
        "interp_flag": True
    })
    merged = pd.merge(df, interp_df, how="outer")
    merged.sort_values("Time", inplace=True)
    merged.interpolate(method="linear", inplace=True)
    return merged[merged["interp_flag"].notna()]

def cut_window(df: pd.DataFrame, start: float=30, end: float=630) -> pd.DataFrame:
    mask = (df["Time"] > start) & (df["Time"] < end)
    return df.loc[mask].iloc[:, 1:14]  # Time 제외, Mode 포함

def preprocess_csv_to_npy(csv_path: Path, npy_dir: Path):
    df = pd.read_csv(csv_path)
    df2 = extract_columns(df)
    df3 = linear_interpolate(df2)
    df4 = cut_window(df3)
    arr = df4.to_numpy()
    npy_path = npy_dir / (csv_path.stem + ".npy")
    np.save(npy_path, arr)

# 3) 전체 NPY 병합 및 mean/std 계산 -------------------------------------------
def compute_mean_std(npy_dir: Path):
    all_arr = []
    for npy in npy_dir.glob("*.npy"):
        all_arr.append(np.load(npy))
    concat = np.concatenate(all_arr, axis=0)
    mean = concat.mean(axis=0)[:12]
    std  = pd.DataFrame(concat).std(axis=0).to_numpy()[:12]
    return mean, std

# 4) 윈도우(300,13) 단위 정규화 저장 -------------------------------------------
def normalize_and_window(npy_dir: Path, out_dir: Path, mean: np.ndarray, std: np.ndarray, window_size: int=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    for npy in npy_dir.glob("*.npy"):
        data = np.load(npy)
        normed = (data[:, :12] - mean) / std
        labels = data[:, 12]
        combined = np.hstack([normed, labels.reshape(-1,1)])
        # reshape 윈도우 단위
        n_windows = len(combined) // window_size
        windows = combined[:n_windows * window_size].reshape(n_windows, window_size, -1)
        np.save(out_dir / npy.name, windows)

# 5) 사용자/기간/랜덤 분할 ----------------------------------------------------
def split_user_independent(npy_dir: Path, out_dir: Path, train_size: int, user_prefix: str="#"):
    rng = np.random.default_rng(442)
    all_files = list(npy_dir.glob(f"*{user_prefix}*.npy"))
    rng.shuffle(all_files)
    train = all_files[:train_size]
    test  = all_files[train_size:]

    def concat_and_save(file_list, save_path):
        arrs = [np.load(f) for f in file_list]
        all_data = np.concatenate(arrs, axis=0)
        np.save(save_path, all_data)

    out_user = out_dir / "user_independent"
    out_user.mkdir(parents=True, exist_ok=True)
    concat_and_save(train, out_user / "user_train.npy")
    concat_and_save(test,  out_user / "user_test.npy")

def split_period_independent(csv_dir: Path, out_dir: Path, exclude_tags: list):
    all_csv = list(csv_dir.glob("*.csv"))
    test_csv = []
    for tag in exclude_tags:
        test_csv += list(csv_dir.glob(f"*{tag}*.csv"))
    train_csv = [f for f in all_csv if f not in test_csv]

    out_period = out_dir / "period_independent"
    out_period.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(f) for f in train_csv]).to_csv(out_period / "period_train.csv", index=False)
    pd.concat([pd.read_csv(f) for f in test_csv ]).to_csv(out_period / "period_test.csv", index=False)

def split_random_all(npy_dir: Path, out_dir: Path, test_ratio: float=0.5):
    all_arr = [np.load(f) for f in npy_dir.glob("*.npy")]
    concat = np.concatenate(all_arr, axis=0)
    train, test = train_test_split(concat, test_size=test_ratio, random_state=442)
    out_rand = out_dir / "random"
    out_rand.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train.reshape(train.shape[0], -1)).to_csv(out_rand / "random_train.csv", index=False)
    pd.DataFrame(test.reshape(test.shape[0], -1)).to_csv(out_rand / "random_test.csv",  index=False)

# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    raw_dir    = Path(args.raw_dir)
    csv_dir    = Path(args.csv_dir)
    npy_raw    = Path(args.npy_raw_dir)
    npy_win    = Path(args.npy_win_dir)
    out_dir    = Path(args.out_dir)
    mean_std   = Path(args.cal_dir)

    # 1) CSV 수집
    collect_sensor_csv(raw_dir, csv_dir)

    # 2) CSV → NPY
    npy_raw.mkdir(parents=True, exist_ok=True)
    for csv in tqdm(csv_dir.glob("*.csv"), desc="Preprocessing CSV"):
        preprocess_csv_to_npy(csv, npy_raw)

    # 3) 전체 병합 → mean/std 계산 & 저장
    mean, std = compute_mean_std(npy_raw)
    mean_std.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mean).to_csv(mean_std / "data_mean.csv", index=False)
    pd.DataFrame(std).to_csv(mean_std / "data_std.csv",  index=False)

    # 4) 정규화 & 윈도우 저장
    normalize_and_window(npy_raw, npy_win, mean, std)

    # 5) 분할
    split_user_independent(npy_win, out_dir, train_size=args.user_train_count)
    split_period_independent(csv_dir, out_dir, exclude_tags=args.exclude_tags)
    split_random_all(npy_win, out_dir, test_ratio=args.random_test_ratio)

    print("All steps completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Har Data Preprocessing Pipeline")
    parser.add_argument("--raw_dir",      type=str, default="./CRC2ND_data")
    parser.add_argument("--csv_dir",      type=str, default="./sensor_data")
    parser.add_argument("--npy_raw_dir",  type=str, default="./npy_raw")
    parser.add_argument("--npy_win_dir",  type=str, default="./prepre_data")
    parser.add_argument("--cal_dir",      type=str, default="./cal_data")
    parser.add_argument("--out_dir",      type=str, default="./final_pre_data")
    parser.add_argument("--user_train_count", type=int, default=23)
    parser.add_argument("--exclude_tags", nargs="+", default=["#2021", "#2022", "#2032", "#2034", "#2036", "#2039"])
    parser.add_argument("--random_test_ratio", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
