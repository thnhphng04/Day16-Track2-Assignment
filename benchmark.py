import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import time
import json

print("--- LightGBM Benchmark on n2-standard-8 (CPU) ---")

# 1. Load Data
start_load = time.time()
df = pd.read_csv('creditcard.csv')
load_time = time.time() - start_load
print(f"Data loaded in {load_time:.2f} seconds. Shape: {df.shape}")

# 2. Preprocess
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Training
print("Training LightGBM model...")
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1,
    'n_jobs': 8  # Optimized for n2-standard-8
}

start_train = time.time()
gbm = lgb.train(params, train_data, num_boost_round=100)
train_time = time.time() - start_train
print(f"Training completed in {train_time:.2f} seconds.")

# 4. Inference Benchmark
# Throughput (1000 rows)
X_bench = X_test.iloc[:1000]
start_inf_batch = time.time()
y_pred_batch = gbm.predict(X_bench)
inf_batch_time = (time.time() - start_inf_batch) / 1000

# Single row latency
X_single = X_test.iloc[[0]]
start_inf_single = time.time()
_ = gbm.predict(X_single)
inf_single_time = time.time() - start_inf_single

# 5. Evaluation
y_pred_prob = gbm.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

metrics = {
    "load_time_sec": load_time,
    "train_time_sec": train_time,
    "auc_roc": roc_auc_score(y_test, y_pred_prob),
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "inference_latency_single_ms": inf_single_time * 1000,
    "inference_throughput_1k_ms_per_row": inf_batch_time * 1000
}

print("\n--- Results ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

with open('benchmark_result.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("\nResults saved to benchmark_result.json")
