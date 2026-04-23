**Họ tên:** Trần Thanh Phong
**MSSV**: 2A202600312

# Báo cáo Thực hành LAB 16: Cloud AI Environment Setup (GCP)
## Kết quả nộp bài - Phương án CPU fallback (Section 7)

### 1. Output Terminal chạy `python3 benchmark.py`
**File**: `ScreenShot/terminal.png`
```text
--- LightGBM Benchmark on n2-standard-8 (CPU) ---
Data loaded in 1.82 seconds. Shape: (284807, 31)
Training LightGBM model...
Training completed in 1.07 seconds.

--- Results ---
load_time_sec: 1.8192
train_time_sec: 1.0731
auc_roc: 0.8784
accuracy: 0.9958
precision: 0.2595
recall: 0.7653
f1_score: 0.3876
inference_latency_single_ms: 0.3352
inference_throughput_1k_ms_per_row: 0.0014

Results saved to benchmark_result.json
```

### 2. Nội dung file `benchmark_result.json`
```json
{
    "load_time_sec": 1.8191866874694824,
    "train_time_sec": 1.0731282234191895,
    "auc_roc": 0.8783876926544394,
    "accuracy": 0.9958393314841473,
    "precision": 0.25951557093425603,
    "recall": 0.7653061224489796,
    "f1_score": 0.3875968992248062,
    "inference_latency_single_ms": 0.3352165222167969,
    "inference_throughput_1k_ms_per_row": 0.0013644695281982422
}
```

### 3. Screenshot GCP Billing Reports
**File**: `ScreenShot/billing.png`


### 4. Mã nguồn `terraform-gcp/`
Mã nguồn đã được chỉnh sửa để chuyển sang `e2-standard-8` (fallback cho n2) và tắt GPU accelerator block.
- Thư mục: [terraform-gcp/]

### 5. Báo cáo ngắn (Comparison & Rationale)
- **Lý do sử dụng CPU:** Tài khoản thực hành bị giới hạn quota GPU (Gemma-4-E2B-it yêu cầu NVIDIA T4 nhưng hạn mức mặc định của Project mới là 0). Để đảm bảo tiến độ bài Lab, tôi đã chuyển sang dùng instance CPU cao cấp (`e2-standard-8`) để huấn luyện mô hình LightGBM.
- **So sánh kết quả:** Mặc dù không dùng GPU, kết quả huấn luyện cực kỳ khả quan: thời gian huấn luyện chỉ mất **1.07 giây**, chỉ số **AUC đạt 0.8784**. Đặc biệt, tốc độ suy luận (inference) cực nhanh (~0.33ms/dòng), chứng minh rằng đối với dữ liệu dạng bảng (Tabular Data), hạ tầng CPU mạnh vẫn đáp ứng rất tốt yêu cầu hiệu năng so với chi phí.
