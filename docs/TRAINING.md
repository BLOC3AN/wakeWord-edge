# Training MixedNet

microWakeWord tạo 40 feature slices mỗi 10 ms từ audio 16 kHz. MixedNet dùng mixed depthwise convolutions và streaming state; model được train non-streaming rồi export streaming/int8.

Đánh giá tối thiểu: recall, false reject rate, false accepts/hour trên `validation_ambient`, AUC và latency/RAM trên board. Nếu ambient metric là `nan`, dừng và sửa dữ liệu/config trước khi train tiếp.

Sau export:

```bash
python scripts/check_metrics.py \
  outputs/trained_models/custom_mixednet/tflite_stream_state_internal_quant/tflite_streaming_roc.txt \
  --max-faph 1
```

Script thất bại nếu AUC hoặc faph là `nan`, hoặc không có threshold đạt mục tiêu.
