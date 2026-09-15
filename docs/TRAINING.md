# Training MixedNet

microWakeWord tạo 40 feature slices mỗi 10 ms từ audio 16 kHz. MixedNet dùng mixed depthwise convolutions và streaming state; model được train non-streaming rồi export streaming/int8.

Đánh giá tối thiểu: recall, false reject rate, false accepts/hour trên `validation_ambient`, AUC và latency/RAM trên board. Nếu ambient metric là `nan`, dừng và sửa dữ liệu/config trước khi train tiếp.
