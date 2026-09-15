# Deployment ESP32-S3

Export tạo `stream_state_internal_quant.tflite`. Firmware phải dùng cùng frontend với lúc train: 16 kHz mono, cửa sổ 30 ms, stride 10 ms và 40 feature slices.

Giữ tensor state giữa các lần invoke. Chỉ phát wake event sau khi xác suất vượt threshold trong số frame liên tiếp đã chọn; threshold phải đo bằng audio thật.

Board mạnh hơn có thể giữ nguyên frontend/dataset và thay MixedNet bằng CRNN, TC-ResNet hoặc Conformer nhỏ.
