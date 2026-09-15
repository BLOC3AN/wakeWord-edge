# Custom wake word với MixedNet

Pipeline tối giản để train wake word tùy ý bằng microWakeWord/MixedNet và xuất model int8 cho TFLite Micro trên ESP32-S3.

## Mục tiêu

- wake word tiếng Việt dài 2–3 âm tiết;
- tự tạo dữ liệu bằng Piper và bổ sung giọng thu thật;
- đánh giá theo recall, false-reject và false-accepts/hour;
- giữ frontend và streaming runtime giống nhau giữa train và firmware.

## Cấu trúc

```text
configs/                 cấu hình mẫu, không chứa đường dẫn máy cá nhân
data/raw/                WAV gốc (không commit)
data/generated/          WAV tổng hợp/augment (không commit)
data/mmap/               RaggedMmap cho trainer (không commit)
data/external/           ambient và speech negatives (không commit)
models/micro-wake-word/  framework MixedNet
models/piper-sample-generator/  công cụ tạo mẫu, không kèm model weights
scripts/                 các bước tạo dữ liệu, train và export
docs/                    quy trình dữ liệu, đánh giá và triển khai
outputs/                 checkpoint/TFLite (không commit)
```

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "tensorflow[and-cuda]==2.21.0"
pip install -e models/micro-wake-word
pip install -e models/piper-sample-generator
```

Tải riêng model Piper theo hướng dẫn trong `docs/DATASET.md`; không đưa file `.pt`/`.onnx` vào Git.

## Pipeline

1. Chọn đúng một wake phrase và ghi âm mẫu thật.
2. Tạo thêm mẫu Piper cho nhiều speaker/tốc độ.
3. Tạo hard negatives tiếng Việt và ambient validation.
4. Chuyển WAV sang RaggedMmap bằng script chuẩn bị dữ liệu.
5. Sửa `configs/mixednet.example.yaml`, chạy `scripts/train_mixednet.sh`.
6. Chạy `scripts/export_tflite.sh`.
7. Đo lại trên ESP32-S3: latency, RAM, false accepts/hour và false rejects.

```bash
cp configs/mixednet.example.yaml configs/mixednet.yaml
./scripts/train_mixednet.sh configs/mixednet.yaml
./scripts/export_tflite.sh configs/mixednet.yaml
```

## Nguyên tắc dữ liệu

Không trộn nhiều câu khác nhau vào positive nếu sản phẩm chỉ có một wake phrase. Với từ 2–3 âm tiết, cần nhiều người nói, tốc độ, khoảng cách mic và noise; negative phải có câu gần âm và hội thoại tiếng Việt. Không chấp nhận kết quả nếu `AUC` hoặc `false accepts/hour` là `nan`.

## Tài liệu

- `docs/DATASET.md`: chuẩn dữ liệu và split chống leakage.
- `docs/TRAINING.md`: MixedNet, hyperparameter và metric.
- `docs/DEPLOYMENT.md`: TFLite Micro/ESP32-S3.

## License

Kiểm tra license của microWakeWord, Piper và các bộ dữ liệu trước khi public repo.
