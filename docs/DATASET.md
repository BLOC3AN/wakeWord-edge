# Dataset

Audio dùng mono PCM WAV, 16 kHz. Giữ speaker-disjoint split; không để cùng một người và cùng một câu xuất hiện ở train lẫn validation/test.

Positive chỉ nên là một wake phrase duy nhất dài 2–3 âm tiết. Piper bổ sung độ đa dạng, không thay thế recording thật.

Negative gồm hội thoại tiếng Việt, từ/cụm gần âm, TV/nhạc/quạt và im lặng. Tập `validation_ambient` phải đủ dài để tính false accepts/hour.

Không augment trước khi split; split trước để tránh leakage. Dùng time shift, gain, noise, reverberation và speed perturbation vừa phải.
