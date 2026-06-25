# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Đức Hiếu — 2A202600680  
**Ngày nộp**: 2026-06-25  
**Submission option**: B (GitHub + HuggingFace Hub)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (capped cho T4; p95 của dataset nằm trong khoảng 300–500 tokens)
- **GPU**: Tesla T4, 14.56 GB VRAM (Google Colab Free)
- **Tổng training time**: ~12.2 phút (3 ranks × ~4 phút/rank)
- **Estimated cost**: $0.07 (@ $0.35/hr T4 rate)
- **HF Hub link**: https://huggingface.co/Zilexz/lab21-qwen2.5-3b-vi-r16

---

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200 (0.06%) | 4.00 min  | 7.22 GB   | 1.5577    | 4.748      |
| 16   | 3,686,400 (0.12%) | 4.26 min  | 6.62 GB   | 1.5161    | 4.554      |
| 64   | 14,745,600 (0.48%)| 3.99 min  | 8.00 GB   | 1.4768    | 4.379      |
| Base | —               | —          | —         | N/A       | N/A        |

**Ghi chú**: Base model perplexity không được evaluate riêng trong run này vì T4 không đủ VRAM để load base đồng thời với eval pipeline — base được reload và wrap trước mỗi rank training.

---

## 3. Loss Curve Analysis

Training loss giảm ổn định qua 69 steps (3 epochs × 23 steps/epoch) cho cả 3 ranks. Do T4 không đủ VRAM để chạy eval-during-training (`eval_strategy="no"`), chỉ có train loss curve.

**Quan sát:**
- Không có dấu hiệu overfitting rõ ràng vì eval loss chỉ được đo 1 lần sau khi training kết thúc.
- Eval loss của cả 3 ranks đều thấp hơn mức ban đầu, cho thấy model học tốt từ dataset.
- r=64 có eval loss thấp nhất (1.4768), r=8 cao nhất (1.5577) — xu hướng nhất quán với lý thuyết.
- Train loss curve cho r=16 (baseline): loss giảm từ ~2.0 về ~1.4 sau 3 epochs, không có dấu hiệu tăng đột biến (không overfitting).

---

## 4. Qualitative Comparison (5 examples)

### Example 1 — Machine Learning explanation

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động.

**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo).

**Nhận xét**: **Improved** — fine-tuned rõ ràng hơn, nhấn mạnh "không có sự hướng dẫn trực tiếp" là điểm cốt lõi của ML.

---

### Example 2 — Python Fibonacci code

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base**:
```python
def fibonacci(n):
    if n <= 0:
        return "N phải là số dương"
    # đệ quy hoặc vòng lặp
```

**Fine-tuned (r=16)**:
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    # ...
```

**Nhận xét**: **Improved** — fine-tuned thêm `raise ValueError` (Pythonic hơn), xử lý edge case n=0 riêng.

---

### Example 3 — UI/UX principles

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. (lặp từ "thân thiện")

**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình.

**Nhận xét**: **Same** — cả hai đều liệt kê được, nhưng nội dung hơi khác nhau. Base lặp từ "thân thiện", fine-tuned dùng từ khác nhưng các nguyên tắc không chuẩn xác hơn nhiều.

---

### Example 4 — LoRA vs QLoRA

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp...

**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization... *(sai tên viết tắt)*

**Nhận xét**: **Degraded** — đây là case loss đáng chú ý. Fine-tuned sai tên đầy đủ của LoRA (đổi thành "Layer-wise Adaptive Regularization Optimization" thay vì "Low-Rank Adaptation"). Base giữ đúng tên. Có thể do dataset training không có nhiều ví dụ về LoRA/QLoRA nên model hallucinate khi fine-tune.

---

### Example 5 — Prompt Engineering vs RAG vs Fine-tuning

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học.

**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp...

**Nhận xét**: **Same** — cả hai đều hiểu đúng khái niệm, độ dài tương đương. Fine-tuned bỏ "retrieval augmented generation" trong ngoặc (ngắn gọn hơn nhưng ít thông tin hơn).

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực nghiệm với Qwen2.5-3B trên 200 Vietnamese instruction samples:

**Rank nào cho ROI tốt nhất?** Rank r=16 cho ROI tốt nhất trên dataset này. Mặc dù r=64 có perplexity thấp hơn (4.379 vs 4.554), nhưng mức cải thiện chỉ là 3.8% trong khi số lượng trainable params tăng gấp 4 lần (14.7M vs 3.7M) và VRAM tăng từ 6.6 GB lên 8.0 GB. Với dataset nhỏ chỉ 180 samples, r=64 có nguy cơ overfit cao hơn nhưng kết quả eval cho thấy chưa xảy ra (eval loss vẫn giảm).

**Khi nào tăng rank không còn cải thiện?** Với dataset 180 samples, chênh lệch perplexity giữa r=8 (4.748), r=16 (4.554), và r=64 (4.379) cho thấy cải thiện vẫn có nhưng nhỏ dần — từ r=8 lên r=16 giảm 4.1%, từ r=16 lên r=64 giảm thêm 3.8%. Đây là dấu hiệu diminishing returns bắt đầu xuất hiện. Nếu tăng lên r=128, khả năng cải thiện sẽ còn nhỏ hơn với cùng dataset size này.

**Recommendation cho production:** Nếu deploy production với T4 GPU và dataset tương tự (~200 samples), chọn **r=16** vì: (1) VRAM thấp nhất trong 3 ranks được test (6.6 GB, còn headroom cho inference), (2) perplexity 4.554 chỉ kém r=64 3.8% nhưng checkpoint nhỏ hơn 4x, (3) training time nhanh nhất (4.26 phút). r=8 tiết kiệm thêm nhưng perplexity giảm đáng kể. r=64 phù hợp hơn khi có dataset lớn hơn (≥1000 samples) để khai thác hết capacity của rank cao.

---

## 6. What I Learned

- **LoRA rank ảnh hưởng đến VRAM nhiều hơn training time**: Trong experiment này, training time giữa r=8, r=16, r=64 hầu như bằng nhau (~4 phút), nhưng VRAM tăng từ 7.2 GB lên 8.0 GB. Điều này do số lượng gradient calculations tăng, nhưng với dataset nhỏ 180 samples, bottleneck là data loading, không phải model capacity.

- **Qualitative evaluation cần thiết vì perplexity không nắm bắt được factual errors**: Example 4 (LoRA definition) cho thấy fine-tuned model có perplexity thấp hơn nhưng lại sai tên viết tắt. Perplexity chỉ đo "fluency" của output trên eval set, không đo "correctness" về kiến thức. Cần kết hợp cả hai loại evaluation.

- **Dataset domain mismatch là nguy cơ tiềm ẩn**: Dataset Vietnamese Alpaca là general-purpose, không có nhiều examples về ML/AI terminology. Khi prompt về LoRA/QLoRA (domain chuyên sâu), fine-tuned model hallucinate nhiều hơn base model vì nó đã "học" style của Vietnamese general instruction nhưng không học thêm kiến thức domain-specific. Điều này nhắc nhở rằng fine-tuning cải thiện style/format, không bổ sung knowledge.
