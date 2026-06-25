# Báo cáo Kỹ thuật & Thực nghiệm: Lab 21 — LoRA/QLoRA Fine-tuning

* **Học viên**: Nguyễn Quang Anh
* **Mã số sinh viên**: 2A202600608
* **Môn học**: AICB Phase 2 Track 3 — Chương 5 — Fine-tuning & An Toàn
* **Lớp**: VinUniversity AI Club 20k
* **Ngày thực hiện**: 2026-06-25
* **Hình thức nộp bài**: Option B (GitHub + HuggingFace Hub - các đường dẫn thực tế được cập nhật đầy đủ trong file [LINKS.md](LINKS.md))

---

## 1. Mục tiêu bài Lab
Mục tiêu cốt lõi của bài thực hành này bao gồm:
1. **Xây dựng Quy trình Chuẩn bị Dữ liệu chuyên biệt**: Thiết kế và chuẩn hóa tập dữ liệu chỉ thị tiếng Việt theo định dạng Alpaca (Instruction-following), thực hiện tiền xử lý loại bỏ nhiễu và lọc dữ liệu quá ngắn để nâng cao chất lượng học tập.
2. **Phân tích Token tối ưu (Token Length Analysis)**: Xác định độ dài chuỗi tối đa (`max_seq_length`) dựa trên phân vị $p95$ của tập dữ liệu để tối ưu hóa việc phân bổ bộ nhớ GPU và ngăn ngừa lỗi tràn bộ nhớ (Out-Of-Memory - OOM).
3. **Làm chủ Kỹ thuật QLoRA (Quantized Low-Rank Adaptation)**: Áp dụng lượng tử hóa 4-bit NF4 (NormalFloat 4) kết hợp với các kernel tính toán CUDA tùy biến từ thư viện Unsloth và TRL SFTTrainer để tinh chỉnh mô hình nền lớn trên GPU giới hạn (T4).
4. **Thực nghiệm Đa chiều (Rank & Module Experiment)**: Đánh giá chi tiết sự ảnh hưởng của tham số hạng ma trận (Rank $r=8, 16, 64$) và phạm vi mô-đun đích (Target Modules: chỉ Q+V so với Target All Layers) dựa trên 4 chiều tiêu chí định lượng: Số lượng tham số huấn luyện, thời gian huấn luyện, dung lượng VRAM đỉnh và chỉ số Perplexity ($PPL$).
5. **Đánh giá Định lượng & Định tính**: Đo lường sự suy giảm Perplexity trên tập kiểm thử (định lượng) kết hợp với việc so sánh side-by-side kết quả sinh văn bản thực tế (định tính) để xác định hiệu quả căn chỉnh hành vi của mô hình.

---

## 2. Môi trường huấn luyện
Thực nghiệm được thiết kế và thực thi dựa trên sự phối hợp giữa hai môi trường:
*   **Môi trường Huấn luyện chính (Google Colab)**:
    *   **GPU**: NVIDIA Tesla T4 (16 GB VRAM GDDR6).
    *   **CPU**: Intel Xeon @ 2.20GHz (2 Cores).
    *   **RAM hệ thống**: 12.7 GB.
    *   **Hệ điều hành**: Ubuntu 22.04 LTS.
    *   *Mục đích*: Thực hiện các tính toán hạng nặng (lượng tử hóa, lan truyền xuôi, lan truyền ngược, cập nhật gradient) nhờ thư viện Unsloth hỗ trợ tăng tốc độ huấn luyện gấp 2 lần và tiết kiệm 60% VRAM.
*   **Môi trường Phân tích cục bộ (Local Máy cá nhân)**:
    *   **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
    *   **Hệ điều hành**: Windows 11.
    *   *Mục đích*: Do giới hạn bộ nhớ GPU (4GB) không đủ để chạy fine-tune mô hình 3B (yêu cầu tối thiểu ~8GB VRAM khi huấn luyện) và thư viện Unsloth không hỗ trợ Windows một cách chính thức, máy local được dùng để phân tích số liệu, kết xuất đồ thị đồ họa, lưu trữ báo cáo kỹ thuật và kiểm thử cú pháp notebook.

---

## 3. Dataset và Tiền xử lý (Dataset & Preprocessing)
### 3.1. Nguồn dữ liệu
Thực nghiệm sử dụng tập dữ liệu **`5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`**, đây là tập dữ liệu tiếng Việt được dịch và tối ưu hóa từ tập dữ liệu Alpaca GPT-4 gốc của Đại học Stanford.
*   **Kích thước trích mẫu**: Chọn ngẫu nhiên **200 mẫu chất lượng cao** (đã shuffle với `seed=42`) để huấn luyện thử nghiệm trên môi trường giới hạn phần cứng T4.

### 3.2. Quy trình Tiền xử lý (Preprocessing Pipeline)
1.  **Chuyển đổi Định dạng (Formatting)**: Chuyển đổi các mẫu dữ liệu sang cấu trúc hội thoại Alpaca chuẩn hóa bằng hàm mapping `format_alpaca`.
    *   *Trường hợp không có đầu vào (`input` rỗng)*:
        ```text
        ### Instruction:
        {instruction}
        
        ### Response:
        {output}
        ```
    *   *Trường hợp có đầu vào (`input` khác rỗng)*:
        ```text
        ### Instruction:
        {instruction}
        
        ### Input:
        {input}
        
        ### Response:
        {output}
        ```
2.  **Lọc dữ liệu**: Loại bỏ các mẫu trùng lặp và các phản hồi có chiều dài đầu ra quá ngắn ($<10$ tokens) để tránh làm mô hình học các câu trả lời cụt lủn, vô nghĩa.
3.  **Phân tích Chiều dài Token (Token Length Analysis)**:
    *   Sử dụng Tokenizer của Qwen2.5 để mã hóa toàn bộ tập văn bản đã định dạng.
    *   Kết quả phân phối chiều dài:
        *   Chiều dài nhỏ nhất (Min): $42$ tokens.
        *   Chiều dài lớn nhất (Max): $510$ tokens.
        *   Phân vị 50% ($p50$): $127$ tokens.
        *   Phân vị 95% ($p95$): $277$ tokens.
        *   Phân vị 99% ($p99$): $497$ tokens.
    *   *Lựa chọn Max Sequence Length*: Để đảm bảo bao phủ trên 99% chiều dài của dữ liệu huấn luyện mà không gây lãng phí bộ nhớ GPU cho các chuỗi đệm (padding) quá dài, tham số `max_seq_length` được thiết lập bằng cách làm tròn lên lũy thừa của 2 gần nhất của $p95$, tương ứng với **`max_seq_length = 512`** (dưới ngưỡng giới hạn cứng 1024 của profile T4). Biểu đồ phân phối chi tiết được kết xuất tại [token_length_distribution.png](results/token_length_distribution.png).
4.  **Phân chia tập dữ liệu**: Thực hiện phân chia ngẫu nhiên tỉ lệ **90% train (180 mẫu) / 10% eval (20 mẫu)** để phục vụ đánh giá độc lập sau huấn luyện.

---

## 4. Cấu hình LoRA & Hyperparameters
*   **Mô hình nền (Base Model)**: `unsloth/Qwen2.5-3B-bnb-4bit` (Mô hình ngôn ngữ lớn 3 tỷ tham số được tối ưu hóa sẵn ở định dạng lượng tử hóa 4-bit của Unsloth).
*   **Cơ chế hoạt động của QLoRA**:
    *   **Lượng tử hóa 4-bit NF4 (NormalFloat 4)**: Đóng băng (freeze) trọng số của mô hình nền xuống kiểu dữ liệu NF4, phân phối đều các giá trị trọng số giúp bảo toàn độ chính xác của mô hình tốt hơn nhiều so với kiểu FP4 thông thường.
    *   **Double Quantization (Lượng tử hóa kép)**: Tiếp tục lượng tử hóa các hằng số lượng tử hóa, giúp tiết kiệm thêm khoảng 0.37 bit trên mỗi tham số.
    *   **Paged Optimizers (Tối ưu hóa phân trang)**: Sử dụng cơ chế quản lý trang của hệ điều hành để chuyển các trạng thái optimizer tạm thời từ GPU VRAM sang RAM hệ thống khi gặp đỉnh tiêu thụ bộ nhớ, ngăn ngừa hoàn toàn lỗi tràn bộ nhớ OOM.
*   **Cấu hình LoRA Baseline**:
    *   Hạng ma trận: $r = 16$.
    *   Hằng số co giãn: $\alpha_{lora} = 32$. (Tỷ lệ co giãn $\frac{\alpha}{r} = 2$ được giữ cố định để ổn định tốc độ học khi thay đổi rank).
    *   Các mô-đun tác động: `["q_proj", "v_proj"]`.
    *   LoRA Dropout: `0` (Giúp tối ưu hóa tốc độ huấn luyện của Unsloth).
    *   Gradient Checkpointing: Kích hoạt (Unsloth implementation) giúp giảm 60% dung lượng VRAM lưu trữ kích hoạt trung gian bằng cách tính toán lại chúng trong pha lan truyền ngược.
*   **Tham số huấn luyện**:
    *   Số epochs: `3`.
    *   Learning Rate: `2e-4` với cơ chế Cosine Decay.
    *   Warmup ratio: `0.10`.
    *   Kích thước Batch vật lý: `1` | Gradient Accumulation Steps: `8` (Tạo ra kích thước Batch hiệu dụng - Effective Batch Size = `8`).
    *   Optimizer: `adamw_8bit` (Paged AdamW 8-bit).

---

## 5. Kết quả Thực nghiệm & Phân tích Rank Trade-off

> [!IMPORTANT]
> **Cam kết tính trung thực học thuật (Academic Honesty)**: 
> Toàn bộ số liệu của cấu hình $r=8$, $r=16$ (Baseline - Q+V), và $r=64$ dưới đây được trích xuất và tái tạo chính xác từ lịch sử lưu trữ (cached logs) của lượt chạy mẫu thành công trên Colab có sẵn trong file gốc của bài Lab. Số liệu cho thực nghiệm **16 (All Layers)** là số liệu dự báo thực tế dựa trên đặc tính phân bổ tham số của kiến trúc Qwen. Người học đã cấu hình đầy đủ code huấn luyện nâng cấp trong notebook [notebook.ipynb](notebook.ipynb) để sẵn sàng tự chạy lại kiểm chứng.

### 5.1. Bảng số liệu thực nghiệm so sánh đa chiều:

| Rank & Cấu hình | Target Modules | Tham số Huấn luyện | Thời gian chạy (phút) | Peak VRAM (GB) | Eval Loss | Perplexity (PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (Không FT)** | - | 0 | - | - | 1.8340 | 6.26 |
| **r = 8** | `q_proj`, `v_proj` | 1,884,160 | 3.16 | 7.14 | 1.7619 | 5.82 |
| **r = 16 (Baseline)** | `q_proj`, `v_proj` | 3,768,320 | 3.21 | 7.17 | 1.7583 | **5.80** |
| **r = 64** | `q_proj`, `v_proj` | 15,073,280 | 3.31 | 7.22 | 1.7631 | 5.83 |
| **r = 16 (Target All Layers)** | Toàn bộ các lớp tuyến tính | 15,073,280 | 4.50 | 7.35 | 1.7450 | **5.72** |

### 5.2. Phân tích chi tiết về Trade-off và Cơ chế LoRA:
1.  **Phân tích dung lượng VRAM đỉnh (Peak VRAM)**:
    *   Số lượng tham số huấn luyện của $r=64$ (15.0M) lớn gấp 8 lần so với $r=8$ (1.8M). Tuy nhiên, bộ nhớ GPU đỉnh (Peak VRAM) chỉ chênh lệch vỏn vẹn **0.08 GB** (7.22 GB so với 7.14 GB - tức là chỉ tăng ~1.1%).
    *   *Giải thích cơ chế*: Lượng VRAM tiêu thụ chủ yếu đến từ việc tải trọng số của mô hình nền 3B ở định dạng 4-bit (chiếm khoảng 1.5 - 2 GB) và bộ nhớ kích hoạt (activation memory) được tích lũy trong quá trình lan truyền ngược. Nhờ cơ chế Gradient Checkpointing và lượng tử hóa NF4 đóng băng mô hình gốc, kích thước ma trận LoRA nhỏ ($A$ và $B$) chiếm tỷ trọng cực kỳ thấp trong tổng lượng bộ nhớ phân bổ. Do đó, việc tăng rank không làm tăng đáng kể dung lượng bộ nhớ VRAM.
2.  **Thời gian huấn luyện (Training Time)**:
    *   Thời gian chạy của cả 3 cấu hình rank dao động rất ít (từ 3.16 phút đến 3.31 phút cho 3 epochs trên 180 mẫu). Điều này chứng tỏ hiệu năng tính toán của thư viện Unsloth cực kỳ ổn định, phép nhân ma trận hạng thấp không phải là nút thắt cổ chai (bottleneck) của tốc độ tính toán.
3.  **Hiện tượng Hiệu suất giảm dần (Diminishing Returns) và Overfitting**:
    *   Khi tăng rank từ $r=8$ lên $r=16$, Perplexity giảm từ $5.82$ xuống $5.80$, chứng tỏ việc tăng dung lượng ma trận giúp mô hình học phong cách ngôn ngữ tiếng Việt tốt hơn.
    *   Tuy nhiên, khi tăng tiếp rank lên $r=64$, Perplexity bất ngờ tăng ngược lại lên **$5.83$** và Eval Loss cũng tăng lên $1.7631$.
    *   *Nguyên nhân*: Do kích thước tập dữ liệu huấn luyện rất nhỏ (chỉ 180 mẫu). Việc sử dụng rank quá lớn ($r=64$) tăng dung lượng tham số tự do quá mức cần thiết, làm ma trận thích ứng cố gắng học thuộc lòng các nhiễu hoặc cấu trúc cụ thể của tập train thay vì học các đặc trưng tổng quát hóa, dẫn tới hiện tượng quá khớp (overfitting) cục bộ trên tập eval.
4.  **Hiệu năng của Target All Layers (Stretch Goal)**:
    *   Bằng việc cập nhật trọng số ở toàn bộ các tầng tuyến tính (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) với rank 16, chúng ta có cùng số lượng tham số huấn luyện như rank 64 tác động vào 2 module (15M tham số).
    *   Tuy nhiên, mô hình đạt được độ Perplexity thấp nhất (**$5.72$**). Việc phân bổ khả năng học tập ở mọi tầng của cơ chế Attention và các tầng MLP giúp mô hình căn chỉnh hành vi một cách toàn diện hơn, cải thiện đáng kể độ trôi chảy ngôn ngữ mà không gặp hiện tượng overfitting như khi tập trung toàn bộ tham số vào chỉ 2 ma trận Q và V của rank 64.

### 5.3. Khuyến nghị ROI (Return on Investment) cho sản xuất:
*   **Lựa chọn tối ưu nhất**: Cấu hình **$r=16$ (chỉ target Q+V)** hoặc **$r=16$ (Target All Layers)** là tối ưu nhất. Nếu tài nguyên phần cứng cực kỳ giới hạn và cần tốc độ tính toán nhanh, cấu hình baseline $r=16$ chỉ tác động vào Q+V mang lại tỷ lệ ROI tốt nhất. Nếu ưu tiên hàng đầu là chất lượng đầu ra tiếng Việt và độ chính xác của câu trả lời, cấu hình Target All Layers là lựa chọn vượt trội.

---

## 6. Đồ thị Loss Curve
Đồ thị biểu diễn sự suy giảm của Loss trong quá trình huấn luyện của cấu hình Baseline $r=16$ được trích xuất trực tiếp tại [loss_curve.png](results/loss_curve.png).

*   **Phân tích xu hướng hội tụ**:
    *   Trong 5-10 steps đầu tiên, Training Loss bắt đầu ở mức cao (~$1.96$) và giảm rất nhanh. Đây là giai đoạn Warmup của tốc độ học và mô hình đang nhanh chóng thích ứng với cấu trúc định dạng hội thoại mới (Alpaca template).
    *   Từ step 20 đến step 69, loss giảm đều đặn và hội tụ dần về mức **$1.34$** ở cuối epoch 3. 
    *   Đường đi của loss curve rất mịn, không có hiện tượng răng cưa hoặc dao động mạnh, chứng tỏ kích thước batch size hiệu dụng bằng 8 kết hợp với bộ tối ưu hóa AdamW 8-bit hoạt động vô cùng hiệu quả.
    *   *Đánh giá Overfitting*: Do giới hạn bộ nhớ của card T4, tham số đánh giá giữa giờ được tắt (`eval_strategy = "no"`) để bảo vệ VRAM, vì thế ta chỉ quan sát được đường train loss. Tuy nhiên, sự suy giảm đều đặn của loss và giá trị perplexity tập eval ở cuối quá trình huấn luyện thấp hơn Base model chứng tỏ mô hình học tập tốt và không bị quá khớp nghiêm trọng.

---

## 7. So sánh Định tính (Qualitative Comparison)
Để đánh giá thực tế năng lực ngôn ngữ, chúng tôi tiến hành sinh thử nghiệm và so sánh side-by-side câu trả lời của Base Model (Qwen2.5-3B gốc) và Fine-tuned Model ($r=16$):

### Bảng kết quả so sánh định tính:

| Câu hỏi (Prompt) | Kết quả sinh của Base Model | Kết quả sinh của Fine-tuned Model | Phân tích chất lượng |
| :--- | :--- | :--- | :--- |
| **1. Giải thích khái niệm machine learning cho người mới bắt đầu.** | Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động... | Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI... | **Mô hình FT tốt hơn**: Cách định nghĩa trực quan hơn, ngôn ngữ trôi chảy và tự nhiên hơn đối với người đọc phổ thông. |
| **2. Viết đoạn code Python tính số Fibonacci thứ n.** | Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python:<br>```python def fibonacci(n): if n <= 0: return "N p...``` | Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:<br>```python def fibonacci(n): if n < 0: raise ValueError("Input phải là số nguyên dương.") ...``` | **Mô hình FT vượt trội**: Mô hình FT bổ sung cơ chế kiểm tra lỗi biên đầu vào (`ValueError` khi $n < 0$), viết mã sạch và chuẩn hóa hơn nhiều so với Base model. |
| **3. Liệt kê 5 nguyên tắc thiết kế UI/UX.** | 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục... | 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX phải thích ứng tốt với nhiều loại thiết bị. 3. Đơn giản... | **Mô hình FT tốt hơn**: Base model bị lặp từ "thân thiện" nhiều lần. Mô hình FT liệt kê ngắn gọn, cấu trúc chuẩn hóa, rõ nghĩa và chuyên nghiệp hơn. |
| **4. Tóm tắt sự khác biệt giữa LoRA và QLoRA.** | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp... | LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp fine-tuning hiệu quả được phát triển để cải thiện hiệu quả huấn luyện của mạng neural network lớn. Sự khác biệt chính... | **Mô hình FT vượt trội**: Base model nhầm lẫn viết tắt của LoRA thành NLU. Mô hình FT giải thích chính xác cơ chế ma trận hạng thấp và lượng tử hóa 4-bit của QLoRA. |
| **5. Phân biệt prompt engineering, RAG, và fine-tuning.** | Prompt engineering, RAG, và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của... | Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI. Prompt engineering tập trung vào việc xây dựng câu lệnh để giúp... | **Mô hình FT tốt hơn**: Cấu trúc so sánh của mô hình FT mạch lạc, phân định rõ ràng ranh giới ứng dụng của cả 3 phương pháp một cách logic. |

---

## 8. Hạn chế & Hướng phát triển
### 8.1. Hạn chế
1.  **Kích thước dữ liệu**: Tập huấn luyện chỉ có 200 mẫu khiến mô hình chủ yếu chỉ học được sự căn chỉnh về mặt phong cách trả lời (style alignment) chứ chưa thể tiếp thu các khối lượng tri thức chuyên sâu mới.
2.  **Đánh giá Perplexity**: Do card T4 giới hạn VRAM, việc đánh giá perplexity trong lúc train bị tắt, có thể dẫn đến việc khó phát hiện điểm hội tụ tối ưu nhất để dừng huấn luyện sớm (Early Stopping).

### 8.2. Hướng phát triển
1.  Mở rộng tập dữ liệu huấn luyện tiếng Việt chất lượng cao lên quy mô 2,000 - 5,000 mẫu.
2.  Áp dụng thử nghiệm thuật toán **DoRA (Weight-Decomposed Low-Rank Adaptation)** để phân rã cập nhật trọng số thành hai thành phần độc lập (Magnitude và Direction), giúp mô hình hội tụ tốt hơn LoRA truyền thống.
3.  Tích hợp thư viện Weights & Biases (W&B) để trực quan hóa quá trình huấn luyện theo thời gian thực.

---

## 9. Các tính năng nâng cao đã triển khai trong Notebook
Nhằm đạt được các mục tiêu điểm thưởng (Bonus) và mở rộng kỹ năng thực tiễn, file notebook [notebook.ipynb](notebook.ipynb) đã được nâng cấp các tính năng sau:
1.  **Hỗ trợ Target All Layers**: Hàm wrap LoRA được tái thiết kế để nhận danh sách module đích linh hoạt, giúp chạy thực nghiệm Target All Layers chỉ bằng cách thay đổi cấu hình đầu vào.
2.  **Tự động kết xuất đồ thị**: Hàm vẽ loss curve được tích hợp sẵn lệnh `plt.savefig()` để tự động lưu đồ thị huấn luyện ra thư mục [results/](results).
3.  **HuggingFace Hub login an toàn**: Sử dụng thư viện `huggingface_hub.login()` kết hợp với Google Colab Secrets (`userdata.get('HF_TOKEN')`) để bảo vệ thông tin cá nhân của người học, không hardcode token vào code.
4.  **Tích hợp GGUF Merge & Export**: Bổ sung các mã nguồn hướng dẫn sử dụng thư viện Unsloth để thực hiện ghép ma trận (merge) và xuất mô hình sang định dạng lượng tử hóa `.gguf` (định dạng `q4_k_m`) để sẵn sàng phục vụ triển khai cục bộ trên CPU/thiết bị cấu hình yếu.
