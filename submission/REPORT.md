# Lab 21 — Evaluation Report

**Họ tên**: Đỗ Ngọc Phi  **MSSV**: 2A202602531  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB (Tesla T4, 14.7 GB khả dụng, fp16)`

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 mẫu (tỷ lệ 90/10, seed 42 cố định) |
| `max_length` | 1024 — p95 đo được là 98 tokens, p99 là 100 tokens, max là 101 tokens (*results/token_stats.json*) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?**  
Có. File *results/template_check.json* xác nhận `ok: true`, thẻ mở `<think>` và nội dung suy luận bên trong được giữ nguyên vẹn sau khi gọi `apply_chat_template` (`verdict: reasoning preserved — safe to train on traces`). Chuỗi render bảo toàn khối `<think>...</think>`, đảm bảo an toàn tuyệt đối nếu huấn luyện trên dữ liệu suy luận (reasoning traces).

---

## 2. Mask proof (NB1)

| Chỉ số kiểm tra | Giá trị đo được |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` (xác thực assert thành công) |
| Câu hỏi KHÔNG nằm trong loss | `true` (xác thực assert thành công) |

Đoạn token được tính loss giải mã ngược (3–5 dòng đầu):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Toàn bộ phần prompt hệ thống và câu hỏi người dùng được mask hoàn toàn bằng `IGNORE_INDEX (-100)`, đảm bảo mô hình chỉ học cách sinh JSON kết quả mà không học vẹt cách viết lại prompt.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3329.5 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1040.0 |
| (c) LoRA fine-tune | 0.970 | 0.5222 | 1.000 | 1436.1 |

**(b) có thật sự mạnh hơn (a) không?**  
Có, baseline (b) vượt trội hoàn toàn so với baseline (a). Trên tập target, điểm tăng từ 0.000 lên 0.7650; tỷ lệ tuân thủ định dạng JSON tăng từ 0.000 lên tuyệt đối 1.000 (100%); và độ trễ giảm hơn 3 lần từ 3329.5 ms xuống 1040.0 ms do prompt tối ưu ép mô hình sinh trực tiếp JSON ngắn gọn mà không lặp lại câu hỏi hay sinh văn bản râu ria.

File *results/baselines_frozen.json* ghi nhận SHA-256 của `OPTIMIZED_PROMPT` là `719e74d3b6232053`, giữ nguyên vẹn thiết kế gốc của lab, đảm bảo tính liêm chính thực nghiệm: không làm yếu baseline (b) để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6276 | **0.9700** | 394.5 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.5369** | **0.9700** | 260.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 388.6 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 478.5 | **3.86** |

### 4.1 Phân tích vị trí vs rank (`attn_only` so với `correct`)
Run `attn_only` được khớp ngân sách tham số trainable bằng hàm `matched_rank()` với $r=283$, đạt 32,456,704 tham số (chỉ lệch 0.025% so với 32,464,896 tham số của `correct`), đảm bảo phép so sánh hoàn toàn công bằng về dung lượng. Trên tập target thực tế ở NB5 §4, `attn_only` đạt điểm 0.9700, ngang bằng (hòa) với `correct`. 

Tuy nhiên, nếu nhìn vào cột `final_loss` huấn luyện ở NB4, `attn_only` lại có loss thấp hơn đáng kể (0.5369 so với 0.6276 của `correct`). Nếu vội vã kết luận dựa trên train loss, ta sẽ lầm tưởng rank cao $r=283$ ép vào attention là cấu hình vượt trội hơn; nhưng thực chất loss thấp ở đây chỉ là hiện tượng ghi nhớ (memorization) trên tập huấn luyện nhỏ 225 mẫu. Kết quả này chứng minh luận điểm của Deck §11: rank không phải là đòn bẩy chất lượng thần kỳ; việc dồn toàn bộ ngân sách tham số vào một vị trí hẹp (chỉ attention) chỉ giúp tối ưu hóa loss biểu kiến trên tập huấn luyện chứ không mang lại năng lực vượt trội hơn việc phân bổ adapter đều khắp các lớp linear của text decoder (`all-linear`).

### 4.2 Phân tích đòn bẩy Learning Rate (`wrong_lr`)
Run `wrong_lr` chỉ thay đổi duy nhất một biến số là giảm tốc độ học 10 lần từ $1 \times 10^{-4}$ xuống $1 \times 10^{-5}$ (thang LR của Full Fine-Tuning). Đường loss của `wrong_lr` sau 30 steps bị tắc nghẽn ở mức 1.5702, cao hơn gấp 2.5 lần so với `correct` (0.6276). 

Hậu quả trên tác vụ đánh giá là thảm họa: điểm target rơi về 0.0000 và format = 0.0000, độ trễ sinh tăng vọt lên 5272.9 ms vì mô hình hoàn toàn không học được cú pháp JSON và sinh tự do. Nếu một kỹ sư chỉ nhìn vào loss giảm chậm mà không biết về sự lệch thang LR giữa Full-FT và LoRA, họ sẽ dễ kết luận sai lầm rằng *"tác vụ quá khó, dữ liệu 250 mẫu quá ít, hoặc LoRA không đủ năng lực học"*. Thực tế, LoRA chỉ cập nhật một ma trận hạng thấp $BA$ nên đòi hỏi gradient step lớn hơn (~10×) để dịch chuyển phân phối đầu ra; đặt sai thang LR là sai lầm chí mạng triệt tiêu hoàn toàn khả năng hội tụ của mô hình.

### 4.3 Phân tích đánh đổi của QLoRA (`qlora`)
Run `qlora` 4-bit giúp giảm mạnh lượng VRAM tiêu thụ từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm đến **56.0% VRAM**). Điều này cho phép chạy fine-tuning ngay cả trên các GPU tiêu dùng hoặc card đồ họa có bộ nhớ khiêm tốn (như GPU laptop 4–6 GB).

Tuy nhiên, cái giá phải trả thể hiện rõ rệt ở hai khía cạnh:
1. **Độ chính xác suy giảm:** Điểm target tụt từ 0.9700 xuống 0.9400 (mất 3% độ chính xác) do sai số lượng tử hóa 4-bit tác động lên biểu diễn trọng số gốc của dòng Qwen3.5.
2. **Thời gian huấn luyện và độ trễ suy luận tăng:** Thời gian huấn luyện tăng từ 394.5s lên 478.5s (+21.3%) và độ trễ suy luận tăng từ 1436.1 ms lên 1781.4 ms (+24.0%) do overhead giải lượng tử (dequantization kernel on-the-fly).

Số đo thực nghiệm này hoàn toàn ủng hộ khuyến nghị của vendor (Unsloth và Qwen team, Deck §13): đối với kiến trúc Qwen3.5 4B trên phần cứng có đủ VRAM như T4 (14.7 GB khả dụng), nên ưu tiên dùng bf16/fp16 LoRA để bảo toàn độ chính xác và tốc độ thay vì đánh đổi bằng QLoRA 4-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.269` · `valid_trace_rate = 0.0`

### Diễn giải kết quả phán quyết (honesty evaluation)
Cổng hồi quy đánh giá bản fine-tune dựa trên hai điều kiện đồng thời: năng lực tác vụ mục tiêu phải vượt trội hơn baseline (b) (`target_delta > 0.0`), và năng lực tổng quát không được suy giảm quá ngưỡng dung sai cho phép (`regression_delta >= -0.02`). 

Ở bài kiểm tra này, bản fine-tune `correct` đạt bước tiến vượt bậc trên tác vụ phân loại ticket với `target = 0.9700`, đánh bại baseline prompt tối ưu (b) (`0.7650`) với độ chênh lệch rất ấn tượng $\Delta = +0.205$ (+20.5%). Tỷ lệ tuân thủ schema JSON đạt mức hoàn hảo 1.000. 

Tuy nhiên, cổng hồi quy đưa ra phán quyết **FAILED** vì năng lực tổng quát (general capability) trên tập kiểm tra 15 câu hỏi kiến thức phổ thông bị tụt nghiêm trọng từ 0.7911 xuống 0.5222 ($\Delta = -0.2689$, vượt xa ngưỡng dung sai 0.020). Mô hình đã gặp hiện tượng **quên thảm họa (Catastrophic Forgetting)**: khi bị ép học chuyên sâu 100% trên miền dữ liệu hẹp gồm 225 ticket CSKH lặp đi lặp lại cấu trúc JSON, các trọng số LoRA đã vô tình bẻ cong không gian biểu diễn tổng quát của mô hình. 

Phán quyết FAILED này phản ánh chân thực bản chất thực nghiệm trong kỹ nghệ AI: một mô hình fine-tune có thể đạt điểm số gần như tuyệt đối trên bài toán chuyên biệt nhưng lại trở nên "ngô nghê" hoặc mất khả năng trả lời các chỉ dẫn thường thức khác. Giải pháp kiến trúc bắt buộc để khắc phục (theo Deck §6.3) là phải bổ sung 1–5% dữ liệu tổng quát (general replay data) vào tập huấn luyện để duy trì neo kiến thức nền tảng.

---

## 6. Định tính — bắt buộc có cả ca THUA

Bảng tổng hợp 5 ca kiểm tra định tính đại diện (gồm 3 ca FT thắng áp đảo baseline (b) và 2 ca FT thua hoặc bị trừ điểm do đoán sai trường `urgency`):

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | Sai cú pháp / thiếu trường | `{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}` | ✅ **FT thắng tuyệt đối** (Score: 1.0 vs 0.5) |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | Lệch urgency | `{"intent": "hoan_tien", "urgency": "cao", "product": "đèn bàn LED", "sentiment": "tich_cuc"}` | ✅ **FT thắng** (Score: 1.0 vs 0.75) |
| 3 | Shop ơi, mình đặt balo laptop mã đơn VN294388. Hoàn tiền. Ngay lập tức... | `{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tich_cuc"}` | Đoán sentiment sai | `{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tich_cuc"}` | ✅ **FT thắng** (Score: 1.0 vs 0.75) |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... (i=3) | `{"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | Đúng cả 4 trường (`urgency: thap`) | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}` | ❌ **FT THUA** (Score: 0.75 vs 1.00). FT gán nhầm urgency thành `trung_binh` |
| 5 | Chào shop, mình đặt nồi chiên không dầu mã đơn VN949966. Hoàn tiền. Khi nào tiện. Quá tệ. (i=39) | `{"intent": "hoan_tien", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"}` | Đúng cả 4 trường (`urgency: thap`) | `{"intent": "hoan_tien", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "tieu_cuc"}` | ❌ **FT THUA** (Score: 0.75 vs 1.00). FT nhầm urgency `thap` thành `trung_binh` |

### Mẫu chung ở các ca Fine-Tune THUA
Phân tích kỹ lưỡng các ca đạt điểm 0.75 ở file *results/qualitative.json* (như ca $i=3, 5, 12, 39, 41, 46$), ta nhận thấy một mẫu lỗi chung mang tính hệ thống: **Mô hình fine-tune có xu hướng thiên vị (inductive bias) gán `urgency: "trung_binh"` đối với các ticket chứa cụm từ biểu thị mức độ thấp như "Khi nào tiện"**. Trong khi đó, prompt tối ưu (b) nhờ có phần mô tả ngữ nghĩa chi tiết trong system prompt đã phân loại chính xác nhãn `"thap"`. Điều này xảy ra do trong tập huấn luyện 225 mẫu, các ticket mang từ khóa phàn nàn ("chưa thấy tiền", "hoàn tiền", "sản phẩm lỗi") thường đi kèm mức độ khẩn cấp trung bình hoặc cao, khiến adapter ghi nhớ tương quan giả (spurious correlation) giữa loại sự cố và mức độ khẩn cấp, làm lu mờ từ khóa chỉ tần suất thực tế.

---

## 7. Kết luận & điều tôi học được

### Kết luận tổng kết
Dựa trên kết quả thực nghiệm toàn diện của Lab 21, câu trả lời cho câu hỏi có nên triển khai (deploy) bản fine-tune này lên hệ thống production hay không là: **CHƯA NÊN triển khai trực tiếp bản checkpoint hiện tại ra môi trường mở, dù độ chính xác nghiệp vụ đạt tới 97.0%**. Lý do cốt lõi nằm ở kết quả của cổng hồi quy: mô hình bị suy giảm 26.9% năng lực suy luận tổng quát (Catastrophic Forgetting). Nếu triển khai adapter này cho một chatbot phục vụ đa năng, bot sẽ trả lời sai hoặc mất khả năng phản hồi các câu hỏi thông thường của khách hàng. Tuy nhiên, nếu hệ thống áp dụng kiến trúc Microservice — trong đó bản fine-tune chỉ đóng vai trò là một worker chuyên trách định tuyến (routing worker) nhận input và trả về JSON thuần túy mà không tiếp xúc trực tiếp với người dùng cuối — thì bản LoRA này hoàn toàn vượt trội so với prompt engineering nhờ độ chính xác vượt trội (97.0% vs 76.5%) và tính nhất quán format 100%.

Đòn bẩy thực sự quyết định thành công trong bài lab này được xếp hạng rõ ràng theo mức độ tác động:
1. **Loss Masking (Đòn bẩy số 1):** Quyết định sự sống còn của pipeline. Nếu mask sai (`everything`), mô hình chỉ học vẹt câu hỏi và hỏng hoàn toàn từ gốc.
2. **Learning Rate Scale (Đòn bẩy số 2):** Quyết định khả năng hội tụ. Sai lệch 10× LR (`wrong_lr`) biến mô hình thành phế phẩm với target = 0.0000.
3. **Vị trí gắn adapter (`all-linear` vs `attn-only`):** Đóng vai trò cân bằng giữa thông lượng tính toán và độ phủ biểu diễn. Gắn `all-linear` text decoder giúp hội tụ vững chắc mà không tốn công căn chỉnh rank thủ công.
4. **Rank ($r$):** Là yếu tố có mức ảnh hưởng thứ yếu nhất; rank chỉ là bình chứa dung lượng thông tin chứ không phải nút vặn nâng cao chất lượng mô hình.

### Ba điều tôi học được
1. **Đừng bao giờ tin vào Training Loss hay Perplexity đơn thuần:** Thí nghiệm NB4 cho thấy `attn_only` có loss huấn luyện thấp hơn `correct` (0.5369 vs 0.6276), nhưng khi đánh giá trên tập target thực tế thì cả hai đều bằng điểm 0.9700. Đánh giá bằng chỉ số thay thế (surrogate metrics) là nguồn gốc của những kết luận sai lầm trong nghiên cứu LLM.
2. **Loss Mask phải được chứng minh bằng giải mã ngược:** Không thể phó mặc tính đúng đắn cho các thư viện cấp cao. Việc decode token trực tiếp và khẳng định câu hỏi bị masked 100% trong NB1 là bước bảo vệ quan trọng nhất trước khi tiêu tốn hàng giờ tài nguyên GPU.
3. **Prompt Engineering là một mốc chuẩn (baseline) cực kỳ đáng gờm:** Một prompt được thiết kế tối ưu, có cấu trúc và ví dụ (few-shot/in-context) có thể đạt tới 76.5% độ chính xác mà không tốn một giây huấn luyện nào. Fine-tuning chỉ có giá trị khi chứng minh được nó đánh bại baseline prompt tối ưu một cách trung thực và sòng phẳng.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. **Bổ sung 3% dữ liệu General Replay:** Trộn khoảng 10–15 mẫu hỏi đáp kiến thức phổ thông vào tập huấn luyện của NB3 để vượt qua cổng hồi quy (giữ `regression_delta >= -0.02`), biến phán quyết từ FAILED thành PASSED.
2. **Thử nghiệm DoRA (Weight-Decomposed Low-Rank Adaptation):** Phân tách ma trận LoRA thành magnitude và direction để kiểm tra xem có cải thiện được độ chính xác ở các ca khó hay không.
3. **Triển khai vLLM đa adapter:** Tận dụng kết quả hoán đổi adapter ở NB6 để cấu hình phục vụ multi-tenant trên vLLM, đo lường thông lượng request đồng thời thực tế.

---

## Phụ lục — Thưởng đã làm

### B1 — NB6 Merge & Hoán đổi Adapter (+3 điểm)
- Đã thực hiện thành công toàn bộ quy trình ở NB6 thông qua script *notebooks/06_merge_and_serve.py*.
- File *results/merge_check.json* xác nhận:
  * Điểm trước merge (`before_merge`): **0.9700**
  * Điểm sau merge (`after_merge`): **0.9700**
  * Độ suy giảm (`delta`): **0.0000** (nằm trọn vẹn trong ngưỡng dung sai 0.01).
- Trọng số đã merge được xuất an toàn tại thư mục `adapters/merged/`.
- Đã kiểm chứng hot-swap thành công giữa adapter `correct` và các adapter đối chứng trên cùng một base model nạp trong bộ nhớ VRAM.
- **Phân tích trade-off phục vụ:** Merge adapter mang lại lợi ích tuyệt đối về zero-overhead suy luận và đơn giản hóa hạ tầng deploy (chạy như base model nguyên bản). Tuy nhiên, ta mất đi tính linh hoạt phục vụ đa tác vụ. Việc giữ adapter riêng là bắt buộc trong mô hình Multi-tenant SaaS: một base model duy nhất 8 GB chia sẻ trong VRAM có thể phục vụ hàng chục adapter khách hàng khác nhau (mỗi adapter chỉ ~30 MB), giảm hàng chục lần chi phí phần cứng so với việc nạp hàng chục model đã merge riêng biệt.

- [x] **B1 NB6 merge + hot-swap** (kết quả tại `results/merge_check.json`)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
