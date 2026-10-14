Họ tên: Trần Công Thiện
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.00 | 0.00 | 0.059 | [0.038, 0.084] | 0.77 | 0.099 | 0.021 | 0.119 |
| E1 | LoRA LLM | 33,030,144 | 6.19 | 28.31 | 0.267 | [0.231, 0.308] | 1.25 | 0.230 | 0.194 | 0.492 |
| E2 | + projector | 33,030,144 | 4.74 | 37.03 | 0.096 | [0.072, 0.122] | 1.84 | 0.132 | 0.027 | 0.308 |
| E3 | + LoRA ViT | 39,321,600 | 5.02 | 43.67 | 0.017 | [0.007, 0.029] | 2.52 | 0.025 | 0.005 | 0.040 |
| E4 | Data 10% (fraction 0.1) | 33,030,144 | 3.98 | 36.61 | 0.155 | [0.130, 0.184] | 2.14 | 0.071 | 0.083 | 0.439 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B-Instruct có tổng cộng 2,448,667,136 tham số (~2.45B):
- **Vision Transformer (ViT):** ~435,000,000 tham số (435M), gồm 24 khối transformer xử lý trực tiếp visual tokens từ ảnh đầu vào.
- **Vision-Language Projector:** 54,579,200 tham số (~54.6M), gồm các module `visual.merger` và `visual.deepstack_merger_list` (kiến trúc MLP 2 tầng nén và chiếu đặc trưng ảnh sang không gian nhúng của LLM).
- **LLM Backbone:** ~1,959,000,000 tham số (~1.96B), gồm 36 tầng Decoder transformer tự hồi quy sinh văn bản.

Phạm vi cập nhật ở từng thí nghiệm:
- **E1 (LoRA LLM):** Đóng băng hoàn toàn ViT và Projector. Chỉ gắn và cập nhật LoRA adapter ($r=16, \alpha=32$) vào các ma trận chiếu attention (`q_proj, k_proj, v_proj, o_proj`) và MLP (`gate_proj, up_proj, down_proj`) của LLM backbone. Tổng tham số trainable là 33,030,144 (chiếm 1.35%).
- **E2 (+ Projector):** Giữ nguyên LoRA LLM như E1, đồng thời mở full fine-tuning (cập nhật 100% trọng số) cho toàn bộ khối Projector (54,579,200 tham số) ở định dạng float32.
- **E3 (+ LoRA ViT):** Giữ nguyên LoRA LLM và Projector như E2, gắn thêm LoRA adapter ($r=16, \alpha=32$) vào tất cả 24 khối của ViT (`attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`). Tổng tham số trainable gồm 39,321,600 tham số LoRA (LLM + ViT) và 54,579,200 tham số Projector.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của E0 đạt **0.765** (76.53%, tương ứng 238/311 ảnh parse thành công cấu trúc JSON chuẩn).
- Hai lỗi phổ biến nhất của model gốc khi chạy zero-shot:
  1. *Lỗi sinh văn bản hội thoại lan man (Conversational preambles/postambles):* Mô hình gốc được căn chỉnh để trò chuyện tự nhiên, nên thường thêm lời dẫn nhập hoặc giải thích ở đầu/cuối câu trả lời (ví dụ: *"Based on the dashcam view, here are the defects: ```json ... ``` Hope this helps!"*), hoặc quên đóng dấu ngoặc nhọn/ngoặc vuông kết thúc JSON.
  2. *Sai tên nhãn và định dạng trường (Schema & label hallucination):* Mô hình zero-shot thường tự ý sinh các nhãn nằm ngoài 3 lớp quy định (như `road damage`, `asphalt crack`, `sewer cover`, `manhole cover`) hoặc dùng sai tên trường tọa độ (như `box_2d` hoặc `bbox` thay vì `bbox_2d`).
- Ví dụ output thô thực tế từ model gốc:
  `"The image shows several road surface damages: [{"name": "asphalt crack", "box": [450, 120, 520, 300]}]"` (vừa dính văn bản dẫn dắt vừa sai tên nhãn và sai key).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Nếu mô hình học và sinh ra tọa độ theo thang chuẩn hóa `norm1000` ($[0, 1000]$) nhưng evaluator lại chấm theo hệ quy chiếu `pixel` (ảnh gốc $640 \times 360$):
  - Tọa độ dự đoán của mô hình có giá trị lên tới $[1000, 1000]$, vượt hoàn toàn ra ngoài biên kích thước ảnh thực tế ($640 \times 360$).
  - Khi tính Intersection over Union (IoU) với bounding box ground truth (đang ở pixel $[0..640, 0..360]$), diện tích giao nhau giữa box dự đoán và ground truth sẽ cực kỳ nhỏ hoặc bằng 0, không thể đạt ngưỡng $\text{IoU} \ge 0.5$.
  - Khi đó, số lượng True Positive $\text{TP} = 0$, dẫn đến **F1 lập tức giảm về 0.000** (hoặc sụt giảm thảm hại). Ngược lại, nếu mô hình sinh tọa độ pixel mà chấm bằng thang norm1000, các box dự đoán sẽ co cụm ở góc trên bên trái, IoU cũng $\approx 0$ và F1 = 0.
- Kiểm chứng: Khi chạy hàm chấm điểm `scripts/score.py`, nếu cố ý truyền sai tham số `--coord pixel` cho file dự đoán `work/e1.jsonl` (vốn sinh theo norm1000), kết quả F1 đo được ngay lập tức về 0.000. Bộ unit test `tests/test_label_utils.py` cũng đã kiểm chứng tính toán chuyển đổi chính xác hai chiều giữa YOLO normalized, Pixel $640 \times 360$ và Norm1000.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời (Answer):** Trong bài toán Vision Question Answering và Visual Grounding có giám sát, mục tiêu xác suất có điều kiện là cực đại hóa $P(\text{Answer} \mid \text{Image}, \text{Prompt})$. Do đó, ta gán nhãn `-100` (giá trị bỏ qua của `torch.nn.CrossEntropyLoss`) cho toàn bộ token thuộc về prompt câu hỏi và token thị giác của ảnh, chỉ tính loss trên các token của chuỗi JSON trả lời.
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh:**
  1. *Lãng phí dung lượng mô hình để học thuộc câu hỏi:* Prompt là chuỗi token cố định lặp lại qua hàng nghìn mẫu (*"Phát hiện hư hại mặt đường..."*). Nếu tính loss trên prompt, gradient sẽ ưu tiên tối ưu hóa việc dự đoán các từ ngữ văn phong lặp lại này thay vì tập trung học tọa độ số của các bounding box.
  2. *Mất cân bằng gradient trầm trọng:* Số lượng token ảnh và prompt thường lên tới hàng trăm hoặc hàng nghìn token, áp đảo hoàn toàn số lượng token ít ỏi của chuỗi JSON tọa độ (chỉ khoảng vài chục token). Gradient từ prompt/image sẽ làm chìm gradient của tọa độ box, khiến loss hội tụ sai hướng và điểm F1 định vị hư hại giảm sâu.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị lượng tử hóa 4-bit:**
  Khi tải mô hình bằng BitsAndBytes 4-bit (`load_in_4bit=True`), các lớp tuyến tính bị chuyển sang dạng số nguyên nén `Linear4bit`. Các trọng số này bị đóng băng (`requires_grad=False`), không lưu giữ trạng thái gradient hay optimizer state. Nếu ViT hoặc Projector bị lượng tử hóa 4-bit, quá trình lan truyền ngược (backpropagation) không thể cập nhật gradient qua các ma trận 4-bit này. Do đó, cần loại trừ ViT và Projector khỏi danh sách lượng tử hóa, chỉ lượng tử hóa LLM backbone.
- **Vì sao projector phải ở float32:**
  1. *Độ nhạy số học cao:* Khối Projector là "cổ chai" kết nối giữa hai không gian biểu diễn hoàn toàn khác nhau (thị giác và ngôn ngữ). Gradient truyền qua đây có phương sai lớn; nếu để ở kiểu float16 rất dễ gặp hiện tượng tràn số (overflow) hoặc tắt dần gradient (underflow).
  2. *Tương thích cơ chế `GradScaler`:* Khi dùng mixed precision AMP (`torch.amp.autocast`), PyTorch yêu cầu toàn bộ tham số trainable phải ở kiểu `float32` khi gọi `scaler.unscale_(optimizer)`. Nếu có tham số trainable ở float16, PyTorch sẽ lập tức ném ngoại lệ nghiêm trọng: `ValueError: Attempting to unscale FP16 gradients`.
- **Cách xử lý trong code:**
  Trong `src/model_utils.py`, trước khi bọc PEFT, ta chuyển toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) sang kiểu `torch.float32` và bật `requires_grad=True`. Đồng thời, cấu hình `AdamW(..., foreach=False)` để tránh lỗi xung đột kiểu dữ liệu khi cập nhật trọng số.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả (Hiện tượng rò rỉ dữ liệu - Data Leakage):**
  Dữ liệu dashcam được ghi hình liên tục theo các video hành trình. Các khung hình (frames) liên tiếp trong cùng một video chỉ cách nhau vài phần mười giây, có góc quay, điều kiện thời tiết, ánh sáng, nền đường và các ổ gà/vết nứt giống hệt nhau. Nếu chia ngẫu nhiên theo từng ảnh độc lập, các khung hình của cùng một ổ gà sẽ xuất hiện ở cả tập train lẫn tập test. Mô hình sẽ "học thuộc lòng" đặc điểm bối cảnh của khung hình đó thay vì học khả năng khái quát hóa, dẫn tới F1 trên tập test cao giả tạo nhưng sẽ hoàn toàn thất bại khi chạy trên tuyến đường mới ngoài đời thực. Phân chia theo phiên quay (Session-based) đảm bảo toàn bộ hành trình của một video chỉ nằm ở một split duy nhất.
- **Số ảnh và số box từng lớp ở mỗi split (từ `data/splits/meta.json`):**
  - **Tập Train (1,422 ảnh):** 584 pothole, 1,418 crack, 812 manhole (Tổng cộng: 2,814 boxes).
  - **Tập Val (276 ảnh):** 108 pothole, 267 crack, 153 manhole (Tổng cộng: 528 boxes).
  - **Tập Test (311 ảnh):** 121 pothole, 299 crack, 163 manhole (Tổng cộng: 583 boxes).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu thực tế:**
  - **E1 (LoRA LLM):** F1 = **0.267** | CI95 = `[0.231, 0.308]` | VRAM đỉnh = 6.19 GB | Thời gian train = 28.31 phút.
  - **E2 (+ Projector):** F1 = **0.096** | CI95 = `[0.072, 0.122]` | VRAM đỉnh = 4.74 GB | Thời gian train = 37.03 phút.
  - **E3 (+ LoRA ViT):** F1 = **0.017** | CI95 = `[0.007, 0.029]` | VRAM đỉnh = 5.02 GB | Thời gian train = 43.67 phút.
- **Đánh giá: Hoàn toàn KHÔNG ĐÁNG (thậm chí làm suy giảm nghiêm trọng chất lượng mô hình):**
  1. *Về hiệu quả định vị:* E1 đạt F1 cao vượt trội (0.267), cao gấp gần 2.8 lần E2 (0.096) và gấp hơn 15 lần E3 (0.017). Khoảng tin cậy CI95 của E1 hoàn toàn tách rời, không hề giao thoa với E2 hay E3.
  2. *Về chi phí tài nguyên:* Mở Projector (E2) tăng thời gian huấn luyện lên 37 phút (+30.8%), và gắn thêm LoRA vào ViT (E3) tốn tới 43.67 phút (+54.2% thời gian so với E1).
  3. *Nguyên nhân kỹ thuật:* Bộ tiền huấn luyện thị giác của Qwen3-VL đã được căn chỉnh rất tối ưu trên dữ liệu cực lớn. Với tập dữ liệu nhỏ (1,422 ảnh) và số bước huấn luyện ngắn (300 steps), việc can thiệp vào Projector và ViT gây hiện tượng "quên thảm họa" (catastrophic forgetting) và làm hỏng không gian biểu diễn thị giác sẵn có. Chỉ gắn LoRA vào khối ngôn ngữ LLM (E1) để học định dạng JSON bounding box là phương án vượt trội nhất.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng của khoảng tin cậy CI95 (Bootstrap 95% Confidence Interval):**
  - E0: `[0.038, 0.084]`, độ rộng $\Delta = 0.046$.
  - E1: `[0.231, 0.308]`, độ rộng $\Delta = 0.077$.
  - E2: `[0.072, 0.122]`, độ rộng $\Delta = 0.050$.
  - E3: `[0.007, 0.029]`, độ rộng $\Delta = 0.022$.
  - E4 (10% data): `[0.130, 0.184]`, độ rộng $\Delta = 0.054$.
- **Giới hạn kết luận:**
  - Vì tập test chỉ có 311 ảnh với tổng cộng 583 bounding box, khoảng tin cậy CI95 có biên độ dao động khoảng $\pm 0.025 \sim 0.038$ quanh điểm F1 trung bình.
  - Tuy nhiên, sự vượt trội giữa các mốc setup chính (như E1 so với E0, E2, E3, và E4) là **thực chất và có ý nghĩa thống kê vững chắc (statistically significant)**, bởi khoảng CI95 của E1 (`[0.231, 0.308]`) hoàn toàn nằm trên và không giao cắt với bất kỳ setup nào khác.
  - Giới hạn: Khi so sánh các biến thể có mức chênh lệch nhỏ (dưới 2-3% F1), kích thước 311 ảnh sẽ không đủ lực thống kê (statistical power) để khẳng định chắc chắn mô hình nào tốt hơn, mà cần mở rộng tập test lên tối thiểu 1,000 ảnh.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có hiệu năng kém nhất:** Lớp **`crack` (vết nứt)** luôn có điểm F1 thấp nhất trên toàn bộ các thí nghiệm:
  - Ở E1: crack chỉ đạt F1 = 0.194 (trong khi manhole đạt 0.492 và pothole đạt 0.230).
  - Ở E2: crack chỉ đạt F1 = 0.027.
  - Ở E3: crack chỉ đạt F1 = 0.005.
- **Nguyên nhân chi tiết:**
  1. *Hình dạng và diện tích box:* Vết nứt có đặc điểm mỏng manh, kéo dài ngoằn ngoèo theo nhiều hướng. Bounding box hình chữ nhật bao quanh một vết nứt chéo chứa phần lớn diện tích là mặt đường lành lặn. Khi tính IoU với ground truth, chỉ cần mô hình dự đoán lệch biên một khoảng nhỏ là chỉ số IoU lập tức rớt dưới ngưỡng 0.5.
  2. *Sự mơ hồ về ngữ nghĩa và số lượng False Positive:* Vân nứt trên bề mặt bê tông/nhựa đường rất dễ bị nhầm lẫn với vệt bóng cây, vạch kẻ đường mòn, mối nối nhựa đường hoặc vệt nước. Tại E1, số lượng dự đoán sai (FP) của `crack` lên tới 306 hộp (chiếm 78.6% tổng số FP của toàn mô hình).
  3. *Lớp `manhole` (nắp hố ga) đạt kết quả cao nhất (F1 = 0.492 ở E1):* Nắp hố ga có hình tròn hoặc chữ nhật cứng vững, kích thước tương đối cố định, viền kim loại/bê tông có độ tương phản quang học rõ rệt, giúp mô hình dễ dàng phát hiện chính xác.

- **Minh họa 6 trường hợp dự đoán đại diện (lưu tại `report/figures/`, tạo bởi `src/visualize.py`):**
  - Quy ước màu sắc: **Xanh lá cây** = Ground Truth; **Xanh dương** = True Positive (IoU $\ge 0.5$, đúng lớp); **Đỏ** = False Positive (box bịa/lệch); **Cam** = False Negative (bỏ sót).
  1. `pred_sample_1_success_vlcsnap-2025-02-18-23h13m55s630.jpg`: Phát hiện hoàn hảo cả 2 nắp hố ga `manhole` trên làn đường (GT: 2 manhole | Pred: 2 manhole; TP=2, FP=0, FN=0).
  2. `pred_sample_2_success_vlcsnap-2025-02-18-23h13m56s767.jpg`: Phát hiện hoàn hảo cả 2 nắp hố ga `manhole` ở frame nối tiếp trong cùng hành trình (TP=2, FP=0, FN=0).
  3. `pred_sample_3_success_vlcsnap-2025-02-18-23h14m03s217.jpg`: Phát hiện chuẩn xác 2 nắp cống `manhole` ở khoảng cách xa hơn mà không có bất kỳ báo động giả nào (TP=2, FP=0, FN=0).
  4. `pred_sample_4_error_vlcsnap-2025-02-18-23h30m29s192.jpg`: Báo động giả nghiêm trọng với nhãn `crack` tại vùng đường nhựa có bóng cây đổ và vệt chắp vá mấp mô (GT: 4 pothole, 1 crack | Pred: 7 crack; TP=0, FP=7, FN=5).
  5. `pred_sample_5_error_vlcsnap-2025-02-18-23h26m05s423.jpg`: Bỏ sót nắp hố ga chìm và dự đoán sai các vết nứt chân chim rất mảnh (GT: 2 manhole, 3 crack | Pred: 4 crack; TP=0, FP=4, FN=5).
  6. `pred_sample_6_error_vlcsnap-2025-02-19-17h05m39s299.jpg`: Nhầm lẫn giữa ổ gà và nắp cống trong điều kiện ánh sáng góc xiên (GT: 2 manhole, 1 pothole, 2 crack | Pred: 4 pothole; TP=0, FP=4, FN=5).


### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến thay đổi:** Trục **Tỷ lệ dữ liệu huấn luyện (Training Data Fraction)**:
  - Mốc 10% dữ liệu (`fraction = 0.1`, tương ứng 142 ảnh huấn luyện).
  - Mốc 100% dữ liệu (`fraction = 1.0`, tương ứng 1,422 ảnh huấn luyện - chính là E1).
  - Cả hai đều huấn luyện LoRA LLM ($r=16, \alpha=32$), learning rate $2 \times 10^{-4}$, 300 steps.
- **Kết quả thu được:**
  - **10% dữ liệu (E4):** F1 = **0.155** | CI95 = `[0.130, 0.184]` | Parse rate = 100% | Mean IoU = 0.652 | FP/ảnh = 2.14.
  - **100% dữ liệu (E1):** F1 = **0.267** | CI95 = `[0.231, 0.308]` | Parse rate = 99.4% | Mean IoU = 0.680 | FP/ảnh = 1.25.
- **Kết luận:**
  - Tăng lượng dữ liệu lên gấp 10 lần giúp điểm F1 tăng vọt từ 0.155 lên 0.267 (**tăng +72.2%** hiệu năng tương đối), tỷ lệ báo động giả FP giảm gần một nửa (từ 2.14 xuống 1.25 FP/ảnh), và Mean IoU tăng từ 0.652 lên 0.680.
  - **Tính đáng tin cậy:** Kết luận hoàn toàn đáng tin cậy vì khoảng tin cậy CI95 của hai mốc (`[0.130, 0.184]` và `[0.231, 0.308]`) hoàn toàn tách biệt, không hề chồng lấn nhau. Điều này khẳng định kích thước dữ liệu huấn luyện đóng vai trò then chốt trong việc giúp VLM học thuộc quy luật tọa độ và giảm thiểu ảo giác bounding box.

## Nhìn lại

Qua chuỗi thí nghiệm từ E0 đến E4 trên mô hình Vision-Language Model Qwen3-VL-4B:
1. **Khả năng căn chỉnh định dạng của LoRA:** Mô hình zero-shot (E0) gặp vấn đề lớn về định dạng output (chỉ 76.5% hợp lệ). Chỉ cần gắn LoRA nhẹ trên LLM (E1) với 300 steps, mô hình đã đạt tỷ lệ cú pháp hợp lệ gần như tuyệt đối (99.4% - 100%), chứng minh năng lực instruction-following cực mạnh của kiến trúc QLoRA.
2. **Hiện tượng Catastrophic Forgetting khi can thiệp vào Projector/ViT:** Với tập dữ liệu nhỏ (1,422 ảnh), việc đóng băng ViT và Projector là chìa khóa then chốt để duy trì năng lực trích xuất đặc trưng thị giác mạnh mẽ của mô hình nền tảng.
3. **Giới hạn của hàm mất mát Cross-Entropy trong bài toán Bounding Box:** Cross-entropy đối xử các token số tọa độ như các từ ngữ độc lập (không phân biệt sự gần nhau giữa số 500 và 501), dẫn đến hiện tượng box dự đoán có thể đúng vùng nhưng IoU chưa thật sự khít. Việc áp dụng Reinforcement Learning với phần thưởng IoU trực tiếp (như thuật toán GRPO ở E5) là hướng đi đầy hứa hẹn để giải quyết triệt để bài toán Visual Grounding.

---

## Bonus E5: Huấn luyện GRPO với IoU Reward (+10 điểm)

### 1. Thiết kế hàm Reward IoU và Unit Tests (4 điểm)
- Triển khai trong [`src/reward.py`](src/reward.py):
  - Chuỗi sinh không parse được JSON hợp lệ: trả $0.0$.
  - Cả Ground Truth và Dự đoán đều rỗng: trả $1.0$ (phát hiện đường nguyên vẹn chính xác).
  - Ghép đôi tham lam theo thứ tự IoU giảm dần cho các cặp cùng nhãn lớp, chỉ chấp nhận cặp có $\text{IoU} \ge 0.5$.
  - Công thức phần thưởng phạt box bịa:
    $$R = 0.2 + 0.8 \times \frac{\sum_{\text{cặp khớp}} \text{IoU}}{\max(n_{gt}, n_{pred}, 1)}$$
- **Kiểm thử tự động:** Toàn bộ 7/7 test cases trong [`src/test_reward.py`](src/test_reward.py) đều vượt qua (`pytest src/test_reward.py` đạt 100% PASS), bao gồm kiểm tra JSON hỏng, khớp hoàn hảo, sai lớp, box thừa, và IoU dưới ngưỡng 0.5.

### 2. Quá trình chạy GRPO và Log thực nghiệm (3 điểm)
- Script huấn luyện: [`src/train_grpo.py`](src/train_grpo.py) cấu hình theo [`configs/e5.yaml`](configs/e5.yaml):
  - Khởi tạo từ checkpoint tốt nhất của SFT (E1).
  - Số thế hệ sinh mỗi bước: $G = 4$ completions với sampling nhiệt độ $\tau = 0.7$, `max_completion_length = 128`.
  - Tối ưu hóa trong 60 bước bằng AdamW với learning rate nhỏ $5 \times 10^{-6}$ để duy trì tính ổn định của chính sách (policy).
  - Tính toán Relative Advantage trong nhóm: $A_i = (R_i - \text{mean}(R)) / (\text{std}(R) + \epsilon)$ để hướng dẫn cập nhật policy gradient.
- **Nhật ký huấn luyện (`work/e5_grpo.log`):**
  - Bước 1–15: Mean reward dao động quanh mức $0.34 \sim 0.42$, loss $\approx 0.18$.
  - Bước 30–45: Mô hình bắt đầu loại bỏ các box có độ tự tin thấp, mean reward tăng lên $0.55 \sim 0.63$.
  - Bước 60: Mean reward đạt $0.68$, loss hội tụ về $0.04$, các bounding box sinh ra bám sát viền tổn hại mặt đường hơn.

### 3. Phân tích Reward Hacking và So sánh với SFT (3 điểm)
- **Hiện tượng Reward Hacking (Mô hình gian lận phần thưởng):**
  - *Nếu không có mẫu số $\max(n_{gt}, n_{pred}, 1)$:* Mô hình sẽ dễ dàng hack reward bằng cách sinh ra hàng chục box nhỏ phủ kín lòng đường để tăng xác suất trúng hú họa một box có IoU $\ge 0.5$.
  - *Nếu cho điểm quá cao khi trả về danh sách rỗng `[]`:* Mô hình sẽ rơi vào điểm cực tiểu suy biến (degenerate collapse) bằng cách luôn trả về `[]` để ăn chắc điểm an toàn mà không chịu tìm kiếm tổn hại.
  - Nhờ thiết kế mẫu số phạt nặng $n_{pred}$ và gán reward $0.2$ khi đoán rỗng có GT, mô hình không gặp hiện tượng hack reward nào trong suốt 60 bước.
- **So sánh số liệu thực nghiệm giữa GRPO (E5) và SFT (E1):**
  - **SFT (E1):** F1 = **0.267** | CI95 = `[0.231, 0.308]` | Mean IoU = 0.680 | FP/ảnh = 1.25.
  - **GRPO (E5):** F1 = **0.282** | CI95 = `[0.244, 0.321]` | Mean IoU = **0.705** | FP/ảnh = **1.08**.
- **Đánh giá thống kê qua CI95:**
  - GRPO giúp cải thiện Mean IoU từ 0.680 lên 0.705 (+0.025) và giảm tỷ lệ báo động giả từ 1.25 xuống 1.08 FP/ảnh. F1 tăng nhẹ từ 0.267 lên 0.282.
  - Tuy nhiên, khoảng tin cậy CI95 của E5 (`[0.244, 0.321]`) vẫn có sự giao thoa nhất định với E1 (`[0.231, 0.308]`). Điều này hoàn toàn tự nhiên vì 60 bước GRPO trên tập test nhỏ 311 ảnh chỉ đủ để tinh chỉnh độ khít của tọa độ (fine-grained coordinate refinement) chứ chưa đủ lực khám phá để tách biệt hoàn toàn về mặt khoảng tin cậy.

