# Slide thuyết trình — thiết kế

## Mục đích

Thuyết trình khoảng 20 phút cho đề tài 5 (phân loại 30 món ăn Việt Nam). Thang điểm: PPT 20%, hiểu bài / hỏi đáp 50%, demo 30%.

Thầy yêu cầu cao. Nhóm phải nói được chính xác mình đã cấu hình những gì, ví dụ ResNet-50 có bao nhiêu tầng, mỗi tầng nhận tensor kích thước nào, số tham số bao nhiêu. Phải trình bày được như một nghiên cứu sinh, không nói chung chung.

Câu chuyện của bộ slide:
1. Nhóm thử 4 kiến trúc trên cùng một tập test, nhóm E.
2. ResNet-50 được chọn theo validation accuracy.
3. Các run T chứng minh từng lựa chọn khi train là cần thiết.
4. Run C1 cho thấy đổi phần cứng giữa chừng không làm sai lệch kết quả.

## Mục tiêu

Bộ slide phải trả lời được 6 yêu cầu của người dùng:
1. Thời lượng khoảng 20 phút.
2. Giải thích model và cách nhóm cấu hình model, đến mức từng tầng.
3. Dữ liệu được chuẩn bị và chuẩn hoá ra sao.
4. Nhóm đã thử những model nào.
5. Thông số train. **Bắt buộc có biểu đồ train và val cho tất cả model.** Confusion matrix là điểm cộng.
6. Demo và kết luận.

Thêm hai yêu cầu chung:
- Mọi con số trên slide đều truy được về code (`colab/`, `src/`) hoặc file kết quả (`colab/results/`).
- Mỗi slide có speaker notes: lời nói, các câu thầy có thể hỏi, và câu trả lời.

## Không làm

- Không train thêm.
- Không chạy E5 hay các run seed.
- Không sửa code của Hoàng (`src/`, `app.py`, `README.md`, `outputs/`, `models/best_model.pt`).

## Điều cần tránh

- Con số không có nguồn, hoặc lấy từ sách thay vì từ chính code của nhóm.
- Chọn model theo test.
- Đặt run chạy trên T4 cạnh run chạy trên M4 rồi so tốc độ.
- Phần nói chính dài quá 20 phút. Chi tiết sâu đưa vào phụ lục.

## Ràng buộc

- Slide dựng bằng Artifact loại **Slides** trên claude.ai (cách A), tải về được dạng .pptx hoặc PDF.
- Ngôn ngữ: tiếng Việt. Thuật ngữ giữ nguyên tiếng Anh (bottleneck, skip connection, AdamW…).
- Script sinh tài liệu chạy local trên CPU, không cần GPU, chỉ mất vài giây. Đây không phải notebook. Claude chỉ chạy sau khi người dùng đã duyệt bản thiết kế này.
- Code mới nằm trong `colab/`, có test, và push thẳng lên `main` theo quy trình hiện tại.

## Các cách đã cân nhắc

1. **Script sinh tài liệu từ code thật, rồi dựng slide.** Người dùng **chọn cách này.** Mọi bảng và hình đều sinh từ model thật và từ `history.csv`, nên khi bị hỏi có thể chỉ ra nguồn.
2. **Viết slide trực tiếp từ kiến thức chung về ResNet-50.** Nhanh hơn, nhưng dễ lệch với cấu hình thật của nhóm (head mới, cách đóng băng backbone, Dropout của EfficientNet).
3. **Thêm notebook `04_slide_assets`.** Đúng quy trình notebook, nhưng mỗi lần sửa hình lại phải chờ người dùng chạy lại. Kết quả như cách 1 mà chậm hơn.

## Phần 1 — Script `colab/tools/slide_assets.py`

Script ghi kết quả vào `colab/results/_slides/` và tạo các file sau:

- **`layers_<arch>.csv`** cho `cnn`, `resnet50`, `mobilenet_v3_large`, `efficientnet_b0`. Mỗi dòng là một module, gồm: tên, loại module, kích thước tensor vào và ra (đo bằng forward hook với ảnh `1×3×224×224`), kernel / stride / padding, số tham số, số tham số trainable.
  - ResNet-50 có thêm bản tóm tắt theo stage: conv1, maxpool, layer1–4 với số block [3, 4, 6, 3], avgpool, fc.
  - Đếm đủ 50 tầng có trọng số: 1 conv1 + 16 bottleneck × 3 conv + 1 fc. Conv ở nhánh downsample không tính.
- **`model_summary.csv`**, mỗi kiến trúc một dòng:
  - tổng tham số, tham số của head, tham số trainable khi đóng băng backbone (T1);
  - MACs ước lượng bằng hook trên Conv và Linear;
  - kích thước feature map cuối.
- **`learning_curves_grid.png`** (bắt buộc): một lưới gồm 11 run (E1–E4, C1, T1–T5). Mỗi ô có accuracy và loss, mỗi biểu đồ có đường train và đường val, đánh dấu epoch có val tốt nhất.
- **`train_val_gap.png`**: khoảng cách train − val accuracy ở epoch tốt nhất của từng run, để thấy run nào overfit (ví dụ T3 không augmentation).
- **`per_class_f1_<run>.png`** cho E2, mô hình cuối: F1 của từng lớp, sắp xếp tăng dần, để nói về các lớp yếu.
- **`confusion_grid.png`**: 4 confusion matrix của E1–E4 đặt cạnh nhau, chuẩn hoá theo hàng.
  - Confusion matrix đầy đủ của E2 đã có sẵn ở `colab/results/E2_resnet50/confusion_matrix.png`.
- **`hyperparams.csv`**: siêu tham số của từng run, đọc từ `config.json` và code.
  - Các giá trị chung: AdamW, weight decay 1e-4, batch 32, early stopping patience 4.
  - Scheduler: CosineAnnealingLR (T_max = số epoch) cho model pretrained. CNN tự xây dùng ReduceLROnPlateau (factor 0.5, patience 1), lr 1e-3, label smoothing 0.05, 15 epoch.
  - Loss: CrossEntropy, có label smoothing ở run có dùng.

Test cho script (TDD, đặt trong `colab/tests/test_slide_assets.py`):
- **ResNet-50 với 30 lớp:**
  - tổng tham số `23,569,502`;
  - head `61,470` tham số;
  - layer4 trả ra `2048×7×7`;
  - số tầng có trọng số là 50;
  - khi đóng băng backbone, số tham số trainable bằng head.
- **CNN tự xây:** kích thước đầu ra của từng block là 32×112×112, 64×56×56, 128×28×28, 256×28×28 (block 4 không có MaxPool), sau GAP là 256, đầu ra cuối là 30.
- **Lưới biểu đồ:** chạy trên `history.csv` giả, tạo đủ một ô cho mỗi run.

## Phần 2 — Nội dung slide (khoảng 17 slide chính và 7 slide phụ lục)

| # | Slide | Nội dung chính | Phút |
|---|---|---|---|
| 1 | Tiêu đề | Tên đề tài, thành viên, môn AI305DV01 | 0,5 |
| 2 | Bài toán | 30 món, ứng dụng, thách thức (món giống nhau: bánh bèo / bánh khọt…), tiêu chí đánh giá | 1 |
| 3 | Dữ liệu | Nguồn (HuggingFace), 24.886 ảnh, phân bố lớp (437–1.517 ảnh), ảnh mẫu | 1,5 |
| 4 | Làm sạch và chia tập | SHA-256 bắt trùng tuyệt đối, dHash bắt gần trùng. Loại 250 ảnh, có cặp trùng khác nhãn. Chia phân tầng 70/15/15, seed 42: 17.407 / 3.733 / 3.746 | 1,5 |
| 5 | Tiền xử lý và chuẩn hoá | Train: RandomResizedCrop(224, scale 0.7–1), HorizontalFlip, ColorJitter(0.15, 0.15, 0.1). Val/test: Resize 256, CenterCrop 224. Normalize `x' = (x − μ)/σ` với μ, σ của ImageNet, và lý do dùng | 1,5 |
| 6 | Phương pháp thí nghiệm | Nhóm E / C / T, chỉ đổi một yếu tố mỗi lần, chọn model theo val, test chỉ để báo cáo | 1 |
| 7 | Các model đã thử | Bảng 4 kiến trúc: ý tưởng chính, số tham số, MACs, head nhóm thay | 1,5 |
| 8 | Kiến trúc ResNet-50 | Sơ đồ các stage, kích thước tensor `3×224² → 64×112² → 256×56² → 512×28² → 1024×14² → 2048×7² → 2048 → 30`, cách đếm 50 tầng | 2 |
| 9 | Bottleneck, skip connection và head | `y = F(x) + x`, khối 1×1 → 3×3 → 1×1, chống vanishing gradient. Head mới `Linear(2048→30)`, 61.470 tham số. Pretrained ImageNet, fine-tune toàn bộ mạng | 1,5 |
| 10 | Cấu hình train | Bảng siêu tham số, loss CE, AdamW, công thức cosine LR, AMP fp16 (T4) / fp32 (M4), early stopping, chọn checkpoint theo val | 1,5 |
| 11 | Biểu đồ train/val (bắt buộc) | `learning_curves_grid.png` cho cả 11 run và nhận xét overfit / underfit | 1,5 |
| 12 | So sánh kiến trúc | `comparison_arch.png` và `accuracy_vs_params.png`, bảng val / test / F1 / top-3 / tốc độ | 1 |
| 13 | Tinh chỉnh (ablation) | `tuning.png` và bảng T1–T5 so với C1, kèm kết luận cho từng yếu tố | 1,5 |
| 14 | Phân tích lỗi | Confusion matrix của E2, F1 theo lớp, 15 ảnh sai | 1,5 |
| 15 | Calibration | Temperature 1,80, ECE 6,0% → 1,3%, vì sao độ tự tin trong demo đáng tin | 0,5 |
| 16 | Demo | Streamlit tại localhost:8501, cách chạy, ảnh chụp màn hình dự phòng | 2 |
| 17 | Kết luận | Kết quả chính, hạn chế (một seed, dữ liệu web), hướng phát triển | 1 |
| A1–A7 | Phụ lục | Bảng đủ từng tầng ResNet-50 · từng tầng CNN tự xây · head của 3 kiến trúc còn lại · siêu tham số đầy đủ · confusion matrix E1–E4 · ảnh hưởng phần cứng (E2 − C1) · F1 từng lớp | — |

Tổng khoảng 21 phút. Khi luyện nói, nếu dài quá thì gộp slide 15 vào 14.

## Đã chốt

- Dùng Artifact loại Slides, theo cách 1.
- Nội dung bám đúng 6 yêu cầu của người dùng.
- Mô hình cuối là E2_resnet50 (`final.json`).

- Nhóm có 3 người. Bộ slide chia thành **3 phần có thời lượng gần bằng nhau**: Phần 1 là slide 1–6 (bài toán, dữ liệu, phương pháp), Phần 2 là slide 7–11 (model và train), Phần 3 là slide 12–17 (kết quả, demo, kết luận). Speaker notes ghi "Phần 1/2/3" và không ghi tên. Người dùng tự phân công người nói.
- Giao diện: người dùng chọn dùng design system "Design System" (https://claude.ai/artifact/3SgyKWVEVqgBKfDAgQsp1i). Khi kiểm tra ngày 2026-11-28, design system này **còn trống** (chưa có tokens hay README). Vì vậy slide dùng giao diện học thuật mặc định của loại Slides. Nếu sau này design system có nội dung thì áp vào.

## Claude tự quyết (người dùng có thể sửa)

- Có 7 slide phụ lục để mở khi bị hỏi sâu. Phần nói chính không dùng tới.
- Kết quả ban đầu của Hoàng trong `outputs/` (EfficientNet-B0, multi-seed) không đưa vào mạch chính. Lý do: dùng pipeline khác, đưa vào dễ gây nhầm. Có thể thêm một slide phụ lục nếu nhóm muốn.
- Sơ đồ kiến trúc vẽ trực tiếp trên slide, không vẽ trong script.

## Còn để người làm quyết

- Cách bố cục và vẽ cụ thể từng slide.
- Màu sắc của biểu đồ.
- Câu chữ trong speaker notes.
