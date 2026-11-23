# Thiết kế: Chạy lại & so sánh mô hình trên Google Colab (Đề tài 5)

Ngày: 2026-11-23

## 1. Mục đích và lý do

- Chạy lại toàn bộ pipeline phân loại 30 món ăn Việt trên **Google Colab bản miễn phí**, không chạy trên máy cá nhân để đỡ tốn tài nguyên.
- Người dùng tự chạy notebook trên Colab, rồi gửi kết quả về để Claude kiểm tra. Đây là vòng lặp làm việc chính.
- Mục tiêu cuối là **bằng chứng cho buổi thuyết trình**. Khi thầy hỏi "vì sao chọn mạng này", nhóm trả lời bằng số liệu: đã thử nhiều mạng và nhiều cấu hình trên **cùng một tập test**, và mạng được chọn tốt hơn ở những tiêu chí cụ thể.
- Đề bài yêu cầu workflow: tiền xử lý → xây dựng → huấn luyện → đánh giá → **tinh chỉnh mô hình**. Code hiện tại còn thiếu phần tinh chỉnh có bằng chứng.

## 2. Mục tiêu / Không làm / Điều phải tránh

**Mục tiêu**
- Mọi con số trong slide do chính nhóm chạy lại trên Colab mà ra, và khớp nhau giữa các file.
- Có bảng so sánh ≥ 4 kiến trúc trên cùng split sạch, kèm accuracy, macro F1, top-3, số tham số, thời gian train và tốc độ suy luận.
- Có thêm vài thử nghiệm tinh chỉnh (ablation) trên mô hình tốt nhất.
- Colab ngắt giữa chừng thì chạy tiếp được, không mất kết quả.
- Có demo chạy được từ Colab.

**Không làm**
- **Không sửa** code của bạn Hoàng: `src/`, `app.py`, `README.md`, `outputs/`, `tests/`. Mọi thứ mới nằm trong `colab/`.
- Không làm slide PowerPoint trong phạm vi này (để sau, khi đã có số liệu).
- Không thiết kế thử nghiệm cần Colab Pro (ViT lớn, quét hyperparameter rộng).

**Điều phải tránh (dù code chạy được vẫn là thất bại)**
- So sánh các mô hình trên split khác nhau, hoặc chọn mô hình/ngưỡng dựa trên tập **test**. Chỉ chọn theo validation, còn test chỉ dùng để báo cáo.
- File kết quả bị ghi đè lẫn nhau giữa các lần chạy (đúng lỗi đang có ở `outputs/metrics.json`).
- Có kết quả mà không biết được chạy bằng cấu hình nào.

## 3. Ràng buộc

- Colab miễn phí: GPU T4 (16 GB), 2 CPU, phiên có thể bị ngắt, giới hạn GPU vài giờ mỗi ngày. Thử nghiệm phải chia nhỏ, mỗi lần chạy ≤ ~45 phút.
- `data/` và `models/` không có trong git, nên Colab phải tự tải data từ Hugging Face (`TuyenTrungLe/vietnamese_food_images`).
- Đọc ảnh trực tiếp từ Google Drive rất chậm. Data phải được copy về đĩa local `/content` ở đầu mỗi phiên.
- Code trong `src/` dùng đường dẫn cố định theo thư mục repo (`ROOT/data/raw`, `ROOT/outputs/clean_split_manifest.json`). Vì vậy data phải nằm đúng `<repo>/data/raw` trên Colab.
- `src/train.build_model` và `src/inference.py` chỉ dựng được EfficientNet-B0.
- Repo: `https://github.com/HonTraan/vietnamese-food-classification` (người dùng đã clone sẵn ở máy). Push thẳng lên `main`: thư mục làm việc của người dùng (`colab/`) tách biệt với của bạn Hoàng nên không lo xung đột.
- Máy người dùng có Google Drive for Desktop, thư mục đồng bộ là `~/Library/CloudStorage/GoogleDrive-<email>/My Drive/`. Drive mất một lúc mới upload xong, nên Colab có thể thấy code cũ nếu chạy ngay sau khi sync.

## Quy trình làm việc (người dùng yêu cầu)

```text
Claude sửa code ở local ──> sync_to_drive.sh ──> Google Drive ──> người dùng chạy notebook trên Colab
        ^                                                              │
        │                                              kết quả ghi vào Drive/runs/
        │                                                              │
        └── chưa ổn: sửa tiếp <── Claude xem xét <── pull_results.sh (Drive ─> colab/results/)
                                         │
                                  ổn: commit + push lên main
```

- GitHub chỉ nhận code **đã chạy ổn trên Colab**. Colab không đọc code từ GitHub.
- Muốn tự xem kết quả thì người dùng mở Drive. Còn không thì Claude kéo kết quả từ thư mục Drive đã đồng bộ về local để xem.

## 4. Các hướng đã cân nhắc

| Hướng | Mô tả | Đánh đổi |
|---|---|---|
| **A. Notebook gọn, gọi code trong repo** ✅ | Notebook `git clone` repo, rồi gọi script trong `colab/` (script này import lại `src/`) | Code chỉ nằm một chỗ; mỗi lần sửa phải push lên GitHub |
| B. Một notebook lớn chép hết code | Tự chứa toàn bộ | Lặp code, dễ lệch với repo, khó chạy tiếp khi bị ngắt |
| C. Zip repo upload lên Drive | Không cần GitHub | Mỗi lần sửa phải upload lại, dễ chạy nhầm bản cũ |

**Người dùng chọn A**, và bổ sung yêu cầu: mọi phần dành cho Colab nằm riêng trong thư mục `colab/`, để bạn Hoàng biết nhóm đã chuyển sang Colab và tránh sửa code của bạn ấy.

Cách đưa code lên Colab (đổi so với ý ban đầu là `git clone`): người dùng muốn chỉ push khi kết quả đã ổn, nên code được **đồng bộ qua Google Drive for Desktop** chứ không qua GitHub. Phương án nhánh `colab-dev` trên GitHub đã được cân nhắc nhưng không chọn, vì Drive đã có sẵn trên máy và không cần thêm nhánh phụ.

## 5. Thiết kế

### 5.1 Cấu trúc thư mục

```text
colab/
├── README.md                 # hướng dẫn chạy (tiếng Việt), ghi chú về code gốc cho bạn Hoàng
├── docs/                     # tài liệu thiết kế (file này)
├── notebooks/
│   ├── 00_setup_data.ipynb   # tải data, làm sạch, EDA, lưu data vào Drive
│   ├── 01_experiments.ipynb  # các thử nghiệm, mỗi thử nghiệm một cell
│   ├── 02_compare.ipynb      # bảng/biểu đồ so sánh, đánh giá + calibration mô hình thắng
│   └── 03_demo.ipynb         # demo ứng dụng
├── train_arch.py             # trainer tổng quát: nhiều kiến trúc + tuỳ chọn tinh chỉnh
├── compare.py                # gom kết quả các run thành bảng + biểu đồ
├── models_zoo.py             # dựng model theo tên kiến trúc
├── sync_to_drive.sh          # local -> Drive: đồng bộ code repo lên Drive
├── pull_results.sh           # Drive -> local: kéo kết quả (không kèm .pt) về colab/results/
└── results/                  # (trong repo) kết quả kéo về từ Drive, không có file .pt
```

**Bố cục trên Drive** (đặt cạnh các thư mục W6–W10 của môn học):

```text
My Drive/DL/vnfood_colab/
├── code/        # bản sao repo, sync_to_drive.sh ghi đè mỗi lần sync
│   └── VERSION.txt   # commit git + "có sửa chưa commit" + thời điểm sync
├── data/        # raw.zip (data đã làm sạch) + manifest, tạo một lần ở notebook 00
└── runs/        # kết quả từng thử nghiệm; sync code không bao giờ đụng vào
```

- `sync_to_drive.sh`: `rsync --delete` toàn bộ working tree của repo sang `code/`, bỏ qua `.git`, `data/`, `models/`, `colab/results/`, `__pycache__`, `.venv`, file PDF đề bài. Sau đó ghi `VERSION.txt`. Chỉ ghi vào `vnfood_colab/code/`.
- `pull_results.sh`: copy `runs/` về `colab/results/`, bỏ qua `*.pt`. Có thể truyền tên một run để chỉ kéo run đó.

`colab/*.py` import `src.dataset` (transform, `ManifestImageFolder`) và `src.cnn_baseline.CNNBaseline` để dùng lại, không sửa các file đó.

### 5.2 Mỗi phiên Colab (cell setup chung ở đầu mỗi notebook)

1. Kiểm tra đang có GPU, nếu không có thì báo lỗi rõ ràng.
2. Mount Google Drive. Thư mục gốc: `MyDrive/DL/vnfood_colab/`.
3. In `code/VERSION.txt` để người dùng đối chiếu với version Claude báo sau khi sync. Lệch thì đợi Drive đồng bộ xong rồi chạy lại cell.
4. Copy `code/` về `/content/vnfood` (đĩa local, nhanh hơn đọc trực tiếp từ Drive) rồi `cd` vào đó.
5. `pip install` những gói Colab còn thiếu.
6. Nếu `Drive/data/raw.zip` đã có thì giải nén về `<repo>/data/raw`. Nếu chưa có, chạy notebook 00 trước.

### 5.3 Notebook 00 — Data (chạy một lần)

- Chạy `python src/dataset.py` (cả 30 lớp), rồi `python src/data_cleaning.py --apply`.
- **Kiểm tra tái lập:** so sánh manifest vừa sinh ra với `outputs/clean_split_manifest.json` trong repo (đếm 17.407 / 3.733 / 3.746, so tập đường dẫn). Khớp thì in "OK". Lệch thì in số dòng khác nhau để người dùng gửi về.
- Chạy `python src/eda.py`.
- Nén `data/raw` (đã làm sạch) cùng manifest lưu lên Drive. Từ đây mọi thử nghiệm dùng chung đúng split này.

### 5.4 Notebook 01 — Thử nghiệm

Tất cả dùng chung: split sạch, seed 42, ảnh 224, AMP (mixed precision), chọn checkpoint theo **validation accuracy**, early stopping patience 4, mỗi run tối đa 12 epoch (CNN 15 epoch). Để so sánh công bằng, các mô hình pretrained ở nhóm 1 dùng chung một công thức: AdamW, lr 3e-4, weight decay 1e-4, cosine schedule, cùng augmentation như `src/dataset.train_transform`. CNN dùng lr 1e-3 giống `src/cnn_baseline.py`.

**Nhóm 1 — So sánh kiến trúc (trả lời "vì sao chọn mạng này")**

| ID | Mô hình | Ghi chú |
|---|---|---|
| E1 | CNN baseline 4 block (train từ đầu) | dùng lại `CNNBaseline` của repo |
| E2 | ResNet-50 fine-tune | kiến trúc kinh điển, ~25M tham số |
| E3 | MobileNetV3-Large fine-tune | nhẹ, nhanh, ~5M tham số |
| E4 | EfficientNet-B0 fine-tune | ứng viên chính, ~5M tham số |
| E5 | ConvNeXt-Tiny fine-tune | kiến trúc hiện đại, ~28M; *tùy chọn* nếu còn quota |

**Nhóm 2 — Tinh chỉnh trên mô hình tốt nhất nhóm 1 (theo validation)**

| ID | Thay đổi so với cấu hình chuẩn | Câu hỏi trả lời được |
|---|---|---|
| T1 | Freeze backbone, chỉ train classifier | Fine-tune toàn bộ có đáng không? |
| T2 | Không ImageNet pretrained | Transfer learning giúp được bao nhiêu? |
| T3 | Không augmentation | Augmentation có giảm overfit không? |
| T4 | Learning rate 1e-3 và 1e-4 (so với 3e-4) | Learning rate ảnh hưởng ra sao? |
| T5 | Label smoothing 0.1 | Có cải thiện / calibration tốt hơn không? |

Thứ tự ưu tiên khi quota ít: E1 → E4 → E2 → E3 → T1 → T3 → T2 → T4 → T5 → E5. Ước tính ~8–10 run × 30–45 phút, chia ra 3–4 ngày.

**Mỗi run** ghi vào `Drive/DL/vnfood_colab/runs/<ID>_<arch>_<cấu hình>/`:
- `config.json`: toàn bộ tham số, commit git, loại GPU, thời gian bắt đầu.
- `history.csv`: theo epoch (train/val loss, train/val accuracy, lr, thời gian epoch).
- `checkpoint_last.pt` (để chạy tiếp khi Colab ngắt) và `best.pt`.
- `metrics.json` trên test: accuracy, macro F1, top-3, classification report, confusion matrix, số tham số, tổng thời gian train, tốc độ suy luận (ảnh/giây trên GPU, batch 1 trên CPU).
- `learning_curves.png`, `confusion_matrix.png`.
- `DONE`: file đánh dấu đã xong. Cell gặp run đã DONE thì bỏ qua. Run chưa DONE mà có `checkpoint_last.pt` thì chạy tiếp từ checkpoint đó.

### 5.5 Notebook 02 — So sánh & đánh giá mô hình thắng

- `compare.py` gom mọi `runs/*/metrics.json` thành `comparison.csv` và các biểu đồ: accuracy/macro F1 theo mô hình, accuracy so với số tham số, learning curves chồng nhau.
- Với mô hình thắng (chọn theo validation): temperature scaling fit trên validation (tái dùng hàm `calibration_metrics` của `src/calibrate.py`), phân tích top-15 ảnh sai, F1 theo từng lớp.
- *Tùy chọn:* chạy multi-seed (42/123/2024) cho mô hình thắng nếu còn quota.
- Không cần nén để tải về: kết quả nằm sẵn trong `Drive/.../runs/`, Claude kéo về bằng `pull_results.sh`.

### 5.6 Notebook 03 — Demo

- Tải `best.pt` của mô hình thắng (đã calibration) từ Drive.
- Nếu mô hình thắng là EfficientNet-B0: copy thành `models/best_model.pt` rồi chạy `app.py` gốc bằng Streamlit qua tunnel `cloudflared`, in ra link public.
- Nếu là kiến trúc khác: `app.py` gốc không dựng được model đó, nên dùng `colab/demo_app.py` (giao diện giống `app.py`: upload ảnh, ngưỡng tin cậy, top-3). App này đọc tên kiến trúc từ checkpoint.

### 5.7 `colab/README.md`

- Hướng dẫn từng bước chạy 00 → 01 → 02 → 03, chạy lại khi bị ngắt, cách gửi kết quả.
- Mục **"Ghi chú về code gốc"** dành cho bạn Hoàng, chỉ liệt kê vấn đề, không sửa:
  - `outputs/metrics.json` đang là kết quả của seed 2024 (88.68%), do multi-seed ghi đè; README ghi 89.08% (seed 42).
  - README ghi "chưa chạy multi-seed" trong khi đã có `multi_seed_results.json`.
  - Lệnh train mặc định trong README thiếu `--fine-tune`, nên chỉ train classifier.
  - Nhánh `--class-weighted` của `train.py` không có tác dụng khi không có manifest.
  - README viết cho Windows/PowerShell.

## 6. Quyết định của Claude (người dùng kiểm tra lại)

- Danh sách thử nghiệm E1–E5, T1–T5 và thứ tự ưu tiên ở trên.
- Thư mục Drive `My Drive/DL/vnfood_colab/` (cạnh W6–W10); tách `code/` (bị ghi đè khi sync) khỏi `data/` và `runs/` (không bao giờ bị ghi đè).
- Sync bằng `rsync` qua thư mục Drive for Desktop, không dùng Google Drive API.
- Sync cả repo (gồm `src/` của bạn Hoàng, chỉ đọc) vì code Colab import từ `src/`.
- Thêm ConvNeXt-Tiny ở mức *tùy chọn*; không đưa ViT vào vì nặng cho T4.
- Dùng AMP để tăng tốc trên T4. Kết quả có thể chênh nhẹ so với số cũ của repo (chạy không AMP), và đây là chủ đích: chạy lại từ đầu.
- Demo dùng Streamlit + cloudflared, giữ đúng `app.py` gốc khi có thể.
- Tài liệu thiết kế đặt trong `colab/docs/` (không tạo `docs/` ở gốc repo, để không động vào phần của bạn Hoàng).

## 7. Để lại cho người code

- Chi tiết `models_zoo.py` (thay classifier cuối cho từng kiến trúc torchvision).
- Batch size theo từng mô hình sao cho vừa T4; `num_workers=2`.
- Cách đo tốc độ suy luận (warm-up, số lần lặp).
- Kiểm tra cục bộ không cần GPU (unit test nhỏ cho `models_zoo`, `compare.py`, logic DONE/resume). Mọi thử nghiệm thật chỉ chạy trên Colab.
