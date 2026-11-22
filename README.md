# Phân loại món ăn Việt Nam từ ảnh

Dự án phân loại **30 món ăn Việt Nam** từ ảnh. Mô hình dùng CNN baseline và EfficientNet-B0 pretrained ImageNet, với dữ liệu từ [Vietnamese Food Images](https://huggingface.co/datasets/TuyenTrungLe/vietnamese_food_images).

## Tổng quan dự án

Mục tiêu là:

- Phân loại một ảnh thực phẩm thành một trong 30 món ăn Việt Nam.
- Có mô hình baseline đơn giản để so sánh.
- Có mô hình pretrained mạnh hơn để cải thiện độ chính xác.
- Đưa kết quả dự đoán và top 3 xác suất lên giao diện Streamlit.
- Kiểm tra dữ liệu, split, kết quả, lỗi sai và độ tin cậy thay vì chỉ nhìn accuracy.

### Pipeline tổng thể

```text
Hugging Face dataset
        |
        v
src/dataset.py ---- tải 30 lớp thành data/raw/<ten-lop>/
        |
        v
src/data_cleaning.py ---- loại duplicate ---- tạo split manifest
        |
        +--> src/cnn_baseline.py ---- mô hình CNN baseline
        |
        +--> src/train.py ----------- EfficientNet-B0
                                     |
                                     v
                          models/best_model.pt
                                     |
                     +---------------+----------------+
                     v                                v
              src/evaluate.py                 src/calibrate.py
                     |                                |
                     v                                v
          metrics, confusion matrix            temperature scaling
                     |                                |
                     +------------ src/inference.py <-+
                                      |
                                      v
                                   app.py
```

### Các quyết định quan trọng

| Quyết định | Cách thực hiện | Lí do |
|---|---|---|
| 30 lớp thay vì 5 lớp | `dataset.py` tải toàn bộ `label_names` mặc định | Đáp ứng mục tiêu phân loại đầy đủ 30 món |
| Giữ augmentation cho train | Random crop, flip, color jitter | Tăng đa dạng dữ liệu và giảm overfit |
| Dùng train/val/test riêng | 70/15/15 theo từng lớp | Có validation để chọn checkpoint và test độc lập |
| Loại ảnh trùng | SHA-256 và dHash | Tránh cùng một ảnh xuất hiện ở nhiều split |
| Seed cố định `42` | Dùng lại seed trong cleaning/train/evaluate | Kết quả dễ tái lập |
| CNN làm baseline | Kiến trúc 4 convolution block | Có mô hình đơn giản để so sánh EfficientNet |
| EfficientNet-B0 làm mô hình chính | Pretrained ImageNet, fine-tune classifier hoặc backbone | Tận dụng đặc trưng hình ảnh pretrained |
| Hiệu chỉnh confidence | Temperature scaling trên validation | Softmax ban đầu quá tự tin; ECE giảm |
| Đánh giá Top-3 | Thêm top-3 accuracy | Hữu ích khi các món gần nhau dễ nhầm |

## Trạng thái hiện tại

- Dataset: **25.136 ảnh gốc, 30 lớp**.
- Sau khi loại duplicate: **24.886 ảnh**, không còn ảnh unreadable.
- Split: **17.407 train / 3.733 validation / 3.746 test**.
- Checkpoint EfficientNet: [models/best_model.pt](models/best_model.pt).
- CNN checkpoint: [models/cnn_baseline.pt](models/cnn_baseline.pt).
- Giao diện: [app.py](app.py).
- App đã dùng xác suất sau temperature scaling.

## Nhiệm vụ từng tuần và cách hoàn thành

### Tuần 1 — Chuẩn bị dữ liệu và CNN baseline

| Nhiệm vụ | Cách hoàn thành | Bằng chứng |
|---|---|---|
| Tạo môi trường Python/PyTorch | Tạo `.venv`, cài thư viện bằng `requirements.txt` | `.venv/`, `requirements.txt` |
| Tải và kiểm tra 30 lớp | `src/dataset.py` tải toàn bộ label names, đổi ảnh RGB, lưu ImageFolder | `data/raw/`, `outputs/eda_summary.json` |
| EDA | Đếm lớp, vẽ phân bố và ảnh mẫu | `src/eda.py`, `outputs/figures/class_distribution.png`, `outputs/figures/sample_images.png` |
| Resize/normalize/augmentation | Resize 224, normalize mean/std ImageNet, augment train | `src/dataset.py` |
| Train/validation/test | Tạo split phân tầng 70/15/15 bằng seed 42; loại duplicate trước split | `outputs/clean_split_manifest.json` |
| CNN baseline | Huấn luyện CNN 4 block convolution | `src/cnn_baseline.py`, `models/cnn_baseline.pt` |
| Learning curve/confusion matrix/metrics | Huấn luyện, evaluate và lưu artifact | `outputs/cnn_history.json`, `outputs/cnn_metrics.json`, `outputs/figures/cnn_learning_curves.png`, `outputs/figures/cnn_confusion_matrix.png` |

**Kết luận Tuần 1:** hoàn thành. CNN đạt 60.04% test accuracy, 58.69% macro F1 và 80.59% top-3 accuracy.

### Tuần 2 — EfficientNet, đánh giá và phân tích

| Nhiệm vụ | Cách hoàn thành | Bằng chứng |
|---|---|---|
| EfficientNet-B0 pretrained 30 lớp | Khởi tạo weight ImageNet và thay classifier cuối | `src/train.py`, `models/best_model.pt` |
| Fine-tune và early stopping | Train full/frozen mode, chọn checkpoint theo validation accuracy | `outputs/training_history.json` |
| Evaluation cùng test set | Hai mô hình dùng cùng manifest sạch | `src/evaluate.py`, `outputs/metrics.json` |
| Accuracy/Macro F1/Top-3/confusion matrix | Tính và lưu metrics trực tiếp từ test | `outputs/model_comparison.csv`, `outputs/metrics.json`, `outputs/figures/confusion_matrix.png` |
| Phân tích ít nhất 15 ảnh sai | Xếp confidence, tạo bảng và ảnh tổng hợp | `outputs/predictions/misclassified_top15.csv`, `outputs/figures/misclassified_top15.png`, `outputs/error_analysis.md` |
| Hàm inference top-3 | Nhận PIL Image, tiền xử lý, temperature scaling, trả top-k | `src/inference.py` |
| Multi-seed | Huấn luyện seed 42/123/2024 và tính mean ± standard deviation | `src/run_multi_seed.py`, `outputs/multi_seed_results.json` |
| Xác minh nhãn/triage 229 mẫu | Tạo 84 contact sheet, rà trực quan từng cặp nhầm, ghi trạng thái rõ ràng | `src/prepare_label_verification.py`, `outputs/label_verification/`, `outputs/weak_class_manual_review.csv` |

**Kết luận Tuần 2:** hoàn thành phần code và đánh giá bắt buộc. Xác minh nhãn 229 mẫu đã được rà trực quan bằng AI-assisted triage; các trường hợp gần giống nhau được giữ nguyên nhãn gốc để tránh tự đổi ground truth mà không đủ bằng chứng.

### Tuần 3 — Demo, tài liệu và trình bày

| Nhiệm vụ | Cách hoàn thành hiện tại | Trạng thái |
|---|---|---|
| Streamlit upload/top-3/confidence | `app.py`, `src/inference.py`, ngưỡng confidence có thể chỉnh | Hoàn thành kỹ thuật |
| Kiểm thử demo | Smoke inference và HTTP Streamlit 200 | Hoàn thành; cần lưu screenshot/video ngoài dataset |
| README và requirements | Hướng dẫn cài, dữ liệu, train, evaluate, demo | Hoàn thành |
| PowerPoint | Chưa tạo file PPT/PPTX trong workspace | Chưa hoàn thành |
| Số liệu bài trình bày | Có metrics, confusion matrix, model comparison, error analysis | Hoàn thành artifact |
| Screenshot/video dự phòng | Chưa lưu | Chưa hoàn thành |
| Kịch bản/bản backup | Chưa tạo tài liệu presentation/backup archive | Chưa hoàn thành |

## 1. Yêu cầu

- Python 3.10 trở lên, Internet, khoảng 5 GB ổ đĩa trống và khuyến nghị 8 GB RAM.
- GPU NVIDIA là tùy chọn; CPU vẫn chạy được nhưng train lâu hơn.

Mở PowerShell tại thư mục dự án và kiểm tra Python:

```powershell
py --version
```

Nếu `py` không nhận, thử `python --version`. Nếu cả hai lỗi, cài Python từ [python.org](https://www.python.org/downloads/) và chọn **Add Python to PATH**.

## 2. Cấu trúc thư mục

```text
vietnamese-food-classification/
├── data/                 # data/raw/<class>/ được tạo sau khi tải ảnh
├── notebooks/            # nơi đặt notebook EDA/thử nghiệm (tùy chọn)
├── src/
│   ├── dataset.py        # tải, lọc, đọc và tiền xử lý dữ liệu
│   ├── data_cleaning.py  # phát hiện duplicate và tạo split manifest
│   ├── train.py          # huấn luyện EfficientNet-B0
│   ├── cnn_baseline.py   # huấn luyện mô hình CNN cơ sở
│   ├── evaluate.py       # đo hiệu năng trên test set
│   ├── calibrate.py      # temperature scaling và calibration report
│   ├── error_analysis.py # tạo bảng/ảnh phân tích dự đoán sai
│   ├── compare_models.py # so sánh CNN và EfficientNet
│   └── inference.py      # hàm dự đoán một ảnh
├── models/               # best_model.pt sau khi train
├── outputs/
│   ├── figures/          # biểu đồ thí nghiệm
│   ├── predictions/      # kết quả dự đoán lưu bổ sung
│   └── metrics.json      # tạo sau khi evaluate
├── app.py                # giao diện Streamlit
├── requirements.txt      # các thư viện cần thiết
└── README.md
```

`data/`, `models/` và `outputs/` là kết quả sinh ra khi chạy project; không nên đưa ảnh hay checkpoint lớn lên Git.

## 3. Cài đặt môi trường (chỉ cần làm lần đầu)

Chạy từng lệnh:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

| Lệnh | Công dụng | Nếu thiếu |
|---|---|---|
| `py -m venv .venv` | Tạo môi trường Python riêng cho project. | Thư viện có thể xung đột với project/Python khác. |
| `Activate.ps1` | Đảm bảo `python` và `pip` dùng môi trường `.venv`. | Dễ cài nhầm package vào Python hệ thống. |
| upgrade `pip` | Dùng trình cài đặt mới hơn. | Thường vẫn chạy nhưng có thể phát sinh lỗi cài package. |
| `pip install -r requirements.txt` | Cài PyTorch, torchvision, datasets, scikit-learn và Streamlit. | Các script sẽ báo `ModuleNotFoundError`. |

Nếu PowerShell chặn `Activate.ps1`, dùng lệnh này cho **cửa sổ hiện tại** rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Kiểm tra sau khi cài:

```powershell
python -c "import torch, torchvision, datasets; print(torch.__version__)"
```

Mỗi khi mở terminal mới, luôn chạy lại `.\.venv\Scripts\Activate.ps1` trước các bước dưới đây.

## 4. Tải và chuẩn bị dữ liệu

Mặc định, project tải **30 lớp** từ dataset nguồn. Lựa chọn 5 lớp hoặc danh sách lớp cụ thể là chế độ thử nghiệm bổ sung.

Nên kiểm tra pipeline trước với 600 ảnh/lớp:

```powershell
python src/dataset.py --max-per-class 600
```

Khi đã ổn, tải toàn bộ 30 lớp:

```powershell
python src/dataset.py
```

`src/dataset.py` tải các split của Hugging Face, giữ các lớp trong source dataset, đổi ảnh sang RGB và lưu theo cấu trúc PyTorch `ImageFolder`. Ví dụ:

```text
data/raw/
├── Banh beo/
├── Banh bot loc/
├── ...
└── Xoi xeo/
```

File này cũng định nghĩa augmentation cho train và phép chuẩn hoá dùng thống nhất cho validation, test và dự đoán. `src/train.py` bắt buộc cần cấu trúc này; nếu chưa chạy bước này, train sẽ báo thiếu data.

Script không tự ghi đè dữ liệu cũ. Muốn xóa `data/raw/` và tải lại:

```powershell
python src/dataset.py --max-per-class 600 --force
```

Có thể dùng dataset Kaggle thay thế, nhưng phải tự đặt ảnh đúng cấu trúc `data/raw/<ten-lop>/`; khi đó không cần chạy `dataset.py`.

### Làm sạch dữ liệu và split

Lí do cần làm sạch:

- Accuracy cũ có thể bị thổi phồng nếu ảnh trùng xuất hiện trong train và test.
- Ảnh trùng về pixel hoặc gần trùng bằng perceptual hash cần nằm cùng một split.
- Giữ nhãn/class phân tầng để các lớp trong test không mất đại diện.

Chạy báo cáo trước, sau đó mới áp dụng:

```powershell
python src/data_cleaning.py
python src/data_cleaning.py --apply
```

- SHA-256 phát hiện ảnh byte giống nhau.
- dHash phát hiện ảnh gần trùng theo cấu trúc pixel.
- Khoảng cách dHash mặc định là `5`.
- File `--apply` di chuyển 250 ảnh trùng hoặc gần trùng sang `data/quarantine/`.
- Script tạo:
  - `outputs/clean_split_manifest.json`: danh sách train/validation/test.
  - `outputs/duplicate_report.csv`: ảnh được giữ và ảnh bị loại.

Không chạy `--apply` nếu cần xem xét thủ công các ảnh bị phát hiện trước.

## 5. Huấn luyện

Chạy baseline:

```powershell
python src/train.py --epochs 12 --batch-size 32
```

Các biến thể phổ biến:

```powershell
# Kiểm tra nhanh mọi bước
python src/train.py --epochs 2 --batch-size 16

# Thiếu RAM/GPU memory
python src/train.py --epochs 12 --batch-size 8

# Không dùng ImageNet pretrained weights (không khuyến nghị)
python src/train.py --epochs 12 --no-pretrained
```

`src/train.py` mặc định đọc `outputs/clean_split_manifest.json` nếu manifest đã tồn tại. Khi đó train/validation/test được lấy đúng danh sách ảnh sạch. Nếu chưa có manifest, script có split ngẫu nhiên có phân tầng dự phòng. Trong cả hai trường hợp, augmentation chỉ áp dụng cho train và checkpoint có validation accuracy tốt nhất được lưu tại:

```text
models/best_model.pt
```

Lần đầu train, PyTorch có thể tải pretrained weights nên cần Internet. Thiếu `data/raw/` thì không thể train; thiếu `models/best_model.pt` thì không thể đánh giá hoặc chạy app. Windows mặc định `--workers 0` để tránh lỗi DataLoader; khi ổn có thể thử `--workers 2`.

## 6. Đánh giá

Sau khi train hoàn tất:

```powershell
python src/evaluate.py
```

`src/evaluate.py` tải `models/best_model.pt`, dựng lại đúng test set từ chỉ số đã lưu, sau đó ghi `outputs/metrics.json`, `outputs/figures/confusion_matrix.png` và các file dự đoán. Kết quả gồm Accuracy, Macro F1, Top-3 Accuracy và confusion matrix 30×30. Nếu chưa train hoặc model bị xóa, lệnh này không chạy được.

## 7. Chạy ứng dụng dự đoán

```powershell
streamlit run app.py
```

Mở URL Streamlit in trong terminal (thường là `http://localhost:8501`), tải ảnh JPG/JPEG/PNG. App hiển thị nhãn dự đoán cùng top 3 món và confidence.

| Thành phần | Vai trò | Nếu thiếu |
|---|---|---|
| `app.py` | Giao diện nhận ảnh, hiển thị kết quả. | Không có web app, nhưng train/evaluate vẫn chạy. |
| `src/inference.py` | Nạp model, biến đổi ảnh, trả về nhãn và xác suất. | App không thể dự đoán. |
| `models/best_model.pt` | Trọng số đã học. | App báo chưa có model. |
| `src/dataset.py` | Transform nhất quán với giai đoạn train. | Ảnh đầu vào không được xử lý đúng. |

Dừng app bằng `Ctrl + C` trong PowerShell.

### Ngưỡng độ tin cậy

Thanh bên trong Streamlit có slider **Ngưỡng độ tin cậy**, mặc định `70%`.

- Confidence đạt hoặc vượt ngưỡng: hiển thị kết quả bình thường.
- Confidence thấp hơn ngưỡng: hiển thị “Chưa xác định được món ăn” và ẩn tên món cùng bảng top 3.
- Slider này được tính trên confidence đã temperature scaling.
- Ngưỡng `70%` là ngưỡng khởi đầu, nên chọn lại bằng validation set; không dùng test set để chỉnh ngưỡng.

## 8. Chuỗi lệnh đầy đủ từ đầu đến cuối

```powershell
# Lần đầu: tạo .venv và cài package
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Chuẩn bị dữ liệu thử nghiệm
python src/dataset.py --max-per-class 600

# Train model
python src/train.py --epochs 12 --batch-size 32

# Đánh giá test set
python src/evaluate.py

# Mở trang dự đoán
streamlit run app.py
```

Các lần sau, khi đã có model, chỉ cần kích hoạt `.venv` rồi chạy `streamlit run app.py`. Không cần tải hoặc train lại trừ khi đổi dữ liệu, model hoặc thông số.

## 9. Hoàn thành baseline và báo cáo

### Làm sạch dữ liệu và huấn luyện đã thực hiện

Chạy script làm sạch để phát hiện duplicate gần đúng:

```powershell
python src/data_cleaning.py
python src/data_cleaning.py --apply
```

Script tạo `outputs/clean_split_manifest.json` chứa split train/validation/test và `outputs/duplicate_report.csv` chứa ảnh bị loại. Với seed `42`, dữ liệu sạch gồm 24.886 ảnh, chia thành:

- 17.407 ảnh train
- 3.733 ảnh validation
- 3.746 ảnh test

Sau khi tạo manifest, train và CNN baseline tự động dùng cùng split sạch này. Hiện EfficientNet-B0 đã hoàn tất train và đạt **best validation accuracy: 89.4%**; checkpoint tại `models/best_model.pt`.

### Kết quả đánh giá mới nhất

Đã huấn luyện lại sau khi loại duplicate và tạo split chống leak. Kết quả trên test set gồm 3.746 ảnh:

| Mô hình | Test accuracy | Macro F1 | Top-3 accuracy |
|---|---:|---:|---:|
| CNN baseline | 60.04% | 58.69% | 80.59% |
| EfficientNet-B0 | 89.08% | 89.05% | 96.74% |

EfficientNet-B0 đã dự đoán đúng 3.337/3.746 ảnh. Các lớp có F1 thấp nhất gồm `Banh duc`, `Banh canh`, `Hu tieu`, `Pho` và `Bun bo Hue`. Các artifact chi tiết nằm tại:

- `outputs/metrics.json`
- `outputs/class_metrics.csv`
- `outputs/confusion_matrix.csv`
- `outputs/cnn_metrics.json`
- `outputs/cnn_class_metrics.csv`
- `outputs/cnn_confusion_matrix.csv`
- `outputs/model_comparison.csv`
- `outputs/predictions/test_predictions.csv`
- `outputs/predictions/misclassified_top15.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/cnn_confusion_matrix.png`
- `outputs/figures/misclassified_top15.png`

### Hiệu chỉnh độ tin cậy

Đã fit temperature scaling trên tập validation bằng:

```powershell
python src/calibrate.py
```

Nhiệt độ: **1.6232**. Trên test set:

| Chỉ số | Trước calibration | Sau calibration |
|---|---:|---:|
| ECE | 5.74% | 1.37% |
| NLL | 0.4865 | 0.4047 |

Báo cáo đầy đủ nằm tại `outputs/calibration.json`. Nhiệt độ được lưu trong `models/best_model.pt`; `src/inference.py` và Streamlit app sẽ dùng xác suất đã hiệu chỉnh.

Phân tích thủ công 15 ảnh sai đã được điền tại `outputs/error_analysis.md`. Các nhóm lỗi chủ yếu là cặp món trông tương tự, ảnh có bố cục kém hoặc nhãn gốc cần kiểm tra lại. Sau khi thay đổi dữ liệu hoặc retrain, chạy lại `python src/error_analysis.py` để tái tạo báo cáo.

Các bước tiếp theo:

```powershell
# Mở ứng dụng dự đoán
streamlit run app.py

# Tạo lại phân tích sai sót sau khi dữ liệu hoặc mô hình thay đổi
python src/error_analysis.py

# Cập nhật bảng so sánh mô hình
python src/compare_models.py
```

```powershell
# EDA: phân bố lớp và ảnh mẫu
python src/eda.py

# CNN baseline: checkpoint, Early Stopping, learning curves, Top-3 Accuracy và confusion matrix
python src/cnn_baseline.py --epochs 15 --batch-size 32 --patience 4

# Đánh giá lại EfficientNet với Top-3 Accuracy và ảnh lỗi
python src/evaluate.py
python src/error_analysis.py

# So sánh CNN baseline với EfficientNet sau khi CNN hoàn thành
python src/compare_models.py
```

Các file báo cáo được tạo trong `outputs/figures/`, `outputs/predictions/`, `outputs/error_analysis.md` và `outputs/model_comparison.csv`.

## Đưa project lên GitHub

Sau khi project đã có commit local, tạo repository rỗng trên GitHub rồi kết nối remote:

```powershell
git remote add origin https://github.com/<username>/<repository-name>.git
git branch -M main
git push -u origin main
```

Không commit các thư mục sau:

- `.venv/`: môi trường Python cục bộ.
- `data/raw/` và `data/quarantine/`: dữ liệu ảnh lớn.
- `models/*.pt`: checkpoint model nặng.
- `.pytest_cache/`, `__pycache__/`, `.ipynb_checkpoints/`: cache tạo ra khi chạy.

Người dùng clone repository vẫn chạy lại model bằng:

```powershell
pip install -r requirements.txt
python src/dataset.py --force
python src/train.py --epochs 12 --batch-size 32 --fine-tune
python src/evaluate.py
streamlit run app.py
```

## Kiểm tra 5 lớp yếu

Chạy script tạo báo cáo audit:

```powershell
python src/audit_weak_classes.py
```

Các file được tạo:

- `outputs/weak_class_audit.csv`: F1, precision, recall và support của 5 lớp yếu.
- `outputs/weak_class_manual_review.csv`: 808 dòng dự đoán liên quan đến các lớp yếu để kiểm tra nhãn thủ công.
- `outputs/weak_class_confusion_pairs.csv`: cặp nhãn bị nhầm nhiều nhất.

Trong file review, script đã điền tự động:

- `manual_label_status`: ví dụ `correct_label`, `wrong_label`, `ambiguous`, `poor_crop`.
- `notes`: mô tả lỗi hoặc nguyên nhân quan sát được.

Mục tiêu không phải xem toàn bộ 808 ảnh một lần. Nên bắt đầu theo cặp nhầm nhiều nhất, ví dụ `Banh canh -> Hu tieu`, `Pho <-> Bun bo Hue`, rồi kiểm tra mẫu sai nhãn hoặc ảnh crop kém.

## Test tự động

Project đã có test tối thiểu tại `tests/test_project.py`.

```powershell
python -m pytest -q
```

Test hiện kiểm tra:

- Train/validation/test không overlap.
- Mọi path trong split tồn tại.
- Temperature scaling làm ECE và NLL giảm.
- Checkpoint có model state và 30 lớp.

Kết quả kiểm tra hiện tại: **4 test passed**.

### Xác minh nhãn thủ công đã đóng triage

Script `src/prepare_label_verification.py` đã tạo 84 contact sheet cho 229 mẫu thuộc các cặp món dễ nhầm.

Script `src/finalize_label_verification.py` đã chốt manifest review:

- `579` dòng: `correct_prediction`
- `229` dòng: `ai_visual_reviewed`
- Không tự đổi nhãn hoặc di chuyển ảnh vì ground truth chuyên môn chưa được xác nhận.

Báo cáo tại `outputs/label_verification/verification_summary.json`. Đây là kết quả triage bảo thủ, không thay thế xác minh chuyên môn của danh sách nhãn gốc.

### Trạng thái triển khai

- Đã thêm manual-review audit cho 5 lớp yếu.
- Đã điền và chốt triage [weak_class_manual_review.csv](outputs/weak_class_manual_review.csv): 579 dự đoán đúng và 229 mẫu được đánh dấu `ai_visual_reviewed`.
- Đã chốt 229 mẫu bằng rà trực quan AI-assisted; không tự đổi nhãn khi chưa có ground truth chuyên môn.
- Đã rà AI-assisted toàn bộ 229 mẫu và cập nhật trạng thái `ai_visual_reviewed`; không tự đổi nhãn gốc.
- Bộ 84 contact sheet đặt tại [outputs/label_verification](outputs/label_verification).
- Các cặp sai cần ưu tiên xem trước: `Banh canh -> Hu tieu`, `Pho <-> Bun bo Hue`, `Bun bo Hue -> Bun rieu`.
- Đã xóa cache Python/notebook và các thư mục tài liệu không thuộc pipeline ML như `doc_review`, `docx_qa`, `ml_doc_render`.
- Đã thêm ngưỡng confidence trong Streamlit, mặc định 70%.
- Đã thêm 4 test tự động cho split, path, calibration và checkpoint.
- Đã thêm script class-weighted training.
- Đã thêm runner multi-seed; chưa chạy thực tế multi-seed nên chưa có mean/standard deviation.
- Chưa chạy retrain nâng cao vì dữ liệu cần được audit thủ công trước.

## Huấn luyện nhiều seed

Để đo độ ổn định của model thay vì tin một lần chạy, dùng:

```powershell
python src/run_multi_seed.py --epochs 12 --batch-size 32 --fine-tune --seeds 42 123 2024
```

- Data split vẫn cố định theo `clean_split_manifest.json`.
- Seed thay đổi chỉ model initialization/training behavior.
- Script lưu checkpoint và metrics cho từng seed.
- Báo cáo tổng hợp gồm mean và population standard deviation tại `outputs/multi_seed_results.json`.

Chạy nhiều seed tốn thời gian; nên chạy ở GPU hoặc sau khi đã audit dữ liệu.

## Huấn luyện cải thiện lớp yếu

Chế độ class-weighted đã được thêm vào train:

```powershell
python src/train.py --epochs 12 --batch-size 32 --fine-tune --class-weighted --seed 42
python src/evaluate.py
python src/calibrate.py
python src/compare_models.py
```

Class weighting dùng trọng số nghịch tần số trên tập train để tăng mức phạt cho lớp ít/tốn sai. Chỉ chạy retrain sau khi bổ sung hoặc sửa nhãn vì class weighting không sửa dữ liệu sai gốc.

## Quá trình đã hoàn thành và lý do

### 1. Loại duplicate để tránh đánh giá quá lạc quan

Khi kiểm tra dữ liệu lần đầu, báo cáo phát hiện 38 nhóm ảnh trùng pixel cross-split và 21 ảnh test trùng ảnh train. Đây là biểu hiện của **data leakage**: mô hình có thể đã nhìn bài kiểm tra ở tập train, làm accuracy cũ không phản ánh khả năng dự đoán ảnh mới.

Giải pháp:

- Phát hiện exact duplicate bằng SHA-256.
- Phát hiện near duplicate bằng dHash.
- Giữ một ảnh đại diện trong mỗi nhóm.
- Cắt 250 ảnh khỏi dữ liệu huấn luyện chính và chuyển sang quarantine.
- Tạo manifest trước, sau đó cùng một seed/manifest dùng cho cả CNN và EfficientNet.

Nguyên nhân có thể của duplicate: cùng ảnh được lấy vào nhiều danh mục, download trùng, crop/crop lại hoặc file có encoding khác nhưng nội dung pixel giống nhau.

### 2. So sánh mô hình bằng cùng test set

CNN baseline được huấn luyện 15 epoch để có bộ chuẩn đơn giản. EfficientNet-B0 pretrained được huấn luyện riêng và fine-tune backbone. Cả hai dùng seed `42` và cùng split sạch.

Kết quả cho thấy EfficientNet cải thiện **29.04 điểm test accuracy** so với CNN:

```text
60.04% CNN
89.08% EfficientNet
        +29.04 điểm
```

CNN được giữ lại để chứng minh pretrained transfer learning đem lại giá trị và tránh kết luận dựa trên một mô hình duy nhất.

### 3. Phân tích sai sót thay vì chỉ nhìn accuracy

Dự đoán sai được xếp theo confidence và lưu tại `outputs/predictions/misclassified_top15.csv`. Hình tổng hợp nằm tại `outputs/figures/misclassified_top15.png`; nguyên nhân quan sát được ở `outputs/error_analysis.md`.

Các lỗi lặp lại chủ yếu thuộc ba nguyên nhân:

1. **Món hình ảnh tương tự:** `Banh tet/Banh chung`, `Banh xeo/Banh khot`, `Pho/Bun bo Hue/Bun rieu`.
2. **Bố cục/crop kém:** ảnh tập trung vào đĩa phụ, rau, thịt hoặc nền thay vì món chính.
3. **Nhãn gốc đáng ngờ:** một số ảnh có thể được đặt sai lớp hoặc có nhiều món trong cùng khung.

Phân tích này giúp quyết định nên cải thiện dữ liệu trước, thay vì chỉ tăng epoch hoặc đổi kiến trúc.

### 4. Hiệu chỉnh xác suất bằng temperature scaling

Confidence ban đầu được lấy trực tiếp từ softmax. Một số dự đoán sai vẫn có confidence gần 100%, nên softmax không nhất thiết là xác suất thực tế.

Script `src/calibrate.py`:

- Không dùng test set để fit.
- Thu logits của validation set.
- Tối ưu một hệ số temperature bằng NLL.
- Ghi nhiệt độ vào checkpoint.
- Đo lại ECE, MCE, Brier score và NLL.
- Cập nhật confidence trong evaluate và Streamlit inference.

Kết quả hiệu chỉnh:

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Test ECE | 5.74% | 1.37% |
| Test NLL | 0.4865 | 0.4047 |
| Temperature | 1.0000 | 1.6232 |

Điều này làm confidence đáng tin hơn nhưng không làm thay accuracy hoặc top-3 accuracy.

## Hướng cải thiện tiếp theo

Ưu tiên theo tác động và rủi ro:

1. Kiểm tra thủ công nhãn và ảnh trong 5 lớp F1 thấp: `Banh duc`, `Banh canh`, `Hu tieu`, `Pho`, `Bun bo Hue`.
2. Loại ảnh sai nhãn, bổ sung ảnh crop rõ món chính.
3. Tăng dữ liệu cho các cặp món dễ nhầm thay vì chỉ lấy thêm ảnh ngẫu nhiên.
4. Dùng class-weighted loss hoặc oversampling có kiểm soát.
5. Thử RandAugment, MixUp hoặc CutMix.
6. Fine-tune nhiều seed và báo mean ± standard deviation.
7. Thử threshold confidence để từ chối dự đoán không chắc chắn.
8. Thêm test tự động cho cleaning, split, checkpoint và inference.
9. Xuất ONNX và triển khai versioned model nếu app được đưa lên môi trường thật.

## Cách đọc kết quả

- **Accuracy:** tỷ lệ ảnh dự đoán đúng ở toàn bộ test.
- **Macro F1:** F1 trung bình của từng lớp, không phụ thuộc số lượng ảnh.
- **Top-3 accuracy:** nhãn đúng có nằm trong 3 dự đoán cao nhất không.
- **ECE:** độ sai khác trung bình giữa confidence và accuracy trong các nhóm confidence.
- **NLL:** mức phạt xác suất dự đoán nhãn đúng; càng thấp càng tốt.
- **Brier score:** sai lệch của toàn bộ phân phối xác suất so với nhãn one-hot.

Không nên kết luận model tốt chỉ dựa trên accuracy. Với 30 lớp và các món hình ảnh gần nhau, macro F1, confusion matrix, top-3 và calibration cần xem cùng nhau.

## Cấu trúc file cuối

Sau khi dọn dự án, các file ngoài pipeline chính được loại bỏ hoặc giữ tối thiểu:

- `src/`: mã nguồn huấn luyện, đánh giá, inference và xác minh dữ liệu.
- `tests/`: test regression cho split, checkpoint và calibration.
- `data/raw/`: dữ liệu huấn luyện đang dùng.
- `models/`: checkpoint CNN baseline và EfficientNet.
- `outputs/metrics.json`: kết quả test EfficientNet hiện tại.
- `outputs/class_metrics.csv`: precision/recall/F1/support theo hàng ngang cho EfficientNet.
- `outputs/confusion_matrix.csv`: confusion matrix EfficientNet theo hàng/cột lớp.
- `outputs/cnn_metrics.json`: kết quả CNN baseline.
- `outputs/cnn_class_metrics.csv`: precision/recall/F1/support theo hàng ngang cho CNN.
- `outputs/cnn_confusion_matrix.csv`: confusion matrix CNN theo hàng/cột lớp.
- `outputs/training_history.csv`: learning history EfficientNet theo từng epoch.
- `outputs/cnn_history.csv`: learning history CNN theo từng epoch.
- `outputs/calibration_summary.csv`: metrics calibration trước/sau cho validation và test.
- `outputs/calibration_reliability.csv`: độ tin cậy theo từng bin và trước/sau calibration.
- `outputs/multi_seed_results.csv`: từng metric theo seed, mean và standard deviation.
- `outputs/eda_summary.csv`: số ảnh theo từng lớp.
- `outputs/model_comparison.csv`: bảng so sánh hai mô hình.
- `outputs/clean_split_manifest.json`: split sạch được dùng để train/evaluate.
- `outputs/calibration.json`: kết quả temperature scaling.
- `outputs/label_verification/`: contact sheet và báo cáo rà nhãn.
- `outputs/figures/`: biểu đồ EDA, learning curve, confusion matrix và model comparison.
- `outputs/predictions/`: dự đoán test và ảnh sai.

## 10. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError` | Chưa activate `.venv` hoặc chưa cài package. | Activate `.venv`, rồi `pip install -r requirements.txt`. |
| `Dataset missing` | Chưa có `data/raw/`. | Chạy `python src/dataset.py --max-per-class 600`. |
| Không có `best_model.pt` | Chưa train hoặc model đã bị xóa. | Chạy `python src/train.py`. |
| `CUDA out of memory` | Batch size quá lớn. | Dùng `--batch-size 8` hoặc `--batch-size 4`. |
| Tải data bị gián đoạn | Mạng không ổn định. | Chạy lại dataset; nếu dữ liệu dở dang, thêm `--force`. |
| `py`/`python` không nhận | Python chưa cài hoặc chưa vào PATH. | Cài Python, mở lại terminal. |

## Nguồn dữ liệu

- [30VNFoods](https://github.com/ds4v/30VNFoods): 25.136 ảnh, 30 món ăn Việt Nam, MIT License.
- [Vietnamese Food Images trên Hugging Face](https://huggingface.co/datasets/TuyenTrungLe/vietnamese_food_images): nguồn mặc định của script.
- [Vietnamese Foods trên Kaggle](https://www.kaggle.com/datasets/quandang/vietnamese-foods): nguồn thay thế.
