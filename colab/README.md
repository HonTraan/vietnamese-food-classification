# Chạy trên Google Colab

Thư mục này chứa mọi thứ để **chạy lại và so sánh mô hình trên Google Colab bản miễn phí**. Mục đích là có bằng chứng cho buổi thuyết trình: đã thử nhiều mạng và nhiều cấu hình trên **cùng một tập test**, và mạng được chọn tốt hơn ở những tiêu chí cụ thể.

> Code gốc trong `src/`, `app.py`, `README.md`, `outputs/` **không bị sửa**. Code trong `colab/` chỉ import lại `src/`.

Thiết kế: [docs/2026-11-23-colab-design.md](docs/2026-11-23-colab-design.md) · Kế hoạch: [docs/2026-11-23-colab-plan.md](docs/2026-11-23-colab-plan.md)

## Quy trình

```text
sửa code ở local ──> bash colab/sync_to_drive.sh ──> Google Drive ──> chạy notebook trên Colab
      ^                                                                │
      │                                                kết quả ghi vào Drive/runs/
      │                                                                │
      └── chưa ổn: sửa tiếp <── xem xét <── bash colab/pull_results.sh (Drive -> colab/results/)
                                   │
                            ổn: commit + push lên main
```

GitHub chỉ nhận code đã chạy ổn trên Colab. Colab đọc code từ Drive, không đọc từ GitHub.

## Bố cục trên Google Drive

```text
My Drive/DL/vnfood_colab/
├── code/       # bản sao repo; sync_to_drive.sh ghi đè mỗi lần sync (có VERSION.txt)
├── data/       # raw.zip + clean_split_manifest.json (notebook 00 tạo một lần)
└── runs/       # kết quả từng run; sync KHÔNG BAO GIỜ đụng tới
    ├── E1_cnn/ ...            # config.json, history.csv, metrics.json, best.pt, *.png, DONE
    ├── tuning_base.json       # mô hình base đã chốt cho các run T*
    ├── _eda/                  # biểu đồ EDA
    └── _compare/              # bảng so sánh, biểu đồ, final.json
```

Cần còn trống khoảng **9 GB** trên Drive (`raw.zip` khoảng 6 GB, checkpoint mỗi run 20–300 MB).

## Chạy trên Colab

Mở notebook từ Drive (`My Drive/DL/vnfood_colab/code/colab/notebooks/…` → *Open with Google Colaboratory*), bật GPU (*Runtime → Change runtime type → T4 GPU*). Ở mỗi notebook, **luôn chạy cell 0 trước**. Cell này mount Drive, in `VERSION.txt`, copy code về `/content/vnfood` và kiểm tra GPU.

| Notebook | Làm gì | Thời gian ước tính |
|---|---|---|
| `00_setup_data` | pytest → tải 30 lớp → loại trùng → so split với repo → lưu data lên Drive → EDA | 20–40 phút (một lần) |
| `01_experiments` | mỗi cell là một run; chạy rải trong nhiều ngày | 30–45 phút/run |
| `02_compare` | bảng + biểu đồ so sánh, chọn mô hình cuối, temperature scaling, 15 ảnh sai | 5–10 phút |
| `03_demo` | Streamlit + link public `trycloudflare.com` | 2–3 phút |

**Khi Colab bị ngắt:** mở lại notebook, chạy cell 0, cell 1, rồi chạy lại **đúng cell đang dở**. Run sẽ tiếp tục từ epoch đã lưu (`checkpoint_last.pt`). Run đã xong in `SKIP ... (DONE)`.

**Kiểm tra version:** `VERSION.txt` có `content_hash` (hash của toàn bộ file code). Cell 0 tự tính lại hash trên Colab và dừng nếu lệch. Lệch nghĩa là Google Drive for Desktop chưa upload xong: đợi 1–2 phút rồi chạy lại cell 0. Nếu code được sync lại trong lúc notebook đang mở, chọn *Runtime → Restart session* rồi chạy lại từ cell 0.

## Chạy trên máy (Mac M4)

Hết quota GPU Colab thì chạy tiếp trên máy, **vẫn bằng các notebook này** (`01`, `02`, `03`). Cell 0 tự nhận biết đang ở Colab hay ở máy. Notebook `00` chỉ chạy trên Colab.

1. Cài môi trường một lần: `bash colab/setup_local.sh`. Script tạo `.venv` (Python 3.12) và kernel Jupyter `vnfood (.venv)`.
2. Bật **Google Drive for Desktop**. Kết quả vẫn ghi vào `My Drive/DL/vnfood_colab/runs/`, nên Colab cũng thấy.
3. Mở `colab/notebooks/01_experiments.ipynb` trong VS Code hoặc Jupyter và chọn kernel `vnfood (.venv)`. Chạy cell 0, rồi cell 1. Lần đầu, cell 1 tải `raw.zip` (6 GB) từ Drive và giải nén vào `data/raw`. Manifest được để ở `data/`, không đụng tới `outputs/` của Hoàng.
4. Chạy cell **Chạy thử tốc độ** để biết mỗi run mất bao lâu trên máy. Muốn đổi số worker đọc ảnh thì đặt `VNFOOD_WORKERS`.
5. Chạy **C1** trước, sau đó là T1 → T3 → T2 → T4a → T4b → T5.

**Vì sao cần C1:** E1–E4 chạy trên T4 với fp16, còn trên máy thì chạy bằng GPU `mps` với fp32. C1 là base chạy lại y hệt trên máy. Mọi run T* chạy trên máy được so với C1, nên mỗi T* chỉ khác đúng một yếu tố. Hiệu số base − C1 cho biết ảnh hưởng của phần cứng. Bảng so sánh có cột `device`. Không so tốc độ giữa run chạy trên T4 và run chạy trên M4.

Không chạy cùng một run ở hai nơi (Colab và máy) cùng lúc.

## Các thử nghiệm

Cấu hình chung: split sạch 17.407 / 3.733 / 3.746, seed 42, ảnh 224, AMP, AdamW lr 3e-4, wd 1e-4, cosine, tối đa 12 epoch, early stopping patience 4, checkpoint chọn theo validation accuracy.

| ID | Thử nghiệm | Câu hỏi trả lời được |
|---|---|---|
| E1 | CNN 4 block tự xây (train từ đầu, lr 1e-3, 15 epoch) | Mạng tự xây đạt được bao nhiêu? |
| E2 | ResNet-50 fine-tune | Kiến trúc kinh điển, lớn (~25M tham số) |
| E3 | MobileNetV3-Large fine-tune | Mạng nhẹ, nhanh (~5M) |
| E4 | EfficientNet-B0 fine-tune | Ứng viên chính (~5M) |
| E5 | ConvNeXt-Tiny fine-tune (tùy chọn) | Kiến trúc hiện đại (~28M) |
| C1 | base chạy lại trên máy M4 (đối chứng) | Phần cứng ảnh hưởng bao nhiêu? Mốc so sánh cho T* chạy trên máy |
| T1 | base + freeze backbone | Fine-tune toàn bộ có đáng không? |
| T2 | base không pretrained | Transfer learning giúp bao nhiêu? |
| T3 | base không augmentation | Augmentation có giảm overfit không? |
| T4a/T4b | base với lr 1e-3 / 1e-4 | Learning rate ảnh hưởng thế nào? |
| T5 | base + label smoothing 0.1 | Có cải thiện / calibration tốt hơn không? |
| S123/S2024 | base với seed khác (tùy chọn) | Kết quả có ổn định không? |

"Base" là mô hình E1–E4 có **validation accuracy** cao nhất, được chốt một lần trong `runs/tuning_base.json`. Mô hình cuối (notebook 02) là run có validation accuracy cao nhất trong mọi run. Tập test chỉ dùng để báo cáo, không dùng để chọn.

Thứ tự ưu tiên khi ít quota: E1 → E4 → E2 → E3 → (C1 nếu chạy trên máy) → T1 → T3 → T2 → T4a → T4b → T5 → E5.

## Lệnh trên máy (macOS, Google Drive for Desktop)

```bash
bash colab/sync_to_drive.sh            # đẩy code lên My Drive/DL/vnfood_colab/code/, in version=...
bash colab/pull_results.sh             # kéo mọi run về colab/results/ (bỏ qua *.pt)
bash colab/pull_results.sh E4_efficientnet_b0   # chỉ kéo một run
bash colab/tests/test_sync.sh          # test hai script trên
python3 colab/tools/build_notebooks.py # build lại .ipynb từ colab/notebooks/src/*.py
.venv/bin/python -m colab.tools.slide_assets   # bảng từng tầng, tóm tắt model, biểu đồ train/val cho slide -> colab/results/_slides/
```

Đặt `VNFOOD_DRIVE_ROOT=/đường/dẫn/vnfood_colab` nếu máy có nhiều tài khoản Drive. Notebook được viết ở `colab/notebooks/src/*.py` (dạng `# %%`). Sửa ở đó rồi build lại, đừng sửa trực tiếp `.ipynb`.

## Ghi chú về code gốc (cho Hoàng)

Những điểm mình phát hiện khi chuyển sang Colab. **Chưa sửa gì trong code của bạn**, chỉ ghi lại để bạn xem:

1. `outputs/metrics.json` đang là kết quả của **seed 2024** (test accuracy 88.68%), vì `src/run_multi_seed.py` ghi đè file này ở mỗi seed. Còn `README.md:362` ghi 89.08%, là số của seed 42 (`outputs/metrics_seed_42.json`). Có thể `models/best_model.pt` cũng là của seed 2024.
2. `README.md:517` ghi "chưa chạy thực tế multi-seed", nhưng đã có `outputs/multi_seed_results.json` (mean 88.73% ± 0.27%).
3. Lệnh train ở `README.md:253` và `README.md:325` thiếu `--fine-tune`, nên chỉ train classifier (backbone bị freeze), không ra được kết quả 89% như README.
4. `src/train.py:116-131`: khi không có manifest, `--class-weighted` không có tác dụng (`counts` luôn rỗng, `class_weights` vẫn là `None`) mà không báo gì.
5. README viết cho Windows/PowerShell (`README.md:118`, `:172`). Trên Colab hay macOS các lệnh `.\.venv\Scripts\Activate.ps1` không chạy được.
