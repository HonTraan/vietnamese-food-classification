# Colab Migration & Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tạo thư mục `colab/` để chạy lại pipeline phân loại 30 món ăn Việt trên Colab miễn phí, so sánh ≥ 4 kiến trúc + các thử nghiệm tinh chỉnh, chạy tiếp được khi Colab ngắt, và có demo.

**Architecture:** Code Python mới nằm trong package `colab/`, chỉ *import* (không sửa) `src/` của bạn Hoàng. Code được đồng bộ từ máy lên Drive bằng `rsync` (Drive for Desktop). Notebook Colab copy code từ Drive về `/content/vnfood`, chạy, và ghi kết quả vào `Drive/.../runs/<run_id>/`. Notebook được viết dưới dạng file `.py` theo kiểu `# %%`, rồi build ra `.ipynb` bằng một script chỉ dùng thư viện chuẩn.

**Tech Stack:** Python 3 (Colab), PyTorch + torchvision (có sẵn trên Colab), scikit-learn, matplotlib, seaborn, `datasets`, Streamlit + cloudflared, bash/rsync (macOS).

**Spec:** `colab/docs/2026-11-23-colab-design.md`

## Global Constraints

- **Không sửa** `src/`, `app.py`, `README.md`, `outputs/`, `tests/`, `requirements.txt`, `.gitignore`. Mọi file mới nằm dưới `colab/`.
- **Không chạy train/test PyTorch trên máy người dùng** (máy local cũng không có torch). Ở local chỉ chạy: `python3 -m py_compile`, script build notebook, các script bash sync/pull, và test bash của chúng. Test pytest chạy trên Colab (cell đầu notebook 00).
- Thư mục Drive: `My Drive/DL/vnfood_colab/` gồm `code/` (bị ghi đè khi sync), `data/`, `runs/` (sync không bao giờ đụng tới).
- Trên local: `~/Library/CloudStorage/GoogleDrive-*/My Drive/DL/vnfood_colab`; có thể đổi bằng biến môi trường `VNFOOD_DRIVE_ROOT`. Trên Colab: `/content/drive/MyDrive/DL/vnfood_colab`. Repo trên Colab: `/content/vnfood`.
- Cấu hình chung: seed 42, ảnh 224, AMP khi có CUDA, chọn checkpoint theo validation accuracy, patience 4, tối đa 12 epoch (CNN 15), batch 32, `num_workers=2`. Mô hình pretrained: AdamW lr 3e-4, wd 1e-4, cosine. CNN: lr 1e-3, label smoothing 0.05 (giống `src/cnn_baseline.py`).
- Thí nghiệm: E1 `cnn`, E2 `resnet50`, E3 `mobilenet_v3_large`, E4 `efficientnet_b0`, E5 `convnext_tiny` (tùy chọn). Tinh chỉnh trên base: T1 freeze, T2 không pretrained, T3 không augmentation, T4a lr 1e-3, T4b lr 1e-4, T5 label smoothing 0.1.
- Chỉ chọn mô hình/temperature theo **validation**; test chỉ để báo cáo.
- `pull_results.sh` không bao giờ copy `*.pt`. Toàn cục `.gitignore` đã chặn `*.pt`.
- Chỉ push lên `main` sau khi người dùng chạy Colab ổn (Task 10). Các task trước chỉ commit ở local.
- Commit kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Colab ngắt giữa epoch:** chạy lại cell phải tiếp tục từ `checkpoint_last.pt` (đúng epoch, history, best, stale, optimizer, scheduler, scaler), không train lại từ đầu và không ghi đè `best.pt` bằng mô hình kém hơn. → test `test_resume_continues_from_last_epoch` (Task 5).
2. **Drive chưa sync xong** nên Colab chạy code cũ. → `VERSION.txt` + in ra ở cell setup; test `test_read_version` (Task 2) và `test_sync_writes_version` (Task 1).
3. **Sync lỡ xoá kết quả:** `rsync --delete` chỉ được phép nhắm vào `.../vnfood_colab/code`. → test `test_sync_refuses_wrong_target` và `test_sync_keeps_runs` (Task 1).
4. **Run tinh chỉnh dùng sai base** vì E5 xong muộn làm đổi "winner". → base được ghi một lần vào `runs/tuning_base.json` rồi giữ cố định; test `test_tuning_base_is_pinned` (Task 6).
5. **Checkpoint EfficientNet không chạy được với `app.py` gốc.** → test `test_efficientnet_checkpoint_loads_in_src_build_model` (Task 3) và checkpoint có các key `model_state`, `classes`, `temperature` (Task 6).

---

## File Structure

```text
colab/
├── __init__.py
├── session.py           # đường dẫn, GPU check, restore/save data, so manifest, đọc VERSION (stdlib + torch tùy chọn)
├── models_zoo.py        # build(arch, ...) / head(model, arch) / count_params
├── experiments.py       # RunConfig, EXPERIMENTS, tuning(), seed_runs(), run_state(), pin_tuning_base()
├── train_arch.py        # ManifestDataset, train(cfg, ...), CLI
├── evaluation.py        # evaluate_model(), measure_speed(), write_artifacts()
├── compare.py           # collect(), pick_best(), write_comparison(), plots
├── analyze.py           # calibrate_run(), misclassified_gallery()
├── demo_app.py          # Streamlit app cho checkpoint bất kỳ kiến trúc
├── sync_to_drive.sh
├── pull_results.sh
├── tools/build_notebooks.py
├── notebooks/src/{00_setup_data,01_experiments,02_compare,03_demo}.py
├── notebooks/{00_setup_data,01_experiments,02_compare,03_demo}.ipynb   # sinh ra
├── tests/               # pytest, chạy trên Colab; test_sync.sh chạy local
├── results/.gitkeep
└── README.md
```

---

### Task 1: Script sync/pull + test bash

**Files:**
- Create: `colab/sync_to_drive.sh`, `colab/pull_results.sh`, `colab/tests/test_sync.sh`, `colab/results/.gitkeep`

**Interfaces:**
- Produces: `VNFOOD_DRIVE_ROOT` env (mặc định resolve glob `~/Library/CloudStorage/GoogleDrive-*/My Drive/DL/vnfood_colab`, lỗi nếu 0 hoặc >1 match). `sync_to_drive.sh` ghi `code/VERSION.txt` gồm các dòng `version=<short-hash>[-dirty]-<YYYYmmddHHMMSS>`, `commit=<full-hash>`, `dirty=yes|no`, `synced_at=<ISO8601>`, và in dòng `version=...` ra stdout. `pull_results.sh [run_id]`.

- [ ] **Step 1: Viết `colab/tests/test_sync.sh`.** Tạo thư mục tạm qua `mktemp -d`, đặt `VNFOOD_DRIVE_ROOT=$tmp/vnfood_colab`, tạo sẵn `$tmp/vnfood_colab/runs/E1_cnn/metrics.json` và `best.pt`. Các test (mỗi test là một hàm bash, sai thì `exit 1`):
  - `test_sync_writes_version`: sau sync có `code/VERSION.txt` chứa `^version=`, `^commit=`, `^dirty=`, `^synced_at=`; có `code/src/train.py` và `code/colab/session.py`; **không** có `code/.git`, `code/data`, `code/models`, `code/colab/results`, `code/Deep Learning_Project_HK-2631.pdf`.
  - `test_sync_keeps_runs`: sau sync vẫn còn `runs/E1_cnn/metrics.json`.
  - `test_sync_refuses_wrong_target`: `VNFOOD_DRIVE_ROOT=$tmp/other` thì exit code ≠ 0 và không tạo `$tmp/other/code` (chỉ chấp nhận root có basename `vnfood_colab`).
  - `test_pull_skips_pt`: `pull_results.sh` (với `VNFOOD_RESULTS_DIR=$tmp/results` để không ghi vào repo khi test) copy `metrics.json` và không copy `best.pt`.
- [ ] **Step 2: Chạy `bash colab/tests/test_sync.sh`.** Kết quả mong đợi: FAIL (script chưa tồn tại).
- [ ] **Step 3: Viết hai script.** `set -euo pipefail`. Repo root = `git rev-parse --show-toplevel`. `rsync -a --delete` với các exclude `.git/ data/ models/ colab/results/ __pycache__/ .venv/ .pytest_cache/ *.pdf .DS_Store`, đích `"$VNFOOD_DRIVE_ROOT/code/"`. `pull_results.sh`: `rsync -a --exclude '*.pt' "$VNFOOD_DRIVE_ROOT/runs/${1:-}" "${VNFOOD_RESULTS_DIR:-<repo>/colab/results}/"`.
- [ ] **Step 4: Chạy `bash colab/tests/test_sync.sh`.** Kết quả mong đợi: in `4 passed`.
- [ ] **Step 5: Commit** `feat(colab): add Drive sync and result pull scripts`.

### Task 2: `colab/session.py`

**Files:** Create `colab/__init__.py`, `colab/session.py`, `colab/tests/conftest.py` (thêm repo root vào `sys.path`), `colab/tests/test_session.py`

**Interfaces — Produces:**
- `DRIVE_ROOT = Path("/content/drive/MyDrive/DL/vnfood_colab")`, `REPO = Path(__file__).resolve().parents[1]`, `RUNS_DIR = DRIVE_ROOT / "runs"`, `DATA_DIR = DRIVE_ROOT / "data"`
- `read_version(path: Path) -> dict[str, str]`
- `compare_manifests(a: dict, b: dict) -> dict` → `{"same": bool, "counts_a": {split: n}, "counts_b": {...}, "only_in_a": {split: n}, "only_in_b": {split: n}}` với các split `train`, `validation`, `test`
- `check_gpu() -> str` (tên GPU; ném `RuntimeError("Chưa bật GPU: Runtime > Change runtime type > T4 GPU")` nếu không có CUDA)
- `save_data(repo: Path = REPO, data_dir: Path = DATA_DIR) -> None`: zip `repo/data/raw` (ZIP_STORED) thành `data_dir/raw.zip`, copy manifest thành `data_dir/clean_split_manifest.json`
- `restore_data(repo: Path = REPO, data_dir: Path = DATA_DIR) -> bool`: nếu có `raw.zip` thì giải nén vào `repo/data/`, copy manifest vào `repo/outputs/`, trả về True; nếu không có thì trả về False

- [ ] **Step 1: Viết test.** `test_read_version` (file 4 dòng thì `["version"]` đúng chuỗi), `test_compare_manifests_same`, `test_compare_manifests_diff` (thêm 1 path vào test của b thì `same is False`, `only_in_b["test"] == 1`), `test_save_restore_roundtrip` (dùng `tmp_path`: tạo `repo/data/raw/A/x.jpg` + `repo/outputs/clean_split_manifest.json`, chạy save, xoá `data/`, restore thì file trở lại và hàm trả về True; `restore_data` với thư mục trống trả về False).
- [ ] **Step 2: Chạy `python3 -m py_compile colab/session.py colab/tests/test_session.py`** (test chạy trên Colab ở Task 10).
- [ ] **Step 3: Implement.** `check_gpu` import torch bên trong hàm, để module vẫn import được khi không có torch.
- [ ] **Step 4: py_compile pass.**
- [ ] **Step 5: Commit** `feat(colab): add session helpers`.

### Task 3: `colab/models_zoo.py`

**Files:** Create `colab/models_zoo.py`, `colab/tests/test_models_zoo.py`

**Interfaces — Produces:**
- `ARCHS = ("cnn", "resnet50", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny")`
- `build(arch: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module`. `cnn` dùng `src.cnn_baseline.CNNBaseline`; `pretrained`/`freeze` với `cnn` thì `ValueError`. Thay head: resnet50 `fc`; mobilenet `classifier[3]`; efficientnet `classifier[1] = Sequential(Dropout(0.3), Linear)` (y hệt `src/train.build_model`); convnext `classifier[2]`. Weights: `<Model>_Weights.DEFAULT`. Arch lạ thì `ValueError`.
- `head(model: nn.Module, arch: str) -> nn.Module` (`fc` hoặc `classifier`)
- `count_params(model) -> int`

- [ ] **Step 1: Viết test.** `test_output_shape[arch]` (parametrize mọi ARCHS, `pretrained=False` trừ cnn, input `(2,3,224,224)` thì output `(2,30)`); `test_freeze_only_head_trainable` (với 4 arch pretrained-capable, `pretrained=False, freeze_backbone=True`: mọi param có `requires_grad` đều thuộc `head(...)`); `test_cnn_rejects_pretrained`; `test_unknown_arch`; `test_efficientnet_checkpoint_loads_in_src_build_model` (`src.train.build_model(30, pretrained=False).load_state_dict(build("efficientnet_b0", 30, pretrained=False).state_dict())` chạy không lỗi).
- [ ] **Step 2–4: py_compile → implement → py_compile.**
- [ ] **Step 5: Commit** `feat(colab): add model zoo`.

### Task 4: `colab/experiments.py`

**Files:** Create `colab/experiments.py`, `colab/tests/test_experiments.py`

**Interfaces — Produces:**
- `@dataclass(frozen=True) RunConfig`: `run_id: str, arch: str, pretrained: bool = True, freeze_backbone: bool = False, augment: bool = True, lr: float = 3e-4, label_smoothing: float = 0.0, epochs: int = 12, batch_size: int = 32, patience: int = 4, seed: int = 42`; `to_dict()`.
- `EXPERIMENTS: dict[str, RunConfig]`: các key `E1..E5`, `run_id = f"{key}_{arch}"`. E1: `arch="cnn", pretrained=False, lr=1e-3, label_smoothing=0.05, epochs=15`.
- `tuning(base: RunConfig) -> dict[str, RunConfig]`: các key `T1, T2, T3, T4a, T4b, T5`, mỗi cái chỉ khác base đúng một trường. `run_id` lần lượt: `T1_<arch>_frozen`, `T2_<arch>_scratch`, `T3_<arch>_noaug`, `T4a_<arch>_lr1e-3`, `T4b_<arch>_lr1e-4`, `T5_<arch>_ls0.1`.
- `seed_runs(base: RunConfig, seeds=(123, 2024)) -> dict[str, RunConfig]`: key `S123`, `S2024`; `run_id = f"S{seed}_{arch}"`.
- `run_state(run_dir: Path) -> Literal["new", "resume", "done"]`: có `DONE` thì done; có `checkpoint_last.pt` thì resume; còn lại là new.
- Thứ tự ưu tiên: `PRIORITY = ["E1", "E4", "E2", "E3", "T1", "T3", "T2", "T4a", "T4b", "T5", "E5"]`.
- `experiments.py` không import `compare.py`. `pin_tuning_base` nằm ở Task 6 để tránh vòng import.

- [ ] **Step 1: Viết test.** `test_experiment_ids`; `test_tuning_differs_in_one_field` (so `to_dict()` bỏ qua `run_id`, mỗi T khác đúng 1 key, đúng giá trị trong Global Constraints); `test_run_state` (tmp_path: new / resume / done).
- [ ] **Step 2–4: py_compile → implement → py_compile.**
- [ ] **Step 5: Commit** `feat(colab): add experiment registry`.

### Task 5: `colab/train_arch.py` + `colab/evaluation.py`

**Files:** Create `colab/train_arch.py`, `colab/evaluation.py`, `colab/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `models_zoo.build/count_params`, `experiments.RunConfig/run_state/EXPERIMENTS/tuning/seed_runs`, `src.dataset.train_transform/eval_transform`.
- Produces:
  - `ManifestDataset(paths: list[str], classes: list[str], root: Path, transform)`: label = `Path(p).parts[-2]`, ảnh `convert("RGB")`.
  - `train(cfg: RunConfig, runs_dir: Path, manifest: dict, root: Path, device: str | None = None, max_batches: int | None = None) -> Path` (trả về run_dir). `max_batches` chỉ dùng cho smoke test.
  - Mỗi epoch ghi `checkpoint_last.pt` = `{model_state, optimizer_state, scheduler_state, scaler_state, epoch, best_val_accuracy, stale_epochs, history}`. Khi val cải thiện thì ghi `best.pt` = `{model_state, classes, arch, config, best_val_accuracy, epoch, test_paths}`. Khi kết thúc: load `best.pt`, gọi `evaluation.write_artifacts`, ghi `DONE`, xoá `checkpoint_last.pt`.
  - `config.json`: `cfg.to_dict()` + `version` (đọc `REPO/VERSION.txt` nếu có) + `gpu` + `started_at`.
  - `history.csv`: `epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr, epoch_seconds`.
  - `evaluation.evaluate_model(model, loader, device, classes) -> dict` gồm `test_accuracy, macro_f1, top3_accuracy, test_loss, report, confusion_matrix`, cùng `rows` (dữ liệu cho `test_predictions.csv`: `image_path, true_label, predicted_label, confidence, correct, top3`).
  - `evaluation.measure_speed(model, device) -> {"gpu_images_per_sec": float | None, "cpu_ms_per_image": float}`: warm-up 5 lần; GPU đo 20 batch × 32; CPU đo 20 lần, batch 1.
  - `evaluation.write_artifacts(run_dir, metrics, history, classes)` ghi `metrics.json` (thêm `params`, `train_seconds`, `best_val_accuracy`, speed), `test_predictions.csv`, `learning_curves.png`, `confusion_matrix.png`.
  - CLI: `python -m colab.train_arch --exp E4 [--tune T1] [--seed-run S123] --runs-dir PATH`. Với `--tune`, base đọc từ `runs_dir/tuning_base.json`. Run đã `done` thì in `SKIP <run_id> (DONE)` rồi thoát với code 0.
- Train: augment=False thì dùng `eval_transform` cho train. Freeze thì gọi `.eval()` cho mọi module ngoài head mỗi epoch (giữ BN). AMP: `torch.autocast("cuda", dtype=torch.float16)` + `torch.amp.GradScaler`, chỉ bật khi device là cuda. Seed: `random/np/torch.manual_seed(cfg.seed)`. Scheduler: CNN dùng `ReduceLROnPlateau(mode="max", factor=.5, patience=1)` như repo; còn lại `CosineAnnealingLR(T_max=epochs)`.

- [ ] **Step 1: Viết `test_train_smoke.py`.** Fixture tạo `tmp/data/raw/{A,B}/` mỗi lớp 6 ảnh PIL ngẫu nhiên 64×64, manifest chia 2/2/2 mỗi lớp. Các test:
  - `test_train_cnn_creates_artifacts`: `cfg=replace(EXPERIMENTS["E1"], epochs=1, batch_size=2)`, `device="cpu"`. Kết quả: có `DONE`, `best.pt`, `metrics.json` (có các key `test_accuracy, macro_f1, top3_accuracy, params, cpu_ms_per_image`), `history.csv` có 1 dòng; không còn `checkpoint_last.pt`.
  - `test_resume_continues_from_last_epoch`: train `epochs=2`, `max_batches=1`; monkeypatch `colab.train_arch._after_epoch` (hook no-op `_after_epoch(epoch: int) -> None`, gọi ngay sau khi ghi `checkpoint_last.pt`) để ném `KeyboardInterrupt` khi `epoch == 1`; gỡ monkeypatch. Chạy lại `train` thì `history.csv` có đúng epoch `[1, 2]` (không có epoch 1 lặp lại), và có `DONE`.
  - `test_skip_when_done`: gọi train lần 2 trên run đã DONE thì không sửa `metrics.json` (mtime giữ nguyên).
- [ ] **Step 2–4: py_compile → implement → py_compile.**
- [ ] **Step 5: Commit** `feat(colab): add resumable multi-arch trainer`.

### Task 6: `colab/compare.py` + `colab/analyze.py`

**Files:** Create `colab/compare.py`, `colab/analyze.py`, `colab/tests/test_compare.py`

**Interfaces — Produces:**
- `collect(runs_dir: Path) -> list[dict]`: mỗi run đã DONE cho một dòng gồm `run_id, exp (phần trước "_"), arch, pretrained, freeze_backbone, augment, lr, label_smoothing, seed, best_val_accuracy, test_accuracy, macro_f1, top3_accuracy, params, train_seconds, gpu_images_per_sec, cpu_ms_per_image`. Sắp xếp theo `PRIORITY` (các run S* xếp cuối).
- `pick_best(rows, prefix: str | None = None) -> dict`: cao nhất theo `best_val_accuracy`; hoà thì so `macro_f1`, rồi `run_id` nhỏ hơn. **Không bao giờ** dùng test.
- `pin_tuning_base(runs_dir: Path, rows: list[dict]) -> RunConfig`: nếu đã có `runs_dir/tuning_base.json` thì đọc lại; nếu chưa có thì yêu cầu E1–E4 đều đã DONE (thiếu thì `RuntimeError` liệt kê run thiếu), chọn `pick_best(rows, prefix="E")`, lấy `EXPERIMENTS[exp]`, rồi ghi file `{"base_exp": "E4", "run_id": ..., "config": ...}`. `train_arch --tune` đọc `config` trong file này.
- `write_comparison(rows, out_dir: Path) -> None`: ghi `comparison.csv`, `comparison_arch.png` (accuracy + macro F1 của E*), `accuracy_vs_params.png`, `learning_curves_all.png` (val accuracy của E* chồng lên nhau), `tuning.png` (T* so với base). Bỏ qua biểu đồ nào không có dữ liệu.
- `analyze.calibrate_run(run_dir: Path, manifest: dict, root: Path, device) -> dict`: thu logits val/test, fit temperature bằng LBFGS (giống `src/calibrate.py`, clamp ≥ 0.05), metric dùng `src.calibrate.calibration_metrics`, ghi `run_dir/calibration.json` (cấu trúc giống `outputs/calibration.json`), và ghi `temperature` vào `best.pt`.
- `analyze.misclassified_gallery(run_dir: Path, root: Path, top_n: int = 15) -> Path`: lấy top-n ảnh sai có confidence cao nhất từ `test_predictions.csv`, ghi `misclassified_top15.png` + `misclassified_top15.csv`.

- [ ] **Step 1: Viết test.** `test_collect_skips_unfinished` (run có metrics nhưng không có DONE thì bị bỏ qua); `test_pick_best_uses_validation_not_test` (run A val .90/test .80, run B val .85/test .95 thì chọn A); `test_pick_best_prefix`; `test_tuning_base_is_pinned` (ghi lần 1 với rows có E4 tốt nhất; gọi lần 2 với rows mà E5 tốt hơn thì vẫn trả về base E4); `test_pin_requires_E1_to_E4_done`; `test_write_comparison_creates_csv` (2 dòng E giả thì có `comparison.csv` với header đúng).
- [ ] **Step 2–4: py_compile → implement → py_compile.**
- [ ] **Step 5: Commit** `feat(colab): add comparison and calibration reports`.

### Task 7: `colab/demo_app.py`

**Files:** Create `colab/demo_app.py`, `colab/tests/test_demo.py`

**Interfaces — Produces:** `load_model(path: Path) -> tuple[nn.Module, list[str], float]` (dùng `checkpoint["arch"]`, mặc định `"efficientnet_b0"` nếu thiếu); `predict(image, model, classes, temperature, top_k=3) -> list[tuple[str, float]]`. Giao diện Streamlit giống `app.py` (tiêu đề, slider ngưỡng 0.50–0.95, mặc định 0.70, top-3, bar chart). Đường dẫn checkpoint lấy từ env `VNFOOD_MODEL`, mặc định `REPO/models/final.pt`. `@st.cache_resource` cho `load_model`.

- [ ] **Step 1: Viết test.** `test_predict_probabilities_sum` (lưu `best.pt` giả từ `build("mobilenet_v3_large", 30, pretrained=False)`, temperature 1.5; `predict` trả về 3 phần tử, giảm dần, tổng ≤ 1). Import module không được chạy giao diện: phần UI đặt trong `main()` và gọi bằng `if __name__ == "__main__"` hoặc khi `streamlit` chạy, kiểm tra qua `st.runtime.exists()`.
- [ ] **Step 2–4: py_compile → implement → py_compile.**
- [ ] **Step 5: Commit** `feat(colab): add architecture-agnostic demo app`.

### Task 8: Notebooks + builder

**Files:** Create `colab/tools/build_notebooks.py`, `colab/notebooks/src/*.py`, và 4 file `.ipynb` sinh ra; `colab/tests/test_build_notebooks.py`

**Interfaces:** `build(src: Path, dst: Path) -> None`: tách cell ở `# %%` (code) và `# %% [markdown]` (các dòng `# ` bỏ tiền tố). Ghi nbformat 4.5, kernel `python3`, metadata `"accelerator": "GPU"` và `"colab": {"provenance": []}`. CLI không tham số thì build cả 4 file.

**Cell setup chung** (dán y hệt ở đầu mỗi notebook source, vì chưa import được `colab.session` trước khi copy code):
```python
from google.colab import drive; drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/DL/vnfood_colab"
print(open(f"{DRIVE}/code/VERSION.txt").read())   # so với version Claude báo
!rsync -a --delete {DRIVE}/code/ /content/vnfood/
%cd /content/vnfood
!pip -q install datasets streamlit pytest
import sys; sys.path.insert(0, "/content/vnfood")
from colab import session; print("GPU:", session.check_gpu())
```

**Nội dung từng notebook** (markdown tiếng Việt giải thích mỗi bước):
- `00_setup_data`: setup → `!python -m pytest -q colab/tests` → nếu `session.restore_data()` trả về False thì chạy `src/dataset.py`, `src/data_cleaning.py --apply`, `session.save_data()` → *trước khi* chạy cleaning, chép manifest của repo thành `outputs/clean_split_manifest.repo.json`; sau cleaning so hai file bằng `compare_manifests` rồi in kết quả → `src/eda.py`, copy `outputs/figures/class_distribution.png` và `sample_images.png` vào `runs/_eda/`.
- `01_experiments`: setup → `restore_data()` (False thì báo "chạy notebook 00 trước") → in bảng `run_state` cho PRIORITY → mỗi cell một lệnh `!python -m colab.train_arch --exp E1 --runs-dir {DRIVE}/runs` (E1, E4, E2, E3) → cell `pin_tuning_base` → mỗi cell một `--tune T1` … `T5` → cell tùy chọn E5 → cell tùy chọn `--seed-run S123/S2024`.
- `02_compare`: setup → restore → `write_comparison(collect(runs), runs/_compare)` + hiển thị bảng → `best = pick_best(rows)` → `calibrate_run` + `misclassified_gallery` cho best → ghi `runs/_compare/final.json` `{run_id, arch, reason: "best validation accuracy"}`.
- `03_demo`: setup → copy `runs/<final>/best.pt` thành `models/final.pt`. Nếu arch là `efficientnet_b0` thì copy thêm thành `models/best_model.pt` và chạy `app.py`; ngược lại chạy `colab/demo_app.py`. Tải `cloudflared` (linux-amd64 release), chạy nền Streamlit cổng 8501 + `cloudflared tunnel --url http://localhost:8501`, đọc log in ra link `trycloudflare.com`.

- [ ] **Step 1: Viết test `test_build_notebooks.py`** (stdlib, chạy được bằng `python3 -m unittest` ở local): build một source 3 cell (markdown, code, code) thì JSON có `nbformat == 4`, đúng 3 cell, đúng loại cell, source markdown không còn `# `.
- [ ] **Step 2: Chạy `python3 -m unittest colab/tests/test_build_notebooks.py`.** Kết quả mong đợi: FAIL.
- [ ] **Step 3: Implement builder + viết 4 notebook source.**
- [ ] **Step 4: Chạy `python3 -m unittest colab/tests/test_build_notebooks.py && python3 colab/tools/build_notebooks.py && python3 -c "import json,glob;[json.load(open(f)) for f in glob.glob('colab/notebooks/*.ipynb')];print('ok')"`.** Kết quả mong đợi: tests OK, in `ok`, có 4 file `.ipynb`.
- [ ] **Step 5: Commit** `feat(colab): add Colab notebooks`.

### Task 9: `colab/README.md`

**Files:** Create `colab/README.md`

- [ ] **Step 1: Viết README (tiếng Việt).** Gồm: mục đích (một đoạn); quy trình sơ đồ (copy từ spec); yêu cầu (Colab miễn phí bật T4; Drive còn trống khoảng ≥ 6 GB cho `raw.zip` + checkpoint); từng bước chạy 00 → 01 → 02 → 03 và thời gian ước tính; khi Colab ngắt thì chạy lại đúng cell đó; kiểm tra `VERSION.txt`; bảng thí nghiệm E*/T* với câu hỏi mà mỗi thí nghiệm trả lời; cách lệnh sync/pull hoạt động (dành cho người dùng máy Mac); mục **"Ghi chú về code gốc (cho Hoàng)"** với 5 điểm trong spec §5.7, mỗi điểm kèm `file:line` (`src/train.py:116-131` cho class-weighted, README dòng 254 cho lệnh thiếu `--fine-tune`, README dòng 518 cho multi-seed).
- [ ] **Step 2: Chạy `bash colab/tests/test_sync.sh && python3 -m py_compile colab/*.py`** để kiểm tra lại toàn bộ.
- [ ] **Step 3: Commit** `docs(colab): add Colab README`.

### Task 10: Sync + chạy kiểm chứng trên Colab (người dùng chạy)

- [ ] **Step 1:** `bash colab/sync_to_drive.sh`, báo cho người dùng chuỗi `version=...`.
- [ ] **Step 2: Người dùng chạy `00_setup_data`.** Kết quả mong đợi: pytest pass hết; `compare_manifests.same == True` (17.407/3.733/3.746); có `data/raw.zip`. Không đạt thì người dùng gửi output, quay lại sửa (superpowers:systematic-debugging) → sync → chạy lại.
- [ ] **Step 3: Người dùng chạy cell E1 của `01_experiments`** (run ngắn nhất, kiểm tra trọn chu trình). Sau đó Claude chạy `bash colab/pull_results.sh E1_cnn` và kiểm tra `metrics.json`/`history.csv` (test accuracy khoảng 60% như repo cũ).
- [ ] **Step 4: Khi ổn:** commit `colab/results/` đã kéo về, rồi `git push origin main`.
- [ ] **Step 5:** Các run còn lại (E4 → … theo PRIORITY) là vòng lặp chạy → pull → xem xét ở các phiên sau, không thuộc phạm vi plan này.
