# Local M4 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho cùng một bộ notebook chạy được cả trên Colab lẫn trên máy M4 (GPU `mps`), có run đối chứng `C1`, để chạy nốt T1–T5 trên máy với kết quả ghi vào `vnfood_colab/runs/` trên Drive.

**Architecture:**
- Cell 0 của mỗi notebook chỉ còn một đoạn khởi động ngắn: trên Colab thì mount Drive và copy code, trên máy thì tìm repo. Sau đó gọi `session.start(repo)`, hàm này lo phần còn lại theo môi trường và trả về các đường dẫn và thiết bị.
- `train_arch` chọn thiết bị theo thứ tự `cuda` → `mps` → `cpu`. Thêm `--control C1`.
- `compare` lấy C1 làm mốc cho biểu đồ tinh chỉnh.

**Tech Stack:** Python 3.12 (`uv`, `.venv`), PyTorch có MPS, cộng các thư viện như plan cũ.

**Spec:** `colab/docs/2026-11-26-local-m4-design.md` (bổ sung cho `colab/docs/2026-11-23-colab-design.md`)

## Global Constraints

- Không sửa `src/`, `app.py`, `README.md`, `outputs/`, `tests/`, `requirements.txt`, `.gitignore`. Ở máy cũng **không ghi** vào `outputs/clean_split_manifest.json`.
- **Claude không tự chạy notebook, train hay đánh giá.** Người dùng chạy notebook.
- Claude chỉ chạy `setup_local.sh` và `pytest` ở máy sau khi người dùng cho phép. Phép này được xin một lần ở bước bàn giao, trước khi triển khai.
- Kiểm tra nhẹ thì Claude tự chạy như trước: `py_compile`, `minirun.py` (chạy test không cần torch), `bash colab/tests/test_sync.sh`, `python3 colab/tools/build_notebooks.py`.
- Thứ tự chọn thiết bị là `cuda` → `mps` → `cpu`. Chỉ bật AMP khi có `cuda`. Trên `mps` train bằng fp32.
- Run đối chứng có `run_id` là `C1_<arch>_m4`, tạo từ base đã chốt (`tuning_base.json`) và chỉ đổi `run_id`. Thứ tự chạy ở máy: C1 → T1 → T3 → T2 → T4a → T4b → T5.
- `runs/` nằm trong thư mục Drive. Ở máy, đường dẫn lấy từ `VNFOOD_DRIVE_ROOT`, nếu không có thì tìm đúng một thư mục `~/Library/CloudStorage/GoogleDrive-*/My Drive/DL/vnfood_colab`.
- Ở máy: data nằm ở `<repo>/data/raw`, manifest ở `<repo>/data/clean_split_manifest.json`. Trên Colab, manifest vẫn ở `<repo>/outputs/clean_split_manifest.json` như cũ.
- Notebook gọi train bằng `!"{sys.executable}" -m colab.train_arch ...`, và mọi đường dẫn đều đặt trong ngoặc kép, vì `My Drive` có dấu cách.
- Chỉ push lên `main` sau khi người dùng chạy C1 ổn (Task 6). Commit kết thúc bằng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Đường dẫn có dấu cách** (`.../My Drive/...`) trong lệnh `!` của notebook: phải được đặt trong ngoặc kép, nếu không thì `argparse` nhận sai đường dẫn. Kiểm bằng `test_notebook_shell_calls_are_safe` (Task 5).
2. **`!python` chạy nhầm Python của hệ thống**, không phải `.venv`, nên báo thiếu torch. Phải dùng `sys.executable`. Kiểm bằng cùng test trên (Task 5).
3. **Giải nén 6 GB bị ngắt giữa chừng** để lại `data/raw` dở dang. Lần sau code thấy thư mục đã có nên bỏ qua giải nén và train thiếu ảnh mà không báo gì. Kiểm bằng `test_restore_recovers_from_interrupted_extract` (Task 1).
4. **Chạy ở máy làm git của Hoàng bị đổi**, vì manifest bị chép vào `outputs/`. Kiểm bằng `test_restore_local_manifest_leaves_outputs_alone` (Task 1).
5. **Drive for Desktop không chạy, hoặc máy có nhiều tài khoản Drive:** phải báo lỗi rõ ràng, không được ghi `runs/` vào chỗ lạ. Kiểm bằng `test_drive_root_requires_single_mirror` (Task 1).

---

### Task 1: Hàm theo môi trường trong `session.py` (không cần torch)

**Files:**
- Modify: `colab/session.py`
- Test: `colab/tests/test_session.py`

**Interfaces:**
- Produces:
  - `drive_root(on_colab: bool, env: Mapping[str, str] = os.environ, home: Path = Path.home()) -> Path`
    - Colab: trả về `DRIVE_ROOT`.
    - Có `VNFOOD_DRIVE_ROOT`: trả về đường dẫn đó.
    - Còn lại: trả về đúng một kết quả của glob `home/"Library/CloudStorage"/"GoogleDrive-*"/"My Drive"/"DL"/"vnfood_colab"`.
    - Không có kết quả nào, hoặc có nhiều hơn một: `RuntimeError`. Thông báo có chữ "VNFOOD_DRIVE_ROOT".
  - `choose_device(cuda: bool, mps: bool) -> str`: trả về `"cuda"`, `"mps"` hoặc `"cpu"`.
  - `restore_data(repo=REPO, data_dir=DATA_DIR, manifest_dst: Path | None = None) -> bool`
    - `manifest_dst` mặc định là `repo/"outputs"/MANIFEST_NAME` (giữ hành vi trên Colab).
    - Giải nén vào `repo/"data"/".extract"` (xoá thư mục này nếu còn sót lại), rồi đổi tên `.extract/raw` thành `data/raw`.
    - Chỉ bỏ qua giải nén khi `data/raw` đã có.

- [ ] **Step 1: Viết các test fail:**
  - `test_drive_root_colab`: trả về `session.DRIVE_ROOT`.
  - `test_drive_root_env_override`: trả về `Path(env["VNFOOD_DRIVE_ROOT"])`.
  - `test_drive_root_finds_mirror`: tạo một thư mục mirror giả dưới `tmp_path`, so đúng đường dẫn đó.
  - `test_drive_root_requires_single_mirror`: cả hai trường hợp không có mirror và có hai mirror (`GoogleDrive-a@x`, `GoogleDrive-b@x`) đều `RuntimeError`. Dùng try/except vì `minirun` không có `pytest.raises`.
  - `test_choose_device_order`: `(True, True) → "cuda"`, `(False, True) → "mps"`, `(False, False) → "cpu"`.
  - `test_restore_local_manifest_leaves_outputs_alone`: `restore_data(fresh, data_dir, manifest_dst=fresh/"data"/MANIFEST_NAME)` ghi manifest vào `data/`, và `not (fresh/"outputs").exists()`.
  - `test_restore_recovers_from_interrupted_extract`: tạo sẵn `fresh/"data"/".extract"/"raw"/"junk.jpg"`, restore xong thì `data/raw/A/x.jpg` có đúng nội dung, không có `junk.jpg`, và không còn `.extract`.
- [ ] **Step 2:** `cd <repo> && python3 -I .superpowers/sdd/2026-11-23-colab-plan/minirun.py colab.tests.test_session`
  - Expected: FAIL với `AttributeError: ... 'drive_root'`.
  - Nếu `minirun` không import được vì `conftest`: chạy với `PYTHONPATH=.`.
- [ ] **Step 3:** Viết `drive_root`, `choose_device` và sửa `restore_data` theo Interfaces.
- [ ] **Step 4:** Chạy lại lệnh ở Step 2. Expected: `N passed` (N ≥ 13). `test_save_restore_roundtrip` cũ vẫn phải pass.
- [ ] **Step 5:** Commit: `feat(colab): environment-aware drive root, device choice, safe local restore`.

### Task 2: Môi trường ở máy (`setup_local.sh`)

**Files:**
- Create: `colab/setup_local.sh`
- Test: `colab/tests/test_sync.sh`, thêm `test_setup_local_dry_run`.

**Interfaces:**
- Produces: `bash colab/setup_local.sh [--dry-run]`.
  - Script tạo `<repo>/.venv` bằng `uv venv --python 3.12`, rồi chạy `uv pip install -r requirements.txt ipykernel`, rồi `.venv/bin/python -m ipykernel install --user --name vnfood --display-name "vnfood (.venv)"`.
  - `--dry-run` chỉ in các lệnh, có chữ `uv venv --python 3.12`, và không tạo gì.

- [ ] **Step 1:** Viết test `test_setup_local_dry_run`: output có `uv venv --python 3.12` và `ipykernel install`. Thêm test vào vòng lặp ở cuối file.
- [ ] **Step 2:** `bash colab/tests/test_sync.sh`. Expected: `FAIL: ...`, vì script chưa có.
- [ ] **Step 3:** Viết `colab/setup_local.sh` (`set -euo pipefail`, tìm `REPO` giống cách `sync_to_drive.sh` làm).
- [ ] **Step 4:** `bash colab/tests/test_sync.sh`. Expected: `5 passed`.
- [ ] **Step 5:** Commit: `feat(colab): local .venv setup script`.
- [ ] **Step 6 (cần phép người dùng):**
  - Chạy `bash colab/setup_local.sh`.
  - Sau đó chạy `.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`. Expected: `True`.
  - Rồi chạy `.venv/bin/python -m pytest colab/tests -q > <workspace>/pytest_baseline.txt; tail -3 <workspace>/pytest_baseline.txt`. Expected: `38 passed` cộng 7 test mới của Task 1, tức 45. Đây là lần đầu suite chạy ở máy.

### Task 3: Thiết bị khi train, run đối chứng, CLI

**Files:**
- Modify: `colab/experiments.py`, `colab/train_arch.py`, `colab/evaluation.py`, `colab/session.py` (thêm `device_label`, `check_gpu`)
- Test: `colab/tests/test_experiments.py`, `colab/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `session.choose_device` (Task 1).
- Produces:
  - `control(base: RunConfig) -> dict[str, RunConfig]`: trả về `{"C1": replace(base, run_id=f"C1_{base.arch}_m4")}`.
  - `PRIORITY = ["E1", "E4", "E2", "E3", "C1", "T1", "T3", "T2", "T4a", "T4b", "T5", "E5"]`.
  - CLI: thêm `--control C1`, đọc base từ `tuning_base.json`, giống `--tune`.
  - `session.device_label(device) -> str`:
    - `cuda`: trả về tên GPU.
    - `mps`: trả về `"mps (<chip>)"`, chip lấy từ `sysctl -n machdep.cpu.brand_string`; lỗi thì trả về `"mps"`.
    - Còn lại: `"cpu"`.
  - `session.check_gpu() -> str`: chấp nhận `cuda` hoặc `mps`. Câu lỗi vẫn như cũ khi không có cả hai.
  - `train(...)`:
    - Thiết bị mặc định là `choose_device(torch.cuda.is_available(), torch.backends.mps.is_available())`.
    - `config.json["device"] = device_label(device)`, và `metrics.json` có thêm khóa `"device"`.
  - `_loader`: `num_workers` là 2 với `cuda`, là `int(os.environ.get("VNFOOD_WORKERS", 4))` với `mps`, và 0 với `cpu`. `pin_memory` chỉ bật với `cuda`.
  - `evaluation.measure_speed`: với `mps`, đo giống `cuda` nhưng đồng bộ bằng `torch.mps.synchronize()`. Khóa trả về giữ là `gpu_images_per_sec`.
  - `train_arch.benchmark(cfg, manifest, root, device=None, batches=30) -> dict`:
    - Trả về các khóa `device`, `seconds_per_batch`, `est_epoch_minutes`, `est_run_hours`.
    - Cách tính: lấy thời gian train `batches` batch (bỏ 3 batch đầu làm warm-up), nhân lên cho `len(train)/batch_size` batch train cộng `len(validation)/batch_size` batch validation nhân 0,4, rồi nhân `cfg.epochs`.
    - Không ghi gì vào `runs/`.

- [ ] **Step 1:** Viết các test fail:
  - `test_control_differs_only_in_run_id`: so `to_dict()` của C1 và base, chỉ khác khóa `run_id`, và `run_id == "C1_resnet50_m4"`.
  - Sửa `test_priority_covers_registry` cho có `"C1"`, và thêm điều kiện `PRIORITY.index("C1") < PRIORITY.index("T1")`.
  - `test_config_and_metrics_record_device` (smoke trên cpu): `config.json["device"] == "cpu"` và `metrics.json["device"] == "cpu"`.
  - `test_benchmark_returns_estimates` (tiny data, cpu, `batches=4`): các giá trị đều dương, và thư mục runs không có file nào mới.
- [ ] **Step 2:**
  - `python3 -I .superpowers/sdd/2026-11-23-colab-plan/minirun.py colab.tests.test_experiments`. Expected: FAIL ở `control`.
  - `.venv/bin/python -m pytest colab/tests/test_train_smoke.py -q`. Expected: 2 test mới FAIL.
- [ ] **Step 3:** Viết code theo Interfaces.
- [ ] **Step 4:** `.venv/bin/python -m pytest colab/tests -q`. Expected: toàn bộ pass.
- [ ] **Step 5:** Commit: `feat(colab): mps device, C1 control run, benchmark`.

### Task 4: Mốc so sánh cho biểu đồ tinh chỉnh và cột thiết bị

**Files:**
- Modify: `colab/compare.py`
- Test: `colab/tests/test_compare.py`

**Interfaces:**
- Produces:
  - `FIELDS` có thêm `"device"` ngay sau `"seed"`, lấy từ `config.json` (`config.get("device")`, các run cũ đã có khóa này).
  - `tuning_reference(rows: list[dict], runs_dir: Path) -> dict | None`:
    - Có run `C1*` đã xong: trả về hàng đó.
    - Không có: trả về hàng có `run_id` trùng với `tuning_base.json`.
    - Không có pin: trả về `None`.
  - `tuning.png` dùng `tuning_reference` làm thanh đầu tiên.
  - `pin_tuning_base` và `pick_best` giữ nguyên.

- [ ] **Step 1:** Viết các test fail:
  - `test_tuning_reference_prefers_control`: có E1–E4, có pin E2 và C1 đã xong, thì kết quả là `"C1_resnet50_m4"`.
  - `test_tuning_reference_falls_back_to_base`: chưa có C1 thì kết quả là `"E2_resnet50"`.
  - `test_collect_has_device`.
  - Helper `_run` cần nhận thêm `device=`.
- [ ] **Step 2:** `python3 -I .superpowers/sdd/2026-11-23-colab-plan/minirun.py colab.tests.test_compare`. Expected: FAIL ở `tuning_reference`. Test nào gọi `write_comparison(plots=True)` cần matplotlib, chỉ chạy được bằng `.venv`.
- [ ] **Step 3:** Viết code.
- [ ] **Step 4:** `.venv/bin/python -m pytest colab/tests -q`. Expected: toàn bộ pass.
- [ ] **Step 5:** Commit: `feat(colab): compare tuning runs against C1 control; device column`.

### Task 5: Notebook chạy được hai nơi

**Files:**
- Modify: `colab/session.py` (thêm `start`), `colab/notebooks/src/01_experiments.py`, `colab/notebooks/src/02_compare.py`, `colab/notebooks/src/03_demo.py`, và các `.ipynb` build lại từ đó.
- Test: `colab/tests/test_session.py`, `colab/tests/test_build_notebooks.py`

**Interfaces:**
- Consumes: `drive_root`, `choose_device`, `check_gpu`, `device_label`, `restore_data(manifest_dst=)`.
- Produces: `session.start(repo: Path, on_colab: bool) -> dict`, với các khóa `on_colab`, `repo`, `runs`, `data`, `manifest`, `device`.
  - Cả hai môi trường: `os.chdir(repo)`, rồi in đường dẫn `runs` và `device_label`.
  - Colab: in `VERSION.txt`, chạy `pip install datasets streamlit pytest` qua `subprocess` bằng `sys.executable`, kiểm `code_hash` (giữ đúng câu lỗi cũ), rồi gọi `check_gpu`. Manifest là `repo/outputs/MANIFEST_NAME`.
  - Máy: `manifest = repo/"data"/MANIFEST_NAME`.
- Cell 0 mới, dùng chung cho 01, 02 và 03:

```python
import sys
from pathlib import Path
ON_COLAB = "google.colab" in sys.modules
if ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    shutil.copytree("/content/drive/MyDrive/DL/vnfood_colab/code", "/content/vnfood", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("data", "models"))
    REPO = Path("/content/vnfood")
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "colab" / "session.py").is_file())
sys.path.insert(0, str(REPO))
from colab import session
env = session.start(REPO, ON_COLAB)
RUNS, MANIFEST, DEVICE = env["runs"], env["manifest"], env["device"]
```

- Cell 1 của 01 và 02: `session.restore_data(REPO, env["data"], manifest_dst=MANIFEST)`.
- Notebook 01:
  - Bảng trạng thái có thêm C1.
  - Thêm cell **Chạy thử tốc độ** trước E1: `train_arch.benchmark(<base>, ...)` rồi in kết quả. Nếu chưa có pin thì dùng `EXPERIMENTS["E2"]`.
  - Thêm cell **C1 — đối chứng trên M4** ngay trước T1.
  - Markdown ghi rõ: E1–E4 đã chạy trên T4, ở máy bắt đầu từ C1.
- Notebook 02: `calibrate_run(..., json.load(open(MANIFEST)), REPO, DEVICE)`.
- Notebook 03: ở máy, chạy `streamlit` bằng `sys.executable -m streamlit` và in `http://localhost:8501`. Chỉ tải cloudflared khi đang ở Colab.
- Notebook 00 giữ cho Colab. Thêm một dòng markdown: "Ở máy không cần notebook này".

- [ ] **Step 1:** Viết các test fail:
  - `test_start_local_paths`: dùng `VNFOOD_DRIVE_ROOT` qua `os.environ` (lưu giá trị cũ và trả lại sau test). Kiểm `env["runs"] == root/"runs"` và `env["manifest"] == repo/"data"/MANIFEST_NAME`. Nếu không có torch thì `device` được phép là `"cpu"`. Muốn vậy thì `start` phải import torch một cách lười, và rơi về `cpu` khi không có torch. Test phải trả lại `cwd` cũ sau khi chạy, vì `start` có `chdir`.
  - `test_notebook_shell_calls_are_safe`: duyệt `colab/notebooks/src/0[123]_*.py`. Mọi dòng bắt đầu bằng `!` mà có `colab.train_arch` thì phải có `"{sys.executable}"`. Mọi `{RUNS}`, `{MANIFEST}` trong dòng `!` phải nằm trong `"..."`. Và không còn `!python `.
- [ ] **Step 2:** `python3 -I .superpowers/sdd/2026-11-23-colab-plan/minirun.py colab.tests.test_session`. Expected: FAIL ở `start` và ở test notebook.
- [ ] **Step 3:** Viết `start`, sửa source các notebook, rồi `python3 colab/tools/build_notebooks.py`.
- [ ] **Step 4:** `.venv/bin/python -m pytest colab/tests -q`. Expected: toàn bộ pass. Thêm `bash colab/tests/test_sync.sh`. Expected: `5 passed`.
- [ ] **Step 5:** Sửa `colab/README.md`: thêm mục "Chạy trên máy (M4)" gồm setup, mở notebook bằng kernel `vnfood (.venv)`, C1 và lý do có C1, lưu ý bật Drive for Desktop. Sau đó `bash colab/sync_to_drive.sh` để bản trên Drive cho Colab cũng mới. Commit: `feat(colab): notebooks run on Colab or local M4`.

### Task 6: Người dùng chạy C1 trên máy

- [ ] **Step 1:** Nhắn người dùng: mở `colab/notebooks/01_experiments.ipynb` trong VS Code hoặc Jupyter, kernel `vnfood (.venv)`, chạy cell 0, cell 1 (tải và giải nén 6 GB, chỉ một lần), rồi cell **Chạy thử tốc độ**. Họ gửi lại số `est_run_hours`.
- [ ] **Step 2:** Nếu `est_run_hours` quá lớn (hơn 4 giờ một run), thì báo người dùng, rồi thử `VNFOOD_WORKERS` khác và ghi `Ruling:`. Nếu ổn, người dùng chạy cell C1.
- [ ] **Step 3:** Khi người dùng báo xong:
  - `bash colab/pull_results.sh C1_resnet50_m4`.
  - Kiểm `metrics.json`: `device` bắt đầu bằng `mps`, test accuracy cùng tầm với E2 (88,8%). Hiệu số E2 − C1 được ghi vào ledger.
  - Kiểm `history.csv`: đủ epoch và không có NaN.
- [ ] **Step 4:** Nếu ổn: commit `colab/results/` cùng các notebook đã chạy, rồi `git push origin main`. Các run T* sau đó đi theo vòng: người dùng chạy → pull → kiểm → push.
