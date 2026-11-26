# %% [markdown]
# # 01 — Thử nghiệm
# Mỗi cell là **một run**. Kết quả tự lưu vào `Drive/DL/vnfood_colab/runs/<run_id>/`, dù chạy trên Colab hay trên máy.
# - E1–E4 đã chạy trên Colab T4. Trên máy (M4) bắt đầu từ **C1** (đối chứng), rồi T1 → T3 → T2 → T4a → T4b → T5.
# - Bị ngắt giữa chừng: mở lại notebook, chạy cell 0, cell 1 rồi chạy lại **đúng cell đang dở** → tự chạy tiếp từ epoch đã lưu.
# - Run đã xong sẽ in `SKIP ... (DONE)`, không train lại.
# - Thứ tự ưu tiên: E1 → E4 → E2 → E3 → C1 → T1 → T3 → T2 → T4a → T4b → T5 → E5.

# %% [markdown]
# ## 0. Chuẩn bị phiên (Colab hoặc máy M4)
# Chạy cell này **mỗi lần mở notebook** (kể cả sau khi bị ngắt). Cell tự nhận biết đang ở đâu:
# - **Colab:** mount Drive, copy code về `/content/vnfood`, kiểm tra code khớp bản Claude vừa sync (hash) và GPU T4 (*Runtime → Change runtime type → T4 GPU*). Nếu Claude sync lại trong lúc notebook đang mở: *Runtime → Restart session* rồi chạy lại từ cell này.
# - **Máy (M4):** chạy thẳng trên repo, kernel `vnfood (.venv)`, GPU `mps`. Kết quả vẫn ghi vào thư mục Drive (bật Drive for Desktop).

# %%
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

# %% [markdown]
# ## 1. Khôi phục data và xem trạng thái các run

# %%
if not session.restore_data(REPO, env["data"], manifest_dst=MANIFEST):
    raise SystemExit("Chưa có data trên Drive — chạy notebook 00 trước.")
from pathlib import Path
from colab.experiments import EXPERIMENTS, PRIORITY, control, run_state, tuning
from colab import compare
for key in PRIORITY:
    run_id = EXPERIMENTS[key].run_id if key in EXPERIMENTS else f"{key} (sau khi chốt base)"
    state = run_state(RUNS / EXPERIMENTS[key].run_id) if key in EXPERIMENTS else "-"
    print(f"{key:4s} {run_id:32s} {state}")

# %% [markdown]
# ### Chạy thử tốc độ trên máy này
# Train thử 30 batch của mô hình base (không ghi gì vào `runs/`), rồi ước tính thời gian mỗi epoch và mỗi run.

# %%
import json
from colab import train_arch
from colab.experiments import RunConfig
pin = RUNS / "tuning_base.json"
trial = RunConfig(**json.loads(pin.read_text())["config"]) if pin.is_file() else EXPERIMENTS["E2"]
estimate = train_arch.benchmark(trial, json.loads(MANIFEST.read_text()), REPO)
print(f"{trial.run_id} trên {estimate['device']}: {estimate['seconds_per_batch']:.2f} s/batch, "
      f"~{estimate['est_epoch_minutes']:.1f} phút/epoch, ~{estimate['est_run_hours']:.1f} giờ/run (tối đa {trial.epochs} epoch)")

# %% [markdown]
# ## 2. So sánh kiến trúc (cùng split sạch, cùng công thức train)
# ### E1 — CNN 4 block tự xây (train từ đầu)

# %%
!"{sys.executable}" -m colab.train_arch --exp E1 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### E4 — EfficientNet-B0 fine-tune (ứng viên chính)

# %%
!"{sys.executable}" -m colab.train_arch --exp E4 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### E2 — ResNet-50 fine-tune

# %%
!"{sys.executable}" -m colab.train_arch --exp E2 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### E3 — MobileNetV3-Large fine-tune

# %%
!"{sys.executable}" -m colab.train_arch --exp E3 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ## 3. Chốt mô hình base cho tinh chỉnh
# Chọn mô hình có **validation accuracy** cao nhất trong E1–E4 (không nhìn test). Chốt một lần, lưu `runs/tuning_base.json`.

# %%
base = compare.pin_tuning_base(RUNS, compare.collect(RUNS))
print("Base cho tinh chỉnh:", base.run_id)
for key, cfg in {**control(base), **tuning(base)}.items():
    print(key, cfg.run_id, run_state(RUNS / cfg.run_id))

# %% [markdown]
# ## 4. Đối chứng trên máy: C1 = base chạy lại trên M4
# Cấu hình **giống hệt** base, chỉ khác phần cứng (M4 `mps`, fp32 thay vì T4 fp16). Các run T* chạy trên máy được so với C1,
# nên mỗi T* chỉ khác đúng một yếu tố. Hiệu số base − C1 là ảnh hưởng của phần cứng.

# %%
!"{sys.executable}" -m colab.train_arch --control C1 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ## 5. Tinh chỉnh (mỗi run chỉ đổi đúng một yếu tố so với mốc: C1 nếu chạy trên máy, base nếu chạy trên Colab)
# ### T1 — Freeze backbone, chỉ train classifier: *fine-tune toàn bộ có đáng không?*

# %%
!"{sys.executable}" -m colab.train_arch --tune T1 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### T3 — Không augmentation: *augmentation có giảm overfit không?*

# %%
!"{sys.executable}" -m colab.train_arch --tune T3 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### T2 — Không dùng ImageNet pretrained: *transfer learning giúp bao nhiêu?*

# %%
!"{sys.executable}" -m colab.train_arch --tune T2 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### T4a / T4b — Learning rate 1e-3 và 1e-4 (base 3e-4)

# %%
!"{sys.executable}" -m colab.train_arch --tune T4a --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %%
!"{sys.executable}" -m colab.train_arch --tune T4b --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### T5 — Label smoothing 0.1

# %%
!"{sys.executable}" -m colab.train_arch --tune T5 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ## 6. Tùy chọn (nếu còn quota)
# ### E5 — ConvNeXt-Tiny fine-tune

# %%
!"{sys.executable}" -m colab.train_arch --exp E5 --runs-dir "{RUNS}" --manifest "{MANIFEST}"

# %% [markdown]
# ### Multi-seed cho base (đo độ ổn định: seed 123 và 2024)

# %%
!"{sys.executable}" -m colab.train_arch --seed-run S123 --runs-dir "{RUNS}" --manifest "{MANIFEST}"
!"{sys.executable}" -m colab.train_arch --seed-run S2024 --runs-dir "{RUNS}" --manifest "{MANIFEST}"
