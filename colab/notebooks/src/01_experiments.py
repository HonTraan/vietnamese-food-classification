# %% [markdown]
# # 01 — Thử nghiệm
# Mỗi cell là **một run**. Kết quả tự lưu vào `Drive/DL/vnfood_colab/runs/<run_id>/`.
# - Colab bị ngắt giữa chừng: mở lại notebook, chạy cell 0, cell 1 rồi chạy lại **đúng cell đang dở** → tự chạy tiếp từ epoch đã lưu.
# - Run đã xong sẽ in `SKIP ... (DONE)`, không train lại.
# - Thứ tự ưu tiên khi ít quota: E1 → E4 → E2 → E3 → T1 → T3 → T2 → T4a → T4b → T5 → E5.

# %% [markdown]
# ## 0. Chuẩn bị phiên Colab
# Chạy cell này **mỗi lần mở notebook** (kể cả sau khi Colab bị ngắt). Nhớ bật GPU: *Runtime → Change runtime type → T4 GPU*.
# Cell tự kiểm tra code trên Colab khớp với bản Claude vừa sync (hash). Nếu Claude sync lại trong lúc notebook đang mở: *Runtime → Restart session* rồi chạy lại từ cell này.

# %%
from google.colab import drive
drive.mount("/content/drive")
import shutil, sys
DRIVE = "/content/drive/MyDrive/DL/vnfood_colab"
RUNS = f"{DRIVE}/runs"
print(open(f"{DRIVE}/code/VERSION.txt").read())
shutil.copytree(f"{DRIVE}/code", "/content/vnfood", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("data", "models"))
%cd /content/vnfood
!pip -q install datasets streamlit pytest
sys.path.insert(0, "/content/vnfood")
from colab import session
expected = session.read_version(f"{DRIVE}/code/VERSION.txt")["content_hash"]
actual = session.code_hash("/content/vnfood")
assert actual == expected, (f"Code chưa đồng bộ xong (hash {actual} != {expected}). Đợi 1-2 phút rồi chạy lại cell này; "
                            "nếu vẫn lệch: Runtime > Restart session rồi chạy lại.")
print("Code khớp Drive, hash:", actual)
print("GPU:", session.check_gpu())

# %% [markdown]
# ## 1. Khôi phục data và xem trạng thái các run

# %%
if not session.restore_data():
    raise SystemExit("Chưa có data trên Drive — chạy notebook 00 trước.")
from pathlib import Path
from colab.experiments import EXPERIMENTS, PRIORITY, run_state, tuning
from colab import compare
for key in PRIORITY:
    run_id = EXPERIMENTS[key].run_id if key in EXPERIMENTS else f"{key} (sau khi chốt base)"
    state = run_state(Path(RUNS) / EXPERIMENTS[key].run_id) if key in EXPERIMENTS else "-"
    print(f"{key:4s} {run_id:32s} {state}")

# %% [markdown]
# ## 2. So sánh kiến trúc (cùng split sạch, cùng công thức train)
# ### E1 — CNN 4 block tự xây (train từ đầu)

# %%
!python -m colab.train_arch --exp E1 --runs-dir {RUNS}

# %% [markdown]
# ### E4 — EfficientNet-B0 fine-tune (ứng viên chính)

# %%
!python -m colab.train_arch --exp E4 --runs-dir {RUNS}

# %% [markdown]
# ### E2 — ResNet-50 fine-tune

# %%
!python -m colab.train_arch --exp E2 --runs-dir {RUNS}

# %% [markdown]
# ### E3 — MobileNetV3-Large fine-tune

# %%
!python -m colab.train_arch --exp E3 --runs-dir {RUNS}

# %% [markdown]
# ## 3. Chốt mô hình base cho tinh chỉnh
# Chọn mô hình có **validation accuracy** cao nhất trong E1–E4 (không nhìn test). Chốt một lần, lưu `runs/tuning_base.json`.

# %%
base = compare.pin_tuning_base(Path(RUNS), compare.collect(Path(RUNS)))
print("Base cho tinh chỉnh:", base.run_id)
for key, cfg in tuning(base).items():
    print(key, cfg.run_id, run_state(Path(RUNS) / cfg.run_id))

# %% [markdown]
# ## 4. Tinh chỉnh (mỗi run chỉ đổi đúng một yếu tố so với base)
# ### T1 — Freeze backbone, chỉ train classifier: *fine-tune toàn bộ có đáng không?*

# %%
!python -m colab.train_arch --tune T1 --runs-dir {RUNS}

# %% [markdown]
# ### T3 — Không augmentation: *augmentation có giảm overfit không?*

# %%
!python -m colab.train_arch --tune T3 --runs-dir {RUNS}

# %% [markdown]
# ### T2 — Không dùng ImageNet pretrained: *transfer learning giúp bao nhiêu?*

# %%
!python -m colab.train_arch --tune T2 --runs-dir {RUNS}

# %% [markdown]
# ### T4a / T4b — Learning rate 1e-3 và 1e-4 (base 3e-4)

# %%
!python -m colab.train_arch --tune T4a --runs-dir {RUNS}

# %%
!python -m colab.train_arch --tune T4b --runs-dir {RUNS}

# %% [markdown]
# ### T5 — Label smoothing 0.1

# %%
!python -m colab.train_arch --tune T5 --runs-dir {RUNS}

# %% [markdown]
# ## 5. Tùy chọn (nếu còn quota)
# ### E5 — ConvNeXt-Tiny fine-tune

# %%
!python -m colab.train_arch --exp E5 --runs-dir {RUNS}

# %% [markdown]
# ### Multi-seed cho base (đo độ ổn định: seed 123 và 2024)

# %%
!python -m colab.train_arch --seed-run S123 --runs-dir {RUNS}
!python -m colab.train_arch --seed-run S2024 --runs-dir {RUNS}
