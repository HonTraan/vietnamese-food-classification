# %% [markdown]
# # 00 — Chuẩn bị dữ liệu (chạy một lần)
# Tải 30 lớp từ Hugging Face, loại ảnh trùng, tạo split 70/15/15, kiểm tra split có khớp với split cũ trong repo không,
# rồi lưu data đã làm sạch lên Drive. Các notebook sau chỉ cần giải nén lại, không tải lại.

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
# ## 1. Kiểm tra code (pytest)
# Phải thấy toàn bộ test **passed**. Nếu có test fail, copy toàn bộ output gửi Claude trước khi chạy tiếp.

# %%
!python -m pytest -q colab/tests

# %% [markdown]
# ## 2. Tải và làm sạch dữ liệu
# Mất khoảng 15–30 phút ở lần đầu. Nếu Drive đã có `data/raw.zip` thì cell chỉ giải nén lại.

# %%
import json
from pathlib import Path
repo = Path("/content/vnfood")
if session.restore_data():
    print("Đã khôi phục data từ Drive — bỏ qua bước tải.")
else:
    shutil.copy(repo / "outputs/clean_split_manifest.json", repo / "outputs/clean_split_manifest.repo.json")
    !python src/dataset.py
    !python src/data_cleaning.py --apply
    original = json.loads((repo / "outputs/clean_split_manifest.repo.json").read_text())
    fresh = json.loads((repo / "outputs/clean_split_manifest.json").read_text())
    result = session.compare_manifests(original, fresh)
    print(json.dumps(result, indent=2))
    print("SPLIT KHỚP REPO: OK" if result["same"] else "SPLIT KHÁC REPO — gửi output này cho Claude")
    session.save_data()
    print("Đã lưu data/raw.zip và manifest lên Drive.")

# %% [markdown]
# ## 3. EDA: phân bố lớp và ảnh mẫu

# %%
!python src/eda.py
eda_dir = Path(RUNS) / "_eda"
eda_dir.mkdir(parents=True, exist_ok=True)
for name in ("class_distribution.png", "sample_images.png"):
    shutil.copy(repo / "outputs/figures" / name, eda_dir / name)
shutil.copy(repo / "outputs/eda_summary.json", eda_dir / "eda_summary.json")
from IPython.display import Image as Show, display
display(Show(str(eda_dir / "class_distribution.png")))
