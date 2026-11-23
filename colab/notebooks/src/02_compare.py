# %% [markdown]
# # 02 — So sánh và đánh giá mô hình được chọn
# Gom mọi run đã xong thành bảng + biểu đồ cho slide, chọn mô hình cuối theo **validation accuracy**,
# rồi hiệu chỉnh độ tin cậy (temperature scaling trên validation) và xem 15 ảnh sai tự tin nhất.

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

# %%
if not session.restore_data():
    raise SystemExit("Chưa có data trên Drive — chạy notebook 00 trước.")
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image as Show, display
from colab import compare
rows = compare.collect(Path(RUNS))
out = Path(RUNS) / "_compare"
compare.write_comparison(rows, out)
pd.DataFrame(rows).drop(columns=["run_dir"]).round(4)

# %%
for name in ("comparison_arch.png", "accuracy_vs_params.png", "learning_curves_all.png", "tuning.png"):
    if (out / name).is_file():
        display(Show(str(out / name)))

# %% [markdown]
# ## Mô hình cuối: validation accuracy cao nhất trong mọi run

# %%
from colab import analyze
best = compare.pick_best(rows)
print("Mô hình cuối:", best["run_id"], "| val acc", round(best["best_val_accuracy"], 4),
      "| test acc", round(best["test_accuracy"], 4))
report = analyze.calibrate_run(Path(best["run_dir"]), json.load(open("outputs/clean_split_manifest.json")),
                               Path("/content/vnfood"), "cuda")
print(f"Temperature={report['temperature']:.4f}; test ECE {report['test_before']['ece']:.4f} -> "
      f"{report['test_after']['ece']:.4f}; NLL {report['test_before']['nll']:.4f} -> {report['test_after']['nll']:.4f}")
display(Show(str(analyze.misclassified_gallery(Path(best["run_dir"]), Path("/content/vnfood")))))
(out / "final.json").write_text(json.dumps({"run_id": best["run_id"], "arch": best["arch"],
                                            "reason": "best validation accuracy"}, indent=2))
