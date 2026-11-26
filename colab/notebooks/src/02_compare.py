# %% [markdown]
# # 02 — So sánh và đánh giá mô hình được chọn
# Gom mọi run đã xong thành bảng + biểu đồ cho slide, chọn mô hình cuối theo **validation accuracy**,
# rồi hiệu chỉnh độ tin cậy (temperature scaling trên validation) và xem 15 ảnh sai tự tin nhất.

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
    REPO = REPO
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "colab" / "session.py").is_file())
sys.path.insert(0, str(REPO))
from colab import session
env = session.start(REPO, ON_COLAB)
RUNS, MANIFEST, DEVICE = env["runs"], env["manifest"], env["device"]

# %%
if not session.restore_data(REPO, env["data"], manifest_dst=MANIFEST):
    raise SystemExit("Chưa có data trên Drive — chạy notebook 00 trước.")
import json
from pathlib import Path
import pandas as pd
from IPython.display import Image as Show, display
from colab import compare
rows = compare.collect(RUNS)
out = RUNS / "_compare"
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
report = analyze.calibrate_run(Path(best["run_dir"]), json.loads(MANIFEST.read_text()),
                               REPO, DEVICE)
print(f"Temperature={report['temperature']:.4f}; test ECE {report['test_before']['ece']:.4f} -> "
      f"{report['test_after']['ece']:.4f}; NLL {report['test_before']['nll']:.4f} -> {report['test_after']['nll']:.4f}")
display(Show(str(analyze.misclassified_gallery(Path(best["run_dir"]), REPO))))
(out / "final.json").write_text(json.dumps({"run_id": best["run_id"], "arch": best["arch"],
                                            "reason": "best validation accuracy"}, indent=2))
