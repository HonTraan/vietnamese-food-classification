# %% [markdown]
# # 03 — Demo
# Chạy app phân loại bằng Streamlit và mở link public qua cloudflared (link `*.trycloudflare.com`).
# Cần chạy notebook 02 trước để có `runs/_compare/final.json`.

# %% [markdown]
# ## 0. Chuẩn bị phiên Colab
# Chạy cell này **mỗi lần mở notebook** (kể cả sau khi Colab bị ngắt). Nhớ bật GPU: *Runtime → Change runtime type → T4 GPU*.
# So dòng `version=...` in ra với version Claude báo sau khi sync; nếu khác thì đợi Drive đồng bộ xong rồi chạy lại cell.

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
print("GPU:", session.check_gpu())

# %%
import json, re, subprocess, time
from pathlib import Path
final = json.loads((Path(RUNS) / "_compare/final.json").read_text())
print("Model demo:", final)
Path("models").mkdir(exist_ok=True)
shutil.copy(Path(RUNS) / final["run_id"] / "best.pt", "models/final.pt")
if final["arch"] == "efficientnet_b0":
    shutil.copy("models/final.pt", "models/best_model.pt")   # app.py gốc của repo dùng được
    app = "app.py"
else:
    app = "colab/demo_app.py"
print("App:", app)

# %%
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared
subprocess.Popen(f"VNFOOD_MODEL=models/final.pt streamlit run {app} --server.port 8501 --server.headless true "
                 "> /content/streamlit.log 2>&1", shell=True)
tunnel = subprocess.Popen("/content/cloudflared tunnel --url http://localhost:8501 > /content/tunnel.log 2>&1", shell=True)
for _ in range(30):
    time.sleep(2)
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
    if match:
        print("Mở demo tại:", match.group(0))
        break
else:
    print(open("/content/tunnel.log").read()[-2000:])
