# %% [markdown]
# # 03 — Demo
# Chạy app phân loại bằng Streamlit. Trên Colab mở link public qua cloudflared (`*.trycloudflare.com`); trên máy mở `http://localhost:8501`.
# Cần chạy notebook 02 trước để có `runs/_compare/final.json`.

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

# %%
import json, os, re, shutil, subprocess, time
final = json.loads((RUNS / "_compare/final.json").read_text())
print("Model demo:", final)
Path("models").mkdir(exist_ok=True)
shutil.copy(RUNS / final["run_id"] / "best.pt", "models/final.pt")
if ON_COLAB and final["arch"] == "efficientnet_b0":
    shutil.copy("models/final.pt", "models/best_model.pt")   # app.py gốc của repo dùng được
    app = "app.py"
else:
    app = "colab/demo_app.py"   # ở máy không ghi đè models/best_model.pt
print("App:", app)
log_dir = Path("/content") if ON_COLAB else REPO / "models"
streamlit = subprocess.Popen([sys.executable, "-m", "streamlit", "run", app, "--server.port", "8501",
                              "--server.headless", "true"], env={**os.environ, "VNFOOD_MODEL": "models/final.pt"},
                             stdout=open(log_dir / "streamlit.log", "w"), stderr=subprocess.STDOUT)
if not ON_COLAB:
    print("Mở demo tại: http://localhost:8501  (dừng: streamlit.terminate())")

# %% [markdown]
# ### Link public (chỉ trên Colab)

# %%
if ON_COLAB:
    subprocess.run("wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/"
                   "cloudflared-linux-amd64 -O /content/cloudflared && chmod +x /content/cloudflared",
                   shell=True, check=True)
    tunnel = subprocess.Popen("/content/cloudflared tunnel --url http://localhost:8501 > /content/tunnel.log 2>&1",
                              shell=True)
    for _ in range(30):
        time.sleep(2)
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/tunnel.log").read())
        if match:
            print("Mở demo tại:", match.group(0))
            break
    else:
        print(open("/content/tunnel.log").read()[-2000:])
