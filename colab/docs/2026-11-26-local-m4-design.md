# Thiết kế: Chạy tiếp các thử nghiệm trên máy M4 (bổ sung cho thiết kế Colab)

Ngày: 2026-11-26 · Bổ sung cho [2026-11-23-colab-design.md](2026-11-23-colab-design.md). Mọi điều trong thiết kế đó vẫn giữ nguyên, trừ những chỗ ghi bên dưới.

## 1. Mục đích và lý do

- Đã hết quota GPU Colab bản miễn phí. Phải vài ngày nữa mới xin lại được, mà còn 6 run tinh chỉnh chưa chạy.
- Người dùng muốn chạy phần còn lại trên máy (Apple M4 Pro, RAM 24 GB), **vẫn dùng notebook** như trên Colab.
- Mục tiêu không đổi: có bằng chứng cho buổi thuyết trình. Vì vậy các so sánh phải **công bằng**, mỗi run tinh chỉnh chỉ khác đúng một yếu tố so với mốc so sánh của nó.

**Trạng thái lúc viết** (trên Drive `runs/`):

| Run | Phần cứng | Kết quả |
|---|---|---|
| E1–E4 | T4 trên Colab | DONE. Val accuracy: E2 ResNet-50 89,3%, E4 88,8%, E3 88,2%, E1 61,2% |
| Base tinh chỉnh | | `tuning_base.json`, đã chốt là `E2_resnet50` |
| T1 | | Mới có `config.json`, chưa xong epoch nào nên chưa có checkpoint (trạng thái `new`) |

Còn lại: T1, T3, T2, T4a, T4b, T5, rồi notebook 02 và 03.

## 2. Mục tiêu / Không làm / Điều phải tránh

**Mục tiêu**
- Một bộ notebook duy nhất, chạy được **cả trên Colab lẫn trên máy**. Cell 0 tự nhận biết đang chạy ở đâu (người dùng chọn cách A).
- Trên máy, train bằng GPU Apple (`mps`).
- Có **run đối chứng `C1`**: cấu hình giống hệt base (E2), chạy trên M4. Mọi run T* được so với C1 chứ không so với E2 (người dùng chọn cách 1, để mọi thứ nhất quán).
- Hiệu số E2 (chạy trên T4) và C1 (chạy trên M4) được báo cáo như "ảnh hưởng của phần cứng".
- Chạy thử ngắn để đo tốc độ M4 và ước tính thời gian từng run, trước khi chạy thật.

**Không làm**
- Không chạy lại E1–E4. Chúng cùng chạy trên T4 với fp16, nên phần so sánh kiến trúc vẫn công bằng.
- Không sửa code của Hoàng (`src/`, `app.py`, `README.md`, `outputs/`, `tests/`). Điều này **gồm cả việc không ghi đè `outputs/clean_split_manifest.json` ở máy**. Trên Colab, `restore_data` vẫn chép manifest vào đó như cũ, vì đó là bản copy tạm.
- Không dùng fp16 hay autocast trên `mps`. Train bằng fp32 cho ổn định, vì run C1 đã hấp thụ khác biệt về phần cứng và độ chính xác số học.

**Điều phải tránh**
- So T* với E2 trong biểu đồ tinh chỉnh khi đã có C1.
- So tốc độ suy luận hay thời gian train giữa run chạy trên T4 và run chạy trên M4 mà không ghi rõ phần cứng. Bảng so sánh tốc độ kiến trúc chỉ dùng E*.
- Cùng một run chạy đồng thời ở hai nơi (Colab và máy).
- **Claude tự chạy notebook, train hay đánh giá mà không hỏi.** Người dùng tự chạy notebook. Claude chỉ chuẩn bị code, rồi kéo kết quả về kiểm tra.

## 3. Ràng buộc

- Máy: M4 Pro, RAM 24 GB, còn trống 207 GB. Đã có `uv` và Homebrew Python 3.14. Máy chưa có torch.
- Thư mục Drive trên máy (Drive for Desktop) để file ở dạng online. Đọc `raw.zip` (6 GB) sẽ kích hoạt tải về.
- `.gitignore` của repo đã có `.venv/`, `data/`, `*.pt`, `models/`, `.ipynb_checkpoints/`.
- Kết quả vẫn ghi vào `My Drive/DL/vnfood_colab/runs/`, qua bản Drive for Desktop trên máy. Drive for Desktop phải đang chạy trong lúc train. Mỗi epoch upload một checkpoint khoảng 200 MB.
- Quy trình giữ nguyên: sửa code → người dùng chạy → Claude kiểm tra → ổn thì push lên `main`.

## 4. Cách làm

**Các cách đã cân nhắc**
- **A. Một bộ notebook chạy được hai nơi, `runs/` để trên Drive.** *Người dùng chọn.* Mọi run nằm một chỗ, có quota Colab lại thì chạy tiếp được ngay. Đổi lại, cell 0 phức tạp hơn một chút và phải bật Drive for Desktop.
- B. Bộ notebook `local_*` riêng, `runs/` để trên máy. Bị loại vì phải giữ hai bộ notebook gần giống nhau, và không chuyển qua lại Colab giữa chừng một run được.
- C. Chạy bằng script CLI, không dùng notebook. Bị loại vì người dùng muốn giữ notebook.

**Những phần đã quyết**
- **Nhận biết môi trường**, ở `colab/session.py`:
  - Trên Colab vẫn dùng `/content/drive/MyDrive/DL/vnfood_colab`.
  - Trên máy dùng `VNFOOD_DRIVE_ROOT` nếu có, nếu không thì tìm `~/Library/CloudStorage/GoogleDrive-*/My Drive/DL/vnfood_colab`, giống cách `sync_to_drive.sh` đang làm.
  - Ở máy, cell 0 bỏ `drive.mount`, `copytree`, `!pip` và phần kiểm tra hash, vì nó chạy thẳng trên repo này. Cell 0 in ra đường dẫn `runs/` và tên thiết bị.
- **Thiết bị**:
  - Thứ tự chọn là `cuda`, rồi `mps`, rồi `cpu`.
  - Chỉ bật AMP (fp16) khi có `cuda`.
  - `config.json` và `metrics.json` ghi tên thiết bị (ví dụ `Tesla T4` hoặc `mps (Apple M4 Pro)`).
  - `check_gpu` chấp nhận cả `mps`.
- **Data ở máy**:
  - Giải nén `raw.zip` từ Drive vào `<repo>/data/raw`, chỉ làm một lần.
  - Manifest đặt ở `<repo>/data/clean_split_manifest.json` (thư mục đã được gitignore), và notebook truyền `--manifest` trỏ tới đó.
- **Run đối chứng**:
  - Tên là `C1_<arch>_m4`, tạo từ base đã chốt, chỉ đổi `run_id`.
  - Chạy bằng `--control C1`, và chạy trước các T*.
  - Biểu đồ tinh chỉnh dùng C1 làm mốc nếu C1 đã xong, nếu chưa thì dùng base.
  - Bảng so sánh có thêm cột thiết bị.
- **Môi trường ở máy**:
  - Script `colab/setup_local.sh` dùng `uv` tạo `.venv` với Python 3.12, rồi cài `requirements.txt` và `ipykernel`.
  - Người dùng mở notebook trong VS Code hoặc Jupyter và chọn kernel `.venv`.
- **Chạy thử**: notebook 01 có một cell chạy vài batch của base, ghi vào thư mục tạm chứ không ghi vào `runs/`. Cell in ra số giây mỗi batch và ước tính thời gian cho mỗi run.
- **Notebook 03 ở máy**: chạy Streamlit trên `localhost`, không cần cloudflared.
- **Thứ tự ở máy**: C1 → T1 → T3 → T2 → T4a → T4b → T5. Tùy chọn thêm E5 và các run seed S*. Ghi chú: S* và E5 chạy trên M4 thì nên so với C1.

**Để builder quyết**
- Số `num_workers` cho `mps`: chọn theo kết quả chạy thử, khởi đầu khoảng 4–6.
- Cách hiện cột thiết bị và biểu đồ "ảnh hưởng phần cứng" (E2 so với C1).
- Mô hình cuối vẫn là run có val accuracy cao nhất trong mọi run, kể cả C1 và T*. Nếu C1 thắng thì đó vẫn là một mô hình hợp lệ, chọn theo validation.

## 5. Kiểm thử

- Các test có sẵn và test mới viết theo TDD.
- Trên máy, sau khi cài môi trường, `pytest colab/tests` chạy được thật. Claude hỏi người dùng trước khi chạy pytest, vì đây là chạy torch trên máy họ.
- Notebook sau khi sửa vẫn phải chạy được trên Colab: mỗi nhánh của cell 0 được tách thành hàm trong `session.py` để test được.
