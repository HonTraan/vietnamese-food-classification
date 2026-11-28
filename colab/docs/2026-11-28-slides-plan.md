# Kế hoạch thực hiện slide thuyết trình

> **Cho agent thực hiện:** BẮT BUỘC dùng sub-skill superpowers:executing-plans (người dùng đã chọn cách làm inline). Các bước dùng checkbox (`- [ ]`) để theo dõi.

**Mục tiêu:** Script `colab/tools/slide_assets.py` sinh mọi bảng và hình từ code và kết quả thật. Sau đó dựng một deck Slides khoảng 20 phút trên claude.ai, có speaker notes chia thành 3 phần.

**Kiến trúc:**
- **Script.** Dựng model bằng `colab.models_zoo.build(..., pretrained=False)` để không phải tải trọng số. Đo kích thước tensor vào/ra bằng forward hook với ảnh `1×3×224×224`. Vẽ hình từ `colab/results/<run>/history.csv` và `metrics.json`. Ghi kết quả vào `colab/results/_slides/`.
- **Deck.** Một Artifact tạo từ loại Slides (`type_url` https://claude.ai/artifact/8jTsAFQMFDb2oA8MsPJ2eL). Ảnh lấy từ `colab/results/`.

**Tech stack:** PyTorch 2.15 / torchvision 0.29, matplotlib, seaborn (đã có trong `.venv`), pytest.

**Spec:** `colab/docs/2026-11-28-slides-design.md`

## Ràng buộc chung

- Không sửa `src/`, `app.py`, `README.md`, `outputs/`, `models/best_model.pt`. Code mới chỉ nằm trong `colab/`.
- Không chạy notebook, không train. Chạy `slide_assets.py` thì được, vì người dùng đã duyệt thiết kế cho phép chạy script này (CPU, vài giây).
- Mọi con số trên slide đều lấy từ `colab/results/**` hoặc `colab/results/_slides/**`. Mỗi bảng ghi nguồn ở chân slide hoặc trong notes.
- Slide viết bằng tiếng Việt, thuật ngữ giữ tiếng Anh. Có 17 slide chính, tổng khoảng 21 phút, và 7 slide phụ lục A1–A7.
- Speaker notes ghi "Phần 1" cho slide 1–6, "Phần 2" cho slide 7–11, "Phần 3" cho slide 12–17. Không ghi tên người nói.
- Mô hình cuối là `E2_resnet50` (`colab/results/_compare/final.json`). Chọn theo val, test chỉ để báo cáo.
- Design system "Design System" đang trống, nên dùng giao diện mặc định của loại Slides.
- Push thẳng lên `main`. Không commit `graphify-out/`.

## Điểm cần review

1. **Lưới biểu đồ không được thiếu hay sai thứ tự run khi có run chưa xong.**
   - Run không có `DONE` (ví dụ T5 đang dở) thì bỏ qua và in cảnh báo, không crash.
   - Thứ tự run theo `compare._order`.
   - Test: `test_grid_skips_unfinished_run`.
2. **CNN có shape tensor khác ResNet-50.** Block 4 không có MaxPool, nên bảng tầng phải ghi đúng `256×28×28`, không ghi `256×14×14`. Test: `test_cnn_block_shapes`.
3. **Đếm tầng ResNet-50.** Conv của nhánh downsample không được tính vào con số 50. Test: `test_resnet50_has_50_weight_layers`.
4. **Confusion matrix phải chuẩn hoá theo hàng.** Lớp có ít ảnh test không được trông như lỗi ít hơn. Mỗi hàng cộng lại bằng 1. Test: `test_row_normalized_confusion`.
5. **Hai lần chạy script phải cho cùng kết quả.** Chạy lại không thay đổi số liệu trong CSV, và không ghi gì ra ngoài `_slides/`. Test: `test_main_writes_only_into_out_dir`.

---

### Task 1: Bảng từng tầng và tóm tắt model

**Files:**
- Create: `colab/tools/slide_assets.py`
- Test: `colab/tests/test_slide_assets.py`

**Interfaces:**
- Produces:
  - `layer_table(arch: str, num_classes: int = 30) -> list[dict]`
    - Mỗi dict có các khoá `name, type, in_shape, out_shape, kernel, stride, padding, params, trainable`. Shape là chuỗi dạng `"64×112×112"`, không có chiều batch.
    - Chỉ ghi các module lá, cộng thêm các block cấp cao `layer1..4` của ResNet.
  - `resnet_stage_summary(num_classes: int = 30) -> list[dict]` trả về `stage, blocks, out_shape, params` cho conv1, maxpool, layer1–4, avgpool, fc.
  - `weight_layer_count(arch: str) -> int` đếm Conv2d và Linear trên đường chính. Với ResNet thì bỏ qua module có tên chứa `downsample`.
  - `model_summary(arch: str, num_classes: int = 30) -> dict` trả về `arch, total_params, head_params, frozen_trainable_params, macs, final_feature_shape`.
    - `macs` là tổng MACs của Conv2d và Linear, đo bằng hook.
    - `frozen_trainable_params` lấy từ `build(..., freeze_backbone=True)`. Với `cnn` thì để `None`.

- [ ] **Step 1: Viết test (RED)**

```python
from colab.tools import slide_assets as sa

def test_resnet50_param_counts():
    s = sa.model_summary("resnet50")
    assert s["total_params"] == 23_569_502
    assert s["head_params"] == 61_470
    assert s["frozen_trainable_params"] == 61_470
    assert s["final_feature_shape"] == "2048×7×7"

def test_resnet50_has_50_weight_layers():
    assert sa.weight_layer_count("resnet50") == 50

def test_resnet_stages():
    rows = {r["stage"]: r for r in sa.resnet_stage_summary()}
    assert [rows[f"layer{i}"]["blocks"] for i in range(1, 5)] == [3, 4, 6, 3]
    assert rows["conv1"]["out_shape"] == "64×112×112"
    assert rows["layer1"]["out_shape"] == "256×56×56"
    assert rows["layer4"]["out_shape"] == "2048×7×7"
    assert rows["fc"]["out_shape"] == "30"

def test_cnn_block_shapes():
    convs = [r for r in sa.layer_table("cnn") if r["type"] == "MaxPool2d" or r["type"] == "AdaptiveAvgPool2d"]
    assert [r["out_shape"] for r in convs] == ["32×112×112", "64×56×56", "128×28×28", "256×1×1"]
    assert sa.layer_table("cnn")[-1]["out_shape"] == "30"

def test_layer_table_params_sum_to_total():
    for arch in ("cnn", "mobilenet_v3_large", "efficientnet_b0"):
        leaf = [r for r in sa.layer_table(arch) if not r["name"].startswith("layer")]
        assert sum(r["params"] for r in leaf) == sa.model_summary(arch)["total_params"]
```

- [ ] **Step 2:** Chạy `.venv/bin/python -m pytest colab/tests/test_slide_assets.py -q`.
  Expected: FAIL với `ModuleNotFoundError: colab.tools.slide_assets`.
- [ ] **Step 3:** Viết 4 hàm của Task 1 trong `colab/tools/slide_assets.py`.
  - Chạy forward hook trên `torch.zeros(1, 3, 224, 224)` ở chế độ `eval()` và `torch.no_grad()`.
  - Định dạng shape bằng `"×".join(map(str, t.shape[1:]))`.
- [ ] **Step 4:** Chạy lại test như Step 2. Expected: 5 passed.
- [ ] **Step 5:** Commit:

```bash
git add colab/tools/slide_assets.py colab/tests/test_slide_assets.py
git commit -m "feat(colab): layer tables and model summary for slides"
```

### Task 2: Hình, bảng siêu tham số và CLI

**Files:**
- Modify: `colab/tools/slide_assets.py`
- Test: `colab/tests/test_slide_assets.py`

**Interfaces:**
- Consumes: `compare.collect(runs_dir)` trả về các dòng có `run_dir` và đã sắp theo `_order`. Các hàm của Task 1.
- Produces:
  - `learning_curves_grid(rows: list[dict], out: Path) -> Path`
    - Lưới 11 hàng × 2 cột (accuracy, loss). Mỗi ô có đường train và đường val, và một đường dọc tại `best_epoch`.
    - Chiều cao hình tỉ lệ với số run.
  - `train_val_gap(rows, out: Path) -> Path` vẽ biểu đồ cột ngang `train_accuracy − val_accuracy` tại epoch có val tốt nhất.
  - `per_class_f1(run_dir: Path, out: Path) -> Path` vẽ cột ngang F1 theo lớp (lấy từ `metrics.json["report"]`), sắp xếp tăng dần.
  - `row_normalized(matrix: list[list[int]]) -> list[list[float]]`.
  - `confusion_grid(rows, out: Path) -> Path` vẽ lưới 2×2 cho E1–E4, chuẩn hoá theo hàng.
  - `hyperparams_table(rows) -> list[dict]` có các cột `run_id, arch, pretrained, freeze_backbone, augment, lr, label_smoothing, epochs, batch_size, patience, seed, optimizer, weight_decay, scheduler, device`.
    - `optimizer = "AdamW"`, `weight_decay = 1e-4`.
    - `scheduler` là `"ReduceLROnPlateau(factor=0.5, patience=1)"` nếu arch là `cnn`, ngược lại là `"CosineAnnealingLR(T_max=epochs)"`.
  - `main(runs_dir: Path, out_dir: Path) -> None` ghi các file sau, và chỉ ghi vào `out_dir`:
    - `layers_<arch>.csv` cho 4 kiến trúc;
    - `resnet50_stages.csv`, `model_summary.csv`, `hyperparams.csv`;
    - 4 file PNG: `learning_curves_grid.png`, `train_val_gap.png`, `per_class_f1_E2_resnet50.png`, `confusion_grid.png`.
  - CLI: `python -m colab.tools.slide_assets [--runs colab/results] [--out colab/results/_slides]`.

- [ ] **Step 1: Viết test (RED).** Tạo run giả bằng helper giống `_run` trong `test_compare.py`, bổ sung `history.csv` và `metrics.json` (có `report`, `confusion_matrix`, `best_epoch`).
  - `test_grid_skips_unfinished_run`: 2 run có `DONE` và 1 run không có. File PNG phải tồn tại, và hàm trả về hoặc ghi log đúng 2 run.
  - `test_row_normalized_confusion`: `row_normalized([[2,2],[0,0]]) == [[0.5,0.5],[0.0,0.0]]`. Hàng toàn 0 không được gây chia cho 0.
  - `test_hyperparams_scheduler`: `cnn` cho ra `ReduceLROnPlateau...`, `resnet50` cho ra `CosineAnnealingLR...`.
  - `test_main_writes_only_into_out_dir`: chạy `main` hai lần trên `tmp_path/runs`.
    - Tập file trong `runs/` không đổi, `out/` có đủ 11 file.
    - `model_summary.csv` của hai lần chạy giống hệt nhau.
- [ ] **Step 2:** Chạy pytest như Task 1. Expected: 4 test mới FAIL với `AttributeError`.
- [ ] **Step 3:** Viết các hàm.
  - Dùng `matplotlib.use("Agg")`, dpi 150.
  - Tiêu đề và nhãn trục bằng tiếng Việt, giống `compare._plot`.
  - Vẽ confusion matrix bằng `seaborn.heatmap`, đặt `vmin=0, vmax=1`.
- [ ] **Step 4:** Chạy test. Expected: 9 passed. Sau đó chạy cả bộ test:
  `.venv/bin/python -m pytest colab/tests -q`. Expected: 67 passed.
- [ ] **Step 5:** Commit:

```bash
git commit -am "feat(colab): slide figures, hyperparams table, CLI"
```

  Nhớ `git add` test file trước nếu chưa track.

### Task 3: Sinh tài liệu thật và kiểm tra lại

- [ ] **Step 1:** Chạy `.venv/bin/python -m colab.tools.slide_assets`.
  Expected: in ra 11 file trong `colab/results/_slides/`, không có cảnh báo bỏ qua run nào.
- [ ] **Step 2:** Đối chiếu số liệu:
  - `model_summary.csv`: hàng resnet50 có `23569502` và `61470`.
  - `hyperparams.csv`: có 11 dòng.
  - Mở 4 file PNG bằng Read, kiểm tra nhãn không chồng chéo và đủ 11 run.
- [ ] **Step 3:** Commit `colab/results/_slides/` và push lên `main`:

```bash
git push origin main
```

### Task 4: Dựng deck Slides

**Interfaces:**
- Consumes: các file trong `colab/results/_slides/`, `colab/results/_compare/*.png`, `colab/results/_eda/*.png`, `colab/results/E2_resnet50/{confusion_matrix,misclassified_top15}.png`, `calibration.json`, `comparison.csv`.

- [ ] **Step 1:** Tạo Artifact bằng `type_url` https://claude.ai/artifact/8jTsAFQMFDb2oA8MsPJ2eL, title "Phân loại món ăn Việt Nam", `auto_open: "after_first_write"`. Làm đúng hướng dẫn mà kết quả tạo Artifact trả về, gồm định dạng file và cách upload ảnh.
- [ ] **Step 2:** Viết slide 1–17 và A1–A7 đúng theo bảng ở Phần 2 của spec.
  - Mỗi slide có notes gồm: phần (1/2/3), lời nói khoảng 60–120 giây, và 2–3 câu hỏi thầy có thể hỏi kèm câu trả lời.
  - Slide 8 vẽ sơ đồ stage ResNet-50 trực tiếp trên slide, số liệu lấy từ `resnet50_stages.csv`.
- [ ] **Step 3:** Kiểm tra số liệu.
  - Lập danh sách mọi con số xuất hiện trên slide.
  - Đối chiếu từng số với file nguồn bằng một lệnh grep hoặc đoạn Python ngắn đọc CSV/JSON.
  - Sửa ngay nếu lệch. Ghi danh sách vào ledger.
- [ ] **Step 4:** Publish. Mở deck để kiểm tra bố cục: không tràn chữ, ảnh đọc được. Báo link cho người dùng.

### Task 5: Hoàn tất

- [ ] **Step 1:** Chạy `graphify update .`. Không commit `graphify-out/`.
- [ ] **Step 2:** Chạy cả bộ test. Expected: 67 passed. Chạy `bash colab/tests/test_sync.sh`. Expected: 5 passed.
- [ ] **Step 3:** Cập nhật `colab/README.md`: thêm một dòng về `slide_assets.py` trong mục "Lệnh trên máy". Commit rồi push lên `main`.
