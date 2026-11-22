# Phân tích 15 ảnh dự đoán sai

Các nhận xét dựa trên quan sát ảnh trong `figures/misclassified_top15.png`.

| STT | Nhãn đúng | Nhãn dự đoán | Confidence | Nguyên nhân có thể |
|---:|---|---|---:|---|
| 1 | Banh can | Banh khot | 100.0% | Hai món có hình dáng bánh rán tương tự; cần kiểm tra nhãn gốc. |
| 2 | Banh tet | Banh chung | 99.8% | Cùng là gạo nếp/bọc lá; góc crop làm hai món khó phân biệt. |
| 3 | Bun bo Hue | Bun rieu | 99.8% | Nước dùng đỏ, sợi bún và rau tạo đặc trưng chung. |
| 4 | Canh chua | Ca kho to | 99.8% | Ảnh cá trong nồi và gia vị che đặc trưng canh chua. |
| 5 | Banh canh | Chao long | 99.7% | Tô nước có thịt/màu tương tự; crop nên tập trung phần sợi. |
| 6 | Banh duc | Nem chua | 99.5% | Bánh/rau ít đặc trưng; cần kiểm tra nhãn gốc và ánh sáng. |
| 7 | Banh cuon | Goi cuon | 99.4% | Cùng có bánh, rau và nước chấm; bố cục gần giống nhau. |
| 8 | Banh tet | Banh chung | 99.2% | Cùng là gạo nếp/bọc lá; góc crop làm hai món khó phân biệt. |
| 9 | Chao long | Bun bo Hue | 99.2% | Mô hình nhầm đặc trưng chung của món bún nước. |
| 10 | Bun bo Hue | Bun rieu | 99.1% | Nước dùng đỏ, sợi bún và rau tạo đặc trưng chung. |
| 11 | Hu tieu | Bun rieu | 99.1% | Màu nước, thịt và sợi bún làm hai tô nước dễ nhầm. |
| 12 | Banh xeo | Banh khot | 98.9% | Hai món là bột chiên, rau và topping; ảnh thiếu chi tiết đặc trưng. |
| 13 | Chao long | Banh canh | 98.6% | Hai nhãn có đặc trưng hình ảnh gần nhau; cần xem lại nhãn và crop. |
| 14 | Pho | Bun rieu | 98.6% | Góc chụp và nước dùng/nhiều topping làm giảm khác biệt. |
| 15 | Banh mi | Goi cuon | 98.6% | Bố cục ảnh tập trung đĩa cuốn/rau thay vì ổ bánh mì. |
