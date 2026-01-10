# 📚 HƯỚNG DẪN SỬ DỤNG HỆ THỐNG

## 🎯 3 CÁCH SỬ DỤNG CHÍNH

### 1️⃣ **TRAIN MODEL** (Chỉ làm 1 lần hoặc khi có cập nhật dữ liệu)

#### Train tất cả models:
```bash
python3 train_all_models.py
```

#### Train từng model riêng:
```bash
# Train model cho phân tích lớp học
python3 train_class_model.py

# Train model cho phân tích cá nhân
python3 train_individual_model.py

# Train model CLO Predictor (dự đoán điểm CLO)
python3 train_clo_predictor.py
```

**⏱️ Thời gian:** Có thể mất 5-15 phút tùy vào dữ liệu và cấu hình máy

**💾 Kết quả:** Models được lưu tại:
- `trained_models/class_model/class_model.pkl`
- `trained_models/individual_model/individual_model.pkl`
- `trained_models/clo_predictor/clo_predictor.pkl`

---

### 2️⃣ **CHẠY HỆ THỐNG** (Sử dụng model đã train - KHÔNG train lại)

#### Chế độ tương tác với file Excel/CSV:
```bash
python3 run_interactive_with_file.py
```

**Tính năng:**
- Phân tích lớp học: Nhập file Excel/CSV có danh sách sinh viên và điểm CLO
- Phân tích cá nhân: Dự đoán điểm CLO và phân tích chi tiết cho 1 sinh viên

#### Chế độ tương tác đơn giản:
```bash
python3 run.py
```

#### Chế độ nâng cao (Enhanced):
```bash
python3 main.py
```

**⚡ Tốc độ:** Nhanh (chỉ load model, không train lại)

---

### 3️⃣ **CHẠY THỬ/TEST** (Kiểm tra hệ thống)

#### Test nhanh với dữ liệu mẫu:
```bash
python3 usage_example.py
```

#### Test từng component:
```python
# Test trong Python
from model_loader import ClassAnalyzer, IndividualAnalyzer

# Test phân tích lớp
analyzer = ClassAnalyzer()
result = analyzer.analyze_v2(
    subject_id="INF1383",
    lecturer_name="GV001",
    students_data=[
        {"mssv": "SV001", "ho_ten": "Nguyễn Văn A", "diem_clo": 5.5},
        {"mssv": "SV002", "ho_ten": "Trần Văn B", "diem_clo": 4.8}
    ]
)

# Test phân tích cá nhân
individual = IndividualAnalyzer()
result = individual.analyze(
    subject_id="INF1383",
    lecturer_name="GV001",
    student_id="SV001"
)
```

---

## 📋 QUY TRÌNH SỬ DỤNG

### Lần đầu tiên:
1. **Train model:**
   ```bash
   python3 train_all_models.py
   ```
2. **Chạy hệ thống:**
   ```bash
   python3 run_interactive_with_file.py
   ```

### Các lần sau (đã có model):
- **Chỉ cần chạy trực tiếp:**
   ```bash
   python3 run_interactive_with_file.py
   ```
- Hệ thống sẽ tự động load model đã train sẵn, **KHÔNG train lại**

### Khi có cập nhật dữ liệu:
1. **Train lại model:**
   ```bash
   python3 train_all_models.py
   ```
2. **Chạy lại hệ thống:**
   ```bash
   python3 run_interactive_with_file.py
   ```

---

## 🔍 KIỂM TRA MODEL ĐÃ TRAIN CHƯA

### Kiểm tra file model:
```bash
# Kiểm tra các file model
ls -la trained_models/class_model/
ls -la trained_models/individual_model/
ls -la trained_models/clo_predictor/
```

### Nếu thiếu model:
- Hệ thống sẽ báo lỗi và hướng dẫn train
- Hoặc tự động train mới (nếu không tìm thấy file)

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Train model trước khi sử dụng lần đầu**
2. **Không cần train lại mỗi lần chạy** - hệ thống tự động load
3. **Train lại khi:**
   - Có cập nhật dữ liệu (như thay PPGD → PPGDfull)
   - Thay đổi tham số training
   - Model bị lỗi hoặc cần cải thiện

---

## 🚀 TÓM TẮT NHANH

| Mục đích | Lệnh |
|----------|------|
| **Train model** | `python3 train_all_models.py` |
| **Chạy hệ thống** | `python3 run_interactive_with_file.py` |
| **Test/Thử** | `python3 usage_example.py` |

---

## 💡 TIPS

- **Train vào buổi tối hoặc khi không dùng máy** (mất thời gian)
- **Sử dụng model đã train vào ban ngày** (nhanh, tiện)
- **Backup thư mục `trained_models/`** trước khi train lại

