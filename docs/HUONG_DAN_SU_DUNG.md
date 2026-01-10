# HƯỚNG DẪN SỬ DỤNG HỆ THỐNG DỰ ĐOÁN CLO

## 📖 MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Hệ thống hoạt động như thế nào?](#2-hệ-thống-hoạt-động-như-thế-nào)
3. [Cách sử dụng cơ bản](#3-cách-sử-dụng-cơ-bản)
4. [Tích hợp vào Backend đơn giản](#4-tích-hợp-vào-backend-đơn-giản)
5. [Ví dụ thực tế](#5-ví-dụ-thực-tế)
6. [Giải thích kết quả](#6-giải-thích-kết-quả)

---

## 1. GIỚI THIỆU

### Hệ thống này làm gì?

Hệ thống **CLO Prediction** giúp bạn:

✅ **Phân tích điểm CLO** của sinh viên  
✅ **Tìm ra nguyên nhân** tại sao sinh viên học kém  
✅ **Đề xuất giải pháp** để cải thiện kết quả học tập  
✅ **Phân tích cả lớp** hoặc **từng sinh viên cụ thể**  

### Ai cần dùng?

- **Giảng viên**: Muốn hiểu tại sao sinh viên học kém và cải thiện phương pháp giảng dạy
- **Nhà trường**: Theo dõi chất lượng đào tạo và can thiệp kịp thời
- **Developer**: Tích hợp vào hệ thống quản lý sinh viên

---

## 2. HỆ THỐNG HOẠT ĐỘNG NHƯ THẾ NÀO?

### Bước 1: Thu thập dữ liệu

Hệ thống cần các thông tin sau:

```
📚 Môn học: Mã môn học (VD: INF1383)
👨‍🏫 Giảng viên: Tên hoặc mã giảng viên
🎓 Sinh viên: Mã sinh viên và điểm CLO (0-6)
```

### Bước 2: Phân tích bằng AI

Hệ thống sử dụng **Machine Learning** (Random Forest & Gradient Boosting) để:

1. **Đánh giá mức độ nghiêm trọng** (Low/Medium/High/Critical)
2. **Tìm kiếm trong 30,000+ mẫu dữ liệu** về nguyên nhân & giải pháp
3. **Chọn ra top K nguyên nhân** phù hợp nhất

### Bước 3: Trả về kết quả

```
📊 Thống kê lớp (nếu phân tích lớp)
💡 Nguyên nhân chính
🎯 Giải pháp khắc phục
⚠️  Danh sách sinh viên cần can thiệp
```

---

## 3. CÁCH SỬ DỤNG CƠ BẢN

### 3.1. Chạy từ Command Line (Demo)

#### Phân tích lớp học:

```bash
python run_interactive_with_file.py
```

Sau đó chọn:
```
1. Phân tích LỚP HỌC - Nhập từ FILE Excel/CSV
```

Chuẩn bị file Excel với format:

| MSSV      | HoTen          | DiemCLO |
|-----------|----------------|---------|
| SV001     | Nguyễn Văn A   | 5.5     |
| SV002     | Trần Thị B     | 4.8     |
| SV003     | Lê Văn C       | 2.5     |

#### Phân tích cá nhân:

```bash
python run.py
```

Nhập thông tin:
```
Mã sinh viên: SV001
Giảng viên: Nguyễn Văn A
Mã môn học: INF1383
```

---

### 3.2. Sử dụng trong Python Code

#### Ví dụ 1: Phân tích lớp học

```python
from model_loader import ClassAnalyzer

# Khởi tạo analyzer
analyzer = ClassAnalyzer(
    model_path="trained_models/class_model/class_model.pkl",
    metadata_path="trained_models/class_model/metadata.pkl"
)

# Phân tích
result = analyzer.analyze(
    subject_id="INF1383",
    lecturer_name="Nguyễn Văn A",
    student_list=["SV001", "SV002", "SV003", "SV004", "SV005"],
    scores=[5.5, 4.8, 3.2, 2.5, 4.5],
    top_k=3
)

# In kết quả
print(f"Điểm trung bình lớp: {result['statistics']['average_score']:.2f}")
print(f"Tỉ lệ đạt: {result['statistics']['pass_rate']:.1f}%")
print(f"Số SV cần can thiệp: {len(result['students_need_attention'])}")
```

#### Ví dụ 2: Phân tích cá nhân

```python
from model_loader import IndividualAnalyzer

# Khởi tạo analyzer
analyzer = IndividualAnalyzer(
    model_path="trained_models/individual_model/individual_model.pkl",
    metadata_path="trained_models/individual_model/metadata.pkl"
)

# Phân tích
result = analyzer.analyze(
    subject_id="INF1383",
    lecturer_name="Nguyễn Văn A",
    student_id="SV001",
    clo_score=3.5,
    top_k=5
)

# In kết quả
print(f"Xếp loại: {result['performance_level']}")
print(f"Mức độ: {result['clo_analysis']['severity_level']}")

for i, item in enumerate(result['clo_analysis']['results'], 1):
    print(f"\nNguyên nhân {i}: {item['reason']}")
    print(f"Giải pháp: {item['solution']}")
```

---

## 4. TÍCH HỢP VÀO BACKEND ĐỠN GIẢN

### 4.1. Chuẩn bị

**Bước 1**: Copy thư mục `hsmh_model` vào project backend của bạn

```bash
cp -r hsmh_model /path/to/your/backend/
```

**Bước 2**: Cài đặt dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

**Bước 3**: Kiểm tra file models đã có chưa

```bash
ls hsmh_model/trained_models/class_model/
# Phải có: class_model.pkl và metadata.pkl

ls hsmh_model/trained_models/individual_model/
# Phải có: individual_model.pkl và metadata.pkl
```

---

### 4.2. Tích hợp với FastAPI (Đơn giản nhất)

#### File: `backend.py`

```python
from fastapi import FastAPI
import sys
sys.path.append('./hsmh_model')

from model_loader import ClassAnalyzer, IndividualAnalyzer

app = FastAPI()

# Load models khi khởi động
class_analyzer = ClassAnalyzer(
    model_path="hsmh_model/trained_models/class_model/class_model.pkl",
    metadata_path="hsmh_model/trained_models/class_model/metadata.pkl"
)

individual_analyzer = IndividualAnalyzer(
    model_path="hsmh_model/trained_models/individual_model/individual_model.pkl",
    metadata_path="hsmh_model/trained_models/individual_model/metadata.pkl"
)

@app.post("/analyze/class")
def analyze_class(data: dict):
    """
    Input:
    {
        "subject_id": "INF1383",
        "lecturer_name": "Nguyễn Văn A",
        "students": [
            {"student_id": "SV001", "score": 5.5},
            {"student_id": "SV002", "score": 4.8}
        ]
    }
    """
    student_list = [s['student_id'] for s in data['students']]
    scores = [s['score'] for s in data['students']]
    
    result = class_analyzer.analyze(
        subject_id=data['subject_id'],
        lecturer_name=data['lecturer_name'],
        student_list=student_list,
        scores=scores,
        top_k=3
    )
    
    return {"success": True, "data": result}

@app.post("/analyze/individual")
def analyze_individual(data: dict):
    """
    Input:
    {
        "subject_id": "INF1383",
        "lecturer_name": "Nguyễn Văn A",
        "student_id": "SV001",
        "clo_score": 3.5
    }
    """
    result = individual_analyzer.analyze(
        subject_id=data['subject_id'],
        lecturer_name=data['lecturer_name'],
        student_id=data['student_id'],
        clo_score=data['clo_score'],
        top_k=5
    )
    
    return {"success": True, "data": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Chạy server:

```bash
python backend.py
```

#### Test API bằng curl:

```bash
# Phân tích lớp
curl -X POST http://localhost:8000/analyze/class \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "students": [
      {"student_id": "SV001", "score": 5.5},
      {"student_id": "SV002", "score": 4.8},
      {"student_id": "SV003", "score": 2.5}
    ]
  }'

# Phân tích cá nhân
curl -X POST http://localhost:8000/analyze/individual \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "student_id": "SV001",
    "clo_score": 3.5
  }'
```

---

### 4.3. Tích hợp với Node.js Backend (Gọi Python service)

#### Cách 1: Sử dụng child_process

```javascript
// backend/analyzer.js
const { spawn } = require('child_process');

function analyzeClass(subjectId, lecturerName, students) {
  return new Promise((resolve, reject) => {
    const python = spawn('python', [
      'hsmh_model/analyze_wrapper.py',
      'class',
      JSON.stringify({ subjectId, lecturerName, students })
    ]);

    let result = '';
    python.stdout.on('data', (data) => {
      result += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        resolve(JSON.parse(result));
      } else {
        reject(new Error('Python script failed'));
      }
    });
  });
}

module.exports = { analyzeClass };
```

#### Cách 2: Chạy Python API riêng và gọi HTTP

```javascript
// backend/analyzer.js
const axios = require('axios');

const PYTHON_API_URL = 'http://localhost:8000';

async function analyzeClass(subjectId, lecturerName, students) {
  try {
    const response = await axios.post(`${PYTHON_API_URL}/analyze/class`, {
      subject_id: subjectId,
      lecturer_name: lecturerName,
      students: students
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing class:', error);
    throw error;
  }
}

module.exports = { analyzeClass };
```

---

## 5. VÍ DỤ THỰC TẾ

### Tình huống 1: Giảng viên muốn biết lớp học tập như thế nào

**Input:**
- Môn: Lập trình Python (INF1383)
- Giảng viên: Nguyễn Văn A
- 5 sinh viên với điểm: 5.5, 4.8, 3.2, 2.5, 4.5

**Code:**

```python
from model_loader import ClassAnalyzer

analyzer = ClassAnalyzer(
    model_path="trained_models/class_model/class_model.pkl",
    metadata_path="trained_models/class_model/metadata.pkl"
)

result = analyzer.analyze(
    subject_id="INF1383",
    lecturer_name="Nguyễn Văn A",
    student_list=["SV001", "SV002", "SV003", "SV004", "SV005"],
    scores=[5.5, 4.8, 3.2, 2.5, 4.5],
    top_k=3
)
```

**Output:**

```json
{
  "statistics": {
    "total_students": 5,
    "average_score": 4.1,
    "pass_rate": 80.0
  },
  "class_general_analysis": {
    "severity_level": "Medium",
    "results": [
      {
        "reason": "Phương pháp giảng dạy chưa phù hợp với đa số sinh viên",
        "solution": "Cần điều chỉnh phương pháp giảng dạy, tăng cường tương tác"
      }
    ]
  },
  "students_need_attention": [
    {
      "student_id": "SV004",
      "clo_score": 2.5,
      "performance_level": "Yếu"
    }
  ]
}
```

**Giải thích:**
- Lớp có 5 sinh viên, điểm TB là 4.1/6 (Khá)
- 80% sinh viên đạt chuẩn (điểm >= 3.0)
- Có 1 sinh viên cần can thiệp (SV004 với 2.5 điểm)
- Mức độ chung: Medium (cần chú ý)
- Nguyên nhân: Phương pháp giảng dạy chưa phù hợp
- Giải pháp: Tăng cường tương tác trong lớp

---

### Tình huống 2: Sinh viên X học kém, tìm nguyên nhân

**Input:**
- Môn: INF1383
- Giảng viên: Nguyễn Văn A
- Sinh viên: SV003
- Điểm CLO: 2.5

**Code:**

```python
from model_loader import IndividualAnalyzer

analyzer = IndividualAnalyzer(
    model_path="trained_models/individual_model/individual_model.pkl",
    metadata_path="trained_models/individual_model/metadata.pkl"
)

result = analyzer.analyze(
    subject_id="INF1383",
    lecturer_name="Nguyễn Văn A",
    student_id="SV003",
    clo_score=2.5,
    top_k=5
)
```

**Output:**

```json
{
  "student_id": "SV003",
  "clo_score": 2.5,
  "performance_level": "Yếu",
  "clo_analysis": {
    "severity_level": "High",
    "results": [
      {
        "reason": "Sinh viên nghỉ học nhiều, không theo dõi bài giảng thường xuyên",
        "solution": "Tăng cường theo dõi chuyên cần, liên hệ phụ huynh, hỗ trợ học bù"
      },
      {
        "reason": "Không hoàn thành bài tập được giao",
        "solution": "Gặp riêng sinh viên, hướng dẫn cách làm bài tập từng bước"
      },
      {
        "reason": "Thiếu kỹ năng cơ bản về môn học",
        "solution": "Sắp xếp lớp học phụ đạo, cung cấp tài liệu bổ trợ"
      }
    ]
  }
}
```

**Giải thích:**
- Sinh viên SV003 xếp loại "Yếu" (2.5/6)
- Mức độ nghiêm trọng: High (cần can thiệp gấp)
- Có 3 nguyên nhân chính:
  1. Nghỉ học nhiều
  2. Không làm bài tập
  3. Thiếu kỹ năng cơ bản
- Giải pháp:
  1. Theo dõi chuyên cần chặt chẽ
  2. Hướng dẫn làm bài tập
  3. Tổ chức lớp phụ đạo

---

### Tình huống 3: Tích hợp vào Web App

**Frontend (React):**

```javascript
// AnalyzeClass.jsx
import { useState } from 'react';
import axios from 'axios';

function AnalyzeClass() {
  const [students, setStudents] = useState([
    { student_id: 'SV001', score: 5.5 },
    { student_id: 'SV002', score: 4.8 },
  ]);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    try {
      const response = await axios.post('http://localhost:8000/analyze/class', {
        subject_id: 'INF1383',
        lecturer_name: 'Nguyễn Văn A',
        students: students
      });
      setResult(response.data.data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h2>Phân tích lớp học</h2>
      <button onClick={handleAnalyze}>Phân tích</button>
      
      {result && (
        <div>
          <h3>Kết quả</h3>
          <p>Điểm TB: {result.statistics.average_score.toFixed(2)}</p>
          <p>Tỉ lệ đạt: {result.statistics.pass_rate.toFixed(1)}%</p>
          
          <h4>Sinh viên cần can thiệp:</h4>
          {result.students_need_attention.map(student => (
            <div key={student.student_id}>
              {student.student_id}: {student.clo_score} - {student.performance_level}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default AnalyzeClass;
```

**Backend (FastAPI):**

```python
# backend.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append('./hsmh_model')

from model_loader import ClassAnalyzer

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
class_analyzer = ClassAnalyzer(
    model_path="hsmh_model/trained_models/class_model/class_model.pkl",
    metadata_path="hsmh_model/trained_models/class_model/metadata.pkl"
)

@app.post("/analyze/class")
def analyze_class(data: dict):
    student_list = [s['student_id'] for s in data['students']]
    scores = [s['score'] for s in data['students']]
    
    result = class_analyzer.analyze(
        subject_id=data['subject_id'],
        lecturer_name=data['lecturer_name'],
        student_list=student_list,
        scores=scores,
        top_k=3
    )
    
    return {"success": True, "data": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 6. GIẢI THÍCH KẾT QUẢ

### 6.1. Thống kê lớp (Class Statistics)

```json
{
  "total_students": 5,        // Tổng số sinh viên
  "average_score": 4.1,       // Điểm trung bình lớp (0-6)
  "min_score": 2.5,           // Điểm thấp nhất
  "max_score": 5.5,           // Điểm cao nhất
  "pass_rate": 80.0           // Tỉ lệ % sinh viên đạt (>= 3.0)
}
```

**Giải thích:**
- `total_students`: Số lượng sinh viên trong lớp
- `average_score`: Điểm CLO trung bình (càng cao càng tốt)
- `pass_rate`: Tỉ lệ sinh viên đạt chuẩn (>= 3.0/6)
  - >= 90%: Lớp xuất sắc
  - 70-90%: Lớp tốt
  - 50-70%: Lớp trung bình
  - < 50%: Cần cải thiện khẩn cấp

---

### 6.2. Severity Level (Mức độ nghiêm trọng)

| Level    | Điểm CLO  | Ý nghĩa                      | Cần làm gì?                        |
|----------|-----------|------------------------------|------------------------------------|
| Low      | 5.0 - 6.0 | Không có vấn đề nghiêm trọng | Duy trì, không cần can thiệp       |
| Medium   | 3.0 - 5.0 | Có vấn đề nhỏ, cần chú ý     | Theo dõi, hỗ trợ nếu cần           |
| High     | 2.0 - 3.0 | Vấn đề đáng kể, cần can thiệp| Can thiệp ngay, gặp riêng sinh viên|
| Critical | 0.0 - 2.0 | Rất nghiêm trọng, khẩn cấp   | Can thiệp khẩn cấp, liên hệ phụ huynh|

---

### 6.3. Performance Level (Xếp loại)

| Level      | Điểm CLO | Mô tả                    |
|------------|----------|--------------------------|
| Xuất sắc   | 5.5 - 6  | Rất xuất sắc, vượt chuẩn |
| Giỏi       | 5.0 - 5.5| Tốt, đạt chuẩn cao       |
| Khá        | 4.0 - 5.0| Khá, đạt chuẩn           |
| Trung bình | 3.0 - 4.0| Đạt chuẩn đầu ra tối thiểu|
| Yếu        | 2.0 - 3.0| Dưới chuẩn, cần cải thiện|
| Kém        | 0.0 - 2.0| Rất kém, cần hỗ trợ đặc biệt|

---

### 6.4. Reasons & Solutions (Nguyên nhân & Giải pháp)

**Cấu trúc:**

```json
{
  "reason": "Sinh viên nghỉ học nhiều...",
  "solution": "Tăng cường theo dõi chuyên cần...",
  "severity": "High",
  "confidence": 0.85
}
```

**Giải thích:**
- `reason`: Nguyên nhân dẫn đến vấn đề
- `solution`: Giải pháp khắc phục cụ thể
- `severity`: Mức độ nghiêm trọng của nguyên nhân này
- `confidence`: Độ tin cậy của dự đoán (0-1)
  - >= 0.8: Rất chính xác
  - 0.6-0.8: Khá chính xác
  - < 0.6: Cần xem xét thêm

---

## 7. XỬ LÝ CÁC TRƯỜNG HỢP ĐẶC BIỆT

### 7.1. Lớp có quá nhiều sinh viên yếu (>30%)

**Kết quả:**
```json
{
  "statistics": {
    "pass_rate": 65.0
  },
  "class_general_analysis": {
    "severity_level": "High"
  }
}
```

**Khuyến nghị:**
1. Xem xét lại phương pháp giảng dạy
2. Tổ chức lớp phụ đạo cho nhóm yếu
3. Điều chỉnh nội dung, tốc độ giảng
4. Tăng cường bài tập thực hành

---

### 7.2. Sinh viên có điểm CLO = 0

**Giải thích:**
- Sinh viên chưa có dữ liệu hoặc vắng mặt hoàn toàn
- Không thể phân tích

**Khuyến nghị:**
- Liên hệ sinh viên ngay lập tức
- Kiểm tra lý do vắng mặt
- Cân nhắc cho bảo lưu hoặc rút môn

---

### 7.3. Tất cả sinh viên đều đạt điểm cao

**Kết quả:**
```json
{
  "statistics": {
    "average_score": 5.3,
    "pass_rate": 100.0
  },
  "class_general_analysis": {
    "severity_level": "Low"
  },
  "students_need_attention": []
}
```

**Giải thích:**
- Lớp học rất tốt, không có vấn đề
- Không cần can thiệp
- Duy trì phương pháp hiện tại

---

## 8. CHECKLIST TÍCH HỢP

### Trước khi tích hợp:

- [ ] Đã cài đặt Python 3.9+
- [ ] Đã cài đặt tất cả dependencies (pandas, numpy, scikit-learn, ...)
- [ ] Đã có file trained models (.pkl)
- [ ] Đã test chạy được ví dụ đơn giản

### Khi tích hợp:

- [ ] Load models 1 lần khi khởi động server (không load mỗi request)
- [ ] Validate input (điểm phải 0-6, student_list không rỗng, ...)
- [ ] Handle exceptions (model không load được, input sai, ...)
- [ ] Log lại requests để debug
- [ ] Test với dữ liệu thật

### Sau khi tích hợp:

- [ ] Kiểm tra performance (response time < 2s)
- [ ] Kiểm tra memory usage
- [ ] Setup monitoring & alerts
- [ ] Viết documentation cho team

---

## 9. TROUBLESHOOTING

### Lỗi: "Model file not found"

**Nguyên nhân:** Đường dẫn tới file .pkl sai

**Giải pháp:**
```python
import os

# Kiểm tra file có tồn tại không
model_path = "hsmh_model/trained_models/class_model/class_model.pkl"
if not os.path.exists(model_path):
    print(f"❌ File không tồn tại: {model_path}")
else:
    print(f"✅ File tồn tại")

# Dùng đường dẫn tuyệt đối
model_path = os.path.abspath("hsmh_model/trained_models/class_model/class_model.pkl")
```

---

### Lỗi: "Model not loaded"

**Nguyên nhân:** Chưa gọi `load()` hoặc load thất bại

**Giải pháp:**
```python
from model_loader import ModelLoader

loader = ModelLoader(model_path, metadata_path)

try:
    loader.load()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
```

---

### Lỗi: "Number of students does not match scores"

**Nguyên nhân:** Độ dài của `student_list` và `scores` không khớp

**Giải pháp:**
```python
# Kiểm tra trước khi gọi analyze
if len(student_list) != len(scores):
    print(f"❌ Mismatch: {len(student_list)} students but {len(scores)} scores")
else:
    result = analyzer.analyze(...)
```

---

### Model chạy chậm

**Nguyên nhân:** Load model mỗi request

**Giải pháp:**
```python
# ❌ SAI: Load mỗi request (chậm)
@app.post("/analyze")
def analyze(data):
    analyzer = ClassAnalyzer(...)  # Load mỗi request
    return analyzer.analyze(...)

# ✅ ĐÚNG: Load 1 lần khi khởi động
class_analyzer = ClassAnalyzer(...)  # Load 1 lần

@app.post("/analyze")
def analyze(data):
    return class_analyzer.analyze(...)  # Dùng lại
```

---

## 10. LIÊN HỆ HỖ TRỢ

Nếu bạn gặp vấn đề, hãy liên hệ:

📧 **Email**: support@example.com  
📱 **Hotline**: 0123-456-789  
📚 **Documentation**: https://docs.example.com  
💬 **Slack**: #hsmh-model-support  

---

**Chúc bạn tích hợp thành công! 🎉**

