# TÀI LIỆU KỸ THUẬT - HỆ THỐNG DỰ ĐOÁN CLO VÀ PHÂN TÍCH SINH VIÊN

## 📋 MỤC LỤC

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Các module chính](#3-các-module-chính)
4. [Luồng dữ liệu](#4-luồng-dữ-liệu)
5. [Hướng dẫn tích hợp Backend](#5-hướng-dẫn-tích-hợp-backend)
6. [API Reference](#6-api-reference)
7. [Deployment](#7-deployment)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Mục đích
Hệ thống **CLO Prediction System** là một ứng dụng Machine Learning được thiết kế để:
- **Dự đoán điểm CLO** (Course Learning Outcomes) của sinh viên
- **Phân tích nguyên nhân** dẫn đến kết quả học tập kém
- **Đề xuất giải pháp** cải thiện hiệu quả học tập
- **Đánh giá hiệu quả** của phương pháp giảng dạy (PPGD) và phương pháp đánh giá (PPDG)

### 1.2. Tính năng chính
✅ **Dự đoán điểm CLO** dựa trên nhiều yếu tố (PPGD, PPDG, điểm rèn luyện, điểm giữa kỳ, chuyên cần)  
✅ **Phân tích lớp học**: Đánh giá tổng quan cả lớp, xác định sinh viên cần hỗ trợ  
✅ **Phân tích cá nhân**: Phân tích chi tiết từng sinh viên  
✅ **Đề xuất giải pháp thông minh**: Dựa trên 6 datasets với 30,000+ mẫu dữ liệu  
✅ **Hỗ trợ nhiều mức độ nghiêm trọng**: Low, Medium, High, Critical  

### 1.3. Công nghệ sử dụng
- **Python 3.9+**
- **scikit-learn**: Random Forest, Gradient Boosting
- **pandas, numpy**: Xử lý dữ liệu
- **matplotlib, seaborn**: Trực quan hóa
- **pickle**: Lưu/tải model

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1. Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND / API                           │
│               (FastAPI / Django / Flask)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL LOADER LAYER                            │
│   • ClassAnalyzer      • IndividualAnalyzer                      │
│   • PredictionTools    • ModelLoader                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              UNIFIED INTEGRATION LAYER                          │
│   • unified_integration.py                                      │
│   • analyze_class() / analyze_individual()                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CORE MODEL LAYER                              │
│   • UnifiedReasonsSolutionsModel (6 datasets)                   │
│   • CLOPredictor (dự đoán điểm)                                 │
│   • PPDGAnalyzer (phân tích phương pháp)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│   • DiemTong.xlsx        • PPDG.xlsx                            │
│   • diemrenluyen.xlsx    • PPGD.xlsx                            │
│   • 6 CSV datasets (30,000+ samples)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2. Cấu trúc thư mục

```
hsmh_model/
├── model/                          # Core models
│   ├── unified_reasons_solutions_model.py    # Model chính (6 datasets)
│   ├── predictor.py                          # CLO Predictor
│   ├── data_loader.py                        # Load dữ liệu
│   ├── feature_engineering.py                # Tạo features
│   ├── model_trainer.py                      # Train models
│   ├── analyze_ppdg.py                       # Phân tích PPDG
│   └── unified_input_handler.py              # Xử lý input
│
├── model_loader.py                 # API Layer (QUAN TRỌNG!)
├── unified_integration.py          # Integration Layer
├── main.py                         # CLI demo
├── run.py                          # Interactive runner
├── usage_example.py                # Ví dụ sử dụng
│
├── trained_models/                 # Trained models
│   ├── class_model/
│   │   ├── class_model.pkl
│   │   └── metadata.pkl
│   └── individual_model/
│       ├── individual_model.pkl
│       └── metadata.pkl
│
├── dulieu/                         # Datasets
│   ├── DiemTong.xlsx               # Điểm tổng sinh viên
│   ├── PPDG.xlsx                   # Phương pháp đánh giá
│   ├── PPGD.xlsx                   # Phương pháp giảng dạy
│   ├── diemrenluyen.xlsx           # Điểm rèn luyện
│   └── *.csv                       # 6 reason-solution datasets
│
└── requirements.txt                # Dependencies
```

---

## 3. CÁC MODULE CHÍNH

### 3.1. `model_loader.py` ⭐ (API LAYER - QUAN TRỌNG NHẤT)

**Đây là file chính để backend tích hợp!**

#### 3.1.1. `ModelLoader` class
```python
class ModelLoader:
    """Load trained model từ file pickle"""
    
    def __init__(self, model_path, metadata_path=None):
        """
        Args:
            model_path: Đường dẫn tới file .pkl của model
            metadata_path: Đường dẫn tới file metadata (optional)
        """
    
    def load(self):
        """Load model từ file"""
    
    def predict_reason_solution(self, dataset_key, features, top_k=3):
        """
        Dự đoán reasons & solutions
        
        Args:
            dataset_key: Loại dataset ('teaching_methods', 'evaluation_methods', 
                         'student_conduct', 'academic_midterm', 'clo_attendance', 
                         'self_study')
            features: List số liệu [score_normalized]
            top_k: Số lượng reasons/solutions trả về
        
        Returns:
            {
                'dataset': str,
                'severity_level': str,      # Low/Medium/High/Critical
                'severity_confidence': float,
                'results': [
                    {
                        'reason': str,
                        'solution': str,
                        'severity': str,
                        'confidence': float
                    }
                ]
            }
        """
```

#### 3.1.2. `ClassAnalyzer` class ⭐
```python
class ClassAnalyzer:
    """Phân tích cho cả lớp học"""
    
    def analyze(self, subject_id, lecturer_name, student_list, scores, top_k=3):
        """
        Phân tích tổng quan lớp học
        
        Args:
            subject_id: Mã môn học (VD: "INF1383")
            lecturer_name: Tên/Mã giảng viên
            student_list: List mã sinh viên ["SV001", "SV002", ...]
            scores: List điểm CLO [5.5, 4.8, 3.2, ...]
            top_k: Số reasons/solutions
        
        Returns:
            {
                'mode': 'class',
                'subject_id': str,
                'lecturer_name': str,
                'statistics': {
                    'total_students': int,
                    'average_score': float,
                    'min_score': float,
                    'max_score': float,
                    'pass_rate': float      # Tỉ lệ % đạt (>= 3.0)
                },
                'class_general_analysis': {
                    'dataset': str,
                    'severity_level': str,
                    'severity_confidence': float,
                    'results': [...]        # Top K reasons & solutions chung
                },
                'students_need_attention': [
                    {
                        'student_id': str,
                        'clo_score': float,
                        'performance_level': str    # Xuất sắc/Giỏi/Khá/...
                    }
                ]
            }
        """
```

#### 3.1.3. `IndividualAnalyzer` class ⭐
```python
class IndividualAnalyzer:
    """Phân tích chi tiết cho 1 sinh viên"""
    
    def analyze(self, subject_id, lecturer_name, student_id, clo_score, top_k=5):
        """
        Phân tích chi tiết 1 sinh viên
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên/Mã giảng viên
            student_id: Mã sinh viên
            clo_score: Điểm CLO (0-6)
            top_k: Số reasons/solutions
        
        Returns:
            {
                'mode': 'individual',
                'subject_id': str,
                'lecturer_name': str,
                'student_id': str,
                'clo_score': float,
                'performance_level': str,
                'clo_analysis': {
                    'severity_level': str,
                    'results': [...]        # Top K reasons & solutions
                }
            }
        """
```

#### 3.1.4. `PredictionTools` class
```python
class PredictionTools:
    """Công cụ dự đoán cho từng loại dataset"""
    
    def predict_teaching_methods(self, score, top_k=3):
        """Dự đoán cho Phương pháp giảng dạy"""
    
    def predict_evaluation_methods(self, score, top_k=3):
        """Dự đoán cho Phương pháp đánh giá"""
    
    def predict_student_conduct(self, score, top_k=3):
        """Dự đoán cho Điểm rèn luyện"""
    
    def predict_academic_midterm(self, score, top_k=3):
        """Dự đoán cho Điểm giữa kỳ"""
    
    def predict_clo_attendance(self, score, top_k=3):
        """Dự đoán cho CLO Attendance"""
```

---

### 3.2. `unified_integration.py`

Module tích hợp cao cấp với nhiều tính năng mở rộng.

#### Hàm chính:

```python
def analyze_class_v2(subject_id, lecturer_id, students_data, top_k=3):
    """
    Phân tích lớp - Version 2 (dễ dùng hơn cho Backend)
    
    Args:
        subject_id: Mã môn học
        lecturer_id: Mã giảng viên
        students_data: List of dict
            [
                {"mssv": "SV001", "ho_ten": "Nguyễn Văn A", "diem_clo": 5.5},
                {"mssv": "SV002", "ho_ten": "Trần Văn B", "diem_clo": 4.8},
                ...
            ]
        top_k: Số reasons/solutions
    
    Returns:
        Dictionary kết quả phân tích
    """

def analyze_individual(subject_id, lecturer_id, student_id, top_k=3):
    """
    Phân tích cá nhân - TỰ ĐỘNG DỰ ĐOÁN điểm CLO
    
    Lưu ý: Hàm này sẽ tự động dự đoán điểm CLO từ dữ liệu sinh viên
    """
```

---

### 3.3. `UnifiedReasonsSolutionsModel`

Model chính xử lý 6 loại datasets:

1. **Teaching Methods** (Phương pháp giảng dạy)
2. **Evaluation Methods** (Phương pháp đánh giá)
3. **Student Conduct** (Điểm rèn luyện)
4. **Academic Midterm** (Điểm giữa kỳ)
5. **CLO Attendance** (Chuyên cần CLO)
6. **Self-Study** (Tự học)

Mỗi dataset có **5000 mẫu dữ liệu** với:
- **Reasons**: Nguyên nhân dẫn đến vấn đề
- **Solutions**: Giải pháp khắc phục
- **Severity Level**: Mức độ nghiêm trọng (Low/Medium/High/Critical)

#### Thuật toán:
- **Random Forest Classifier** (200 trees, max_depth=15)
- **Gradient Boosting Classifier** (100 estimators, max_depth=10)
- Tự động chọn model tốt nhất dựa trên accuracy

---

### 3.4. `analyze_ppdg.py`

Phân tích hiệu quả của **Phương pháp đánh giá (PPDG)**.

**Các PPDG được hỗ trợ:**
- EM 1: Đánh giá chuyên cần
- EM 2: Đánh giá bài tập cá nhân
- EM 3: Đánh giá thuyết trình
- EM 4: Đánh giá làm việc nhóm
- EM 5: Đánh giá tự học tại thư viện
- EM 6: Kiểm tra viết
- EM 7: Kiểm tra trắc nghiệm
- EM 8: Đánh giá báo cáo/tiểu luận
- EM 9: Đánh giá thực tập
- EM 10: Đánh giá báo cáo thực tập tại doanh nghiệp
- EM 11: Đánh giá thực hành tại phòng thí nghiệm
- EM 12: Đánh giá bài tập lớn/Đồ án cá nhân
- EM 14: Đánh giá khóa luận tốt nghiệp

**Tính năng:**
- Phân tích tần suất sử dụng PPDG
- Tính ma trận tương quan giữa các PPDG
- Đánh giá hiệu quả từng PPDG
- Đề xuất PPDG phù hợp

---

## 4. LUỒNG DỮ LIỆU

### 4.1. Luồng phân tích lớp học

```
1. Backend nhận request từ Frontend
   ↓
2. Gọi ClassAnalyzer.analyze()
   - Input: subject_id, lecturer_name, student_list, scores
   ↓
3. ClassAnalyzer xử lý:
   - Tính thống kê lớp (điểm TB, min, max, pass_rate)
   - Normalize điểm (điểm/6.0)
   ↓
4. Gọi UnifiedReasonsSolutionsModel
   - Dự đoán severity_level từ điểm TB lớp
   - Lấy top_k reasons & solutions phù hợp
   ↓
5. Xác định sinh viên cần can thiệp (điểm < 3.0)
   ↓
6. Trả về JSON kết quả cho Backend
   ↓
7. Backend format và gửi cho Frontend
```

### 4.2. Luồng phân tích cá nhân

```
1. Backend nhận request từ Frontend
   ↓
2. Gọi IndividualAnalyzer.analyze()
   - Input: subject_id, lecturer_name, student_id, clo_score
   ↓
3. IndividualAnalyzer xử lý:
   - Normalize điểm
   - Xác định performance_level
   ↓
4. Gọi UnifiedReasonsSolutionsModel
   - Dự đoán severity_level
   - Lấy top_k reasons & solutions chi tiết
   ↓
5. Trả về JSON kết quả cho Backend
   ↓
6. Backend format và gửi cho Frontend
```

---

## 5. HƯỚNG DẪN TÍCH HỢP BACKEND

### 5.1. Cài đặt

#### Bước 1: Clone/Copy project
```bash
cd /path/to/your/backend/project
cp -r /path/to/hsmh_model ./hsmh_model
```

#### Bước 2: Cài đặt dependencies
```bash
pip install -r hsmh_model/requirements.txt
```

hoặc thêm vào `requirements.txt` của backend:
```
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.2.2
matplotlib==3.9.4
seaborn==0.13.2
openpyxl==3.1.2
```

#### Bước 3: Kiểm tra trained models
Đảm bảo có 2 thư mục:
```
hsmh_model/trained_models/
├── class_model/
│   ├── class_model.pkl
│   └── metadata.pkl
└── individual_model/
    ├── individual_model.pkl
    └── metadata.pkl
```

---

### 5.2. Tích hợp với FastAPI ⭐ (KHUYẾN NGHỊ)

#### File: `backend/app.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
sys.path.append('./hsmh_model')

from model_loader import ClassAnalyzer, IndividualAnalyzer

app = FastAPI(title="CLO Prediction API")

# Khởi tạo analyzers (global)
class_analyzer = None
individual_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Load models khi khởi động server"""
    global class_analyzer, individual_analyzer
    
    try:
        print("🔄 Đang load models...")
        
        class_analyzer = ClassAnalyzer(
            model_path="hsmh_model/trained_models/class_model/class_model.pkl",
            metadata_path="hsmh_model/trained_models/class_model/metadata.pkl"
        )
        
        individual_analyzer = IndividualAnalyzer(
            model_path="hsmh_model/trained_models/individual_model/individual_model.pkl",
            metadata_path="hsmh_model/trained_models/individual_model/metadata.pkl"
        )
        
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise


# ==================== PYDANTIC MODELS ====================

class StudentScore(BaseModel):
    student_id: str
    score: float  # 0-6

class ClassAnalysisRequest(BaseModel):
    subject_id: str
    lecturer_name: str
    students: List[StudentScore]
    top_k: Optional[int] = 3

class IndividualAnalysisRequest(BaseModel):
    subject_id: str
    lecturer_name: str
    student_id: str
    clo_score: float  # 0-6
    top_k: Optional[int] = 5


# ==================== ENDPOINTS ====================

@app.get("/")
def read_root():
    return {
        "message": "CLO Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/api/analyze/class",
            "/api/analyze/individual",
            "/health"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "class_analyzer": "loaded" if class_analyzer else "not loaded",
        "individual_analyzer": "loaded" if individual_analyzer else "not loaded"
    }

@app.post("/api/analyze/class")
def analyze_class_endpoint(request: ClassAnalysisRequest):
    """
    Phân tích lớp học
    
    Example request:
    {
        "subject_id": "INF1383",
        "lecturer_name": "Nguyễn Văn A",
        "students": [
            {"student_id": "SV001", "score": 5.5},
            {"student_id": "SV002", "score": 4.8},
            {"student_id": "SV003", "score": 3.2}
        ],
        "top_k": 3
    }
    """
    if class_analyzer is None:
        raise HTTPException(status_code=500, detail="Class analyzer not loaded")
    
    try:
        # Extract student_list and scores
        student_list = [s.student_id for s in request.students]
        scores = [s.score for s in request.students]
        
        # Validate
        if len(student_list) != len(scores):
            raise HTTPException(status_code=400, detail="Mismatch between student_list and scores")
        
        if not student_list:
            raise HTTPException(status_code=400, detail="Student list is empty")
        
        # Analyze
        result = class_analyzer.analyze(
            subject_id=request.subject_id,
            lecturer_name=request.lecturer_name,
            student_list=student_list,
            scores=scores,
            top_k=request.top_k
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/individual")
def analyze_individual_endpoint(request: IndividualAnalysisRequest):
    """
    Phân tích cá nhân
    
    Example request:
    {
        "subject_id": "INF1383",
        "lecturer_name": "Nguyễn Văn A",
        "student_id": "SV001",
        "clo_score": 3.5,
        "top_k": 5
    }
    """
    if individual_analyzer is None:
        raise HTTPException(status_code=500, detail="Individual analyzer not loaded")
    
    try:
        # Validate
        if not 0 <= request.clo_score <= 6:
            raise HTTPException(status_code=400, detail="CLO score must be between 0 and 6")
        
        # Analyze
        result = individual_analyzer.analyze(
            subject_id=request.subject_id,
            lecturer_name=request.lecturer_name,
            student_id=request.student_id,
            clo_score=request.clo_score,
            top_k=request.top_k
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### Chạy server:
```bash
python backend/app.py
```

#### Test API:
```bash
# Health check
curl http://localhost:8000/health

# Analyze class
curl -X POST http://localhost:8000/api/analyze/class \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "students": [
      {"student_id": "SV001", "score": 5.5},
      {"student_id": "SV002", "score": 4.8},
      {"student_id": "SV003", "score": 3.2}
    ],
    "top_k": 3
  }'

# Analyze individual
curl -X POST http://localhost:8000/api/analyze/individual \
  -H "Content-Type: application/json" \
  -d '{
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "student_id": "SV001",
    "clo_score": 3.5,
    "top_k": 5
  }'
```

---

### 5.3. Tích hợp với Django

#### File: `myapp/views.py`

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import sys
sys.path.append('./hsmh_model')

from model_loader import ClassAnalyzer, IndividualAnalyzer

# Global analyzers
class_analyzer = ClassAnalyzer(
    model_path="hsmh_model/trained_models/class_model/class_model.pkl",
    metadata_path="hsmh_model/trained_models/class_model/metadata.pkl"
)

individual_analyzer = IndividualAnalyzer(
    model_path="hsmh_model/trained_models/individual_model/individual_model.pkl",
    metadata_path="hsmh_model/trained_models/individual_model/metadata.pkl"
)

@csrf_exempt
def analyze_class_view(request):
    """Endpoint phân tích lớp"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        subject_id = data.get('subject_id')
        lecturer_name = data.get('lecturer_name')
        students = data.get('students', [])
        top_k = data.get('top_k', 3)
        
        student_list = [s['student_id'] for s in students]
        scores = [s['score'] for s in students]
        
        result = class_analyzer.analyze(
            subject_id=subject_id,
            lecturer_name=lecturer_name,
            student_list=student_list,
            scores=scores,
            top_k=top_k
        )
        
        return JsonResponse({'success': True, 'data': result})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
def analyze_individual_view(request):
    """Endpoint phân tích cá nhân"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        subject_id = data.get('subject_id')
        lecturer_name = data.get('lecturer_name')
        student_id = data.get('student_id')
        clo_score = data.get('clo_score')
        top_k = data.get('top_k', 5)
        
        result = individual_analyzer.analyze(
            subject_id=subject_id,
            lecturer_name=lecturer_name,
            student_id=student_id,
            clo_score=clo_score,
            top_k=top_k
        )
        
        return JsonResponse({'success': True, 'data': result})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
```

#### File: `myapp/urls.py`

```python
from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze/class', views.analyze_class_view, name='analyze_class'),
    path('api/analyze/individual', views.analyze_individual_view, name='analyze_individual'),
]
```

---

### 5.4. Tích hợp với Flask

#### File: `backend/app.py`

```python
from flask import Flask, request, jsonify
import sys
sys.path.append('./hsmh_model')

from model_loader import ClassAnalyzer, IndividualAnalyzer

app = Flask(__name__)

# Load models
print("🔄 Loading models...")
class_analyzer = ClassAnalyzer(
    model_path="hsmh_model/trained_models/class_model/class_model.pkl",
    metadata_path="hsmh_model/trained_models/class_model/metadata.pkl"
)

individual_analyzer = IndividualAnalyzer(
    model_path="hsmh_model/trained_models/individual_model/individual_model.pkl",
    metadata_path="hsmh_model/trained_models/individual_model/metadata.pkl"
)
print("✅ Models loaded!")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'class_analyzer': 'loaded',
        'individual_analyzer': 'loaded'
    })

@app.route('/api/analyze/class', methods=['POST'])
def analyze_class():
    try:
        data = request.json
        
        subject_id = data.get('subject_id')
        lecturer_name = data.get('lecturer_name')
        students = data.get('students', [])
        top_k = data.get('top_k', 3)
        
        student_list = [s['student_id'] for s in students]
        scores = [s['score'] for s in students]
        
        result = class_analyzer.analyze(
            subject_id=subject_id,
            lecturer_name=lecturer_name,
            student_list=student_list,
            scores=scores,
            top_k=top_k
        )
        
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/individual', methods=['POST'])
def analyze_individual():
    try:
        data = request.json
        
        subject_id = data.get('subject_id')
        lecturer_name = data.get('lecturer_name')
        student_id = data.get('student_id')
        clo_score = data.get('clo_score')
        top_k = data.get('top_k', 5)
        
        result = individual_analyzer.analyze(
            subject_id=subject_id,
            lecturer_name=lecturer_name,
            student_id=student_id,
            clo_score=clo_score,
            top_k=top_k
        )
        
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

---

## 6. API REFERENCE

### 6.1. POST `/api/analyze/class`

**Mô tả**: Phân tích tổng quan lớp học

**Request Body**:
```json
{
  "subject_id": "string",           // Mã môn học
  "lecturer_name": "string",        // Tên/Mã giảng viên
  "students": [
    {
      "student_id": "string",       // Mã sinh viên
      "score": 5.5                  // Điểm CLO (0-6)
    }
  ],
  "top_k": 3                        // Optional, default=3
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "mode": "class",
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "statistics": {
      "total_students": 5,
      "average_score": 4.36,
      "min_score": 2.5,
      "max_score": 5.5,
      "pass_rate": 80.0
    },
    "class_general_analysis": {
      "dataset": "Chuyên cần CLO",
      "severity_level": "Medium",
      "severity_confidence": 0.85,
      "results": [
        {
          "reason": "Sinh viên thiếu động lực học tập...",
          "solution": "Tăng cường tương tác trong lớp...",
          "severity": "Medium",
          "confidence": 0.85
        }
      ]
    },
    "students_need_attention": [
      {
        "student_id": "SV003",
        "clo_score": 2.5,
        "performance_level": "Yếu"
      }
    ]
  }
}
```

---

### 6.2. POST `/api/analyze/individual`

**Mô tả**: Phân tích chi tiết 1 sinh viên

**Request Body**:
```json
{
  "subject_id": "string",
  "lecturer_name": "string",
  "student_id": "string",
  "clo_score": 3.5,                 // 0-6
  "top_k": 5                        // Optional, default=5
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "mode": "individual",
    "subject_id": "INF1383",
    "lecturer_name": "Nguyễn Văn A",
    "student_id": "SV001",
    "clo_score": 3.5,
    "performance_level": "Trung bình",
    "clo_analysis": {
      "dataset": "Chuyên cần CLO",
      "severity_level": "Medium",
      "severity_confidence": 0.82,
      "results": [
        {
          "reason": "Sinh viên nghỉ học nhiều...",
          "solution": "Tăng cường theo dõi chuyên cần...",
          "severity": "Medium",
          "confidence": 0.82
        }
      ]
    }
  }
}
```

---

### 6.3. Severity Levels

| Level    | Mô tả                                | Điểm CLO (normalized) |
|----------|--------------------------------------|-----------------------|
| Low      | Không có vấn đề nghiêm trọng         | > 0.83 (> 5.0/6)      |
| Medium   | Cần chú ý, có vấn đề nhỏ             | 0.5 - 0.83 (3-5/6)    |
| High     | Cần can thiệp, vấn đề đáng kể        | 0.33 - 0.5 (2-3/6)    |
| Critical | Khẩn cấp, cần can thiệp ngay lập tức | < 0.33 (< 2/6)        |

---

### 6.4. Performance Levels

| Level       | Điểm CLO | Mô tả               |
|-------------|----------|---------------------|
| Xuất sắc    | >= 5.5   | Rất xuất sắc        |
| Giỏi        | >= 5.0   | Tốt                 |
| Khá         | >= 4.0   | Khá                 |
| Trung bình  | >= 3.0   | Đạt chuẩn đầu ra    |
| Yếu         | >= 2.0   | Dưới chuẩn          |
| Kém         | < 2.0    | Rất kém             |

---

## 7. DEPLOYMENT

### 7.1. Deployment với Docker

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy hsmh_model
COPY hsmh_model/ ./hsmh_model/

# Copy backend
COPY backend/ ./backend/

# Expose port
EXPOSE 8000

# Run
CMD ["python", "backend/app.py"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./hsmh_model/trained_models:/app/hsmh_model/trained_models:ro
      - ./hsmh_model/dulieu:/app/hsmh_model/dulieu:ro
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

#### Build & Run:
```bash
docker-compose up --build
```

---

### 7.2. Deployment trên Server (Production)

#### Sử dụng Gunicorn + Nginx

**File: `gunicorn_config.py`**
```python
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
```

**Chạy với Gunicorn:**
```bash
gunicorn -c gunicorn_config.py backend.app:app
```

**Nginx config:**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

### 7.3. Environment Variables

Tạo file `.env`:
```bash
# Model paths
CLASS_MODEL_PATH=hsmh_model/trained_models/class_model/class_model.pkl
CLASS_METADATA_PATH=hsmh_model/trained_models/class_model/metadata.pkl
INDIVIDUAL_MODEL_PATH=hsmh_model/trained_models/individual_model/individual_model.pkl
INDIVIDUAL_METADATA_PATH=hsmh_model/trained_models/individual_model/metadata.pkl

# Server config
HOST=0.0.0.0
PORT=8000
DEBUG=False

# API config
MAX_STUDENTS_PER_REQUEST=100
DEFAULT_TOP_K=3
```

Load trong code:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class_model_path = os.getenv('CLASS_MODEL_PATH')
```

---

## 8. XỬ LÝ LỖI VÀ LOGGING

### 8.1. Logging

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use in code
logger.info("Model loaded successfully")
logger.error(f"Error analyzing class: {e}")
```

### 8.2. Error Handling

```python
from fastapi import HTTPException

try:
    result = class_analyzer.analyze(...)
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise HTTPException(status_code=500, detail="Model not initialized")
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 9. BEST PRACTICES

### 9.1. Performance Optimization

1. **Load models một lần** khi khởi động server (không load mỗi request)
2. **Cache kết quả** nếu input giống nhau
3. **Giới hạn số lượng sinh viên** mỗi request (VD: max 100)
4. **Sử dụng async/await** cho I/O operations
5. **Pagination** cho kết quả lớn

### 9.2. Security

1. **Rate limiting**: Giới hạn số request/phút
2. **Authentication**: JWT hoặc API keys
3. **Input validation**: Validate tất cả input
4. **CORS**: Cấu hình CORS phù hợp
5. **HTTPS**: Luôn dùng HTTPS trong production

### 9.3. Monitoring

1. **Health checks**: Endpoint `/health` để monitoring
2. **Metrics**: Track response time, error rate
3. **Logging**: Log tất cả requests và errors
4. **Alerts**: Setup alerts cho errors

---

## 10. FAQ

### Q1: Model load chậm?
**A**: Models được load 1 lần khi khởi động server. Nếu vẫn chậm:
- Kiểm tra dung lượng file .pkl
- Tăng RAM server
- Sử dụng model compression

### Q2: Làm sao để update model?
**A**: 
1. Train lại model với data mới
2. Lưu file .pkl mới
3. Restart server để load model mới
4. Hoặc implement hot-reload mechanism

### Q3: Có thể train model online không?
**A**: Có, implement endpoint `/api/train`:
```python
@app.post("/api/train")
def train_model_endpoint():
    # Load new data
    # Retrain model
    # Save new model
    # Reload model
    pass
```

### Q4: Xử lý nhiều request đồng thời?
**A**: 
- Dùng Gunicorn/Uvicorn với nhiều workers
- Load balancing với Nginx
- Horizontal scaling với Kubernetes

---

## 11. CONTACT & SUPPORT

- **Email**: support@example.com
- **Documentation**: https://docs.example.com
- **GitHub**: https://github.com/your-org/hsmh_model

---

## 12. CHANGELOG

### Version 1.0.0 (Current)
- ✅ Initial release
- ✅ Class analysis
- ✅ Individual analysis
- ✅ 6 datasets integration
- ✅ FastAPI/Django/Flask support

### Version 1.1.0 (Planned)
- 🔄 Online learning
- 🔄 Model compression
- 🔄 Advanced caching
- 🔄 Real-time predictions

---

**© 2024 CLO Prediction System. All rights reserved.**

