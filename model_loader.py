import pickle
import os
from typing import Dict, List, Optional, Any
import logging

# Import analyze_individual - tương thích với cả local và package
try:
    # Thử import từ package (khi đã được cài đặt)
    from hsmh_model.unified_integration import analyze_individual
except ImportError:
    try:
        # Thử import từ local (khi chạy từ source)
        from unified_integration import analyze_individual
    except ImportError:
        # Nếu không tìm thấy, raise error rõ ràng
        analyze_individual = None
        logging.getLogger().warning(
            "Cannot import analyze_individual. "
            "Please ensure unified_integration.py is available in the package."
        )

# Đường dẫn gốc của package
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom Unpickler để map module path từ 'model' sang 'hsmh_model.model'
class ModelUnpickler(pickle.Unpickler):
    """Custom unpickler để map module path khi load model từ package"""
    def find_class(self, module, name):
        # Map 'model.*' -> 'hsmh_model.model.*'
        if module.startswith('model.'):
            module = module.replace('model.', 'hsmh_model.model.', 1)
        elif module == 'model':
            module = 'hsmh_model.model'
        return super().find_class(module, name)

# Hard code đường dẫn model (sẽ được build kèm package)
_CLASS_MODEL_PATH = os.path.join(_BASE_DIR, "trained_models", "class_model", "class_model.pkl")
_CLASS_METADATA_PATH = os.path.join(_BASE_DIR, "trained_models", "class_model", "metadata.pkl")
_INDIVIDUAL_MODEL_PATH = os.path.join(_BASE_DIR, "trained_models", "individual_model", "individual_model.pkl")
_INDIVIDUAL_METADATA_PATH = os.path.join(_BASE_DIR, "trained_models", "individual_model", "metadata.pkl")

class ModelLoader:
    """
    Class để load và sử dụng model đã huấn luyện từ file pickle
    
    Model paths được hard code vì sẽ được build kèm trong package.
    
    Attributes:
        model_path: Đường dẫn đến file model (hard coded)
        metadata_path: Đường dẫn đến file metadata (hard coded)
        model: Object model đã load
        metadata: Object metadata đã load
        is_loaded: Trạng thái load của model
    
    Ví dụ:
        loader = ModelLoader()
        loader.load()
        result = loader.predict_reason_solution('clo_attendance', [0.6], top_k=3)
    """
    
    model_path: str = ""
    metadata_path: str = ""
    model: Any = None
    metadata: Any = None
    is_loaded: bool = False
    
    def __init__(self, model_path: str, metadata_path: str):
        """
        Khởi tạo ModelLoader với đường dẫn hard coded
        
        Args:
            model_path: Đường dẫn đến file model (hard coded từ package)
            metadata_path: Đường dẫn đến file metadata (hard coded từ package)
        """
        self.model = None
        self.metadata = None
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.is_loaded = False
    
    def load(self):
        if self.model_path is None or not os.path.exists(self.model_path):
            logging.getLogger().error(f"Model path is not valid: {self.model_path}")
            raise Exception(f"Model path is not valid: {self.model_path}")
        
        if self.metadata_path and not os.path.exists(self.metadata_path):
            logging.getLogger().error(f"Metadata path is not valid: {self.metadata_path}")
            raise Exception(f"Metadata path is not valid: {self.metadata_path}")
        
        try:
            # Sử dụng custom unpickler để map module path
            with open(self.model_path, 'rb') as f:
                unpickler = ModelUnpickler(f)
                self.model = unpickler.load()
            
            if self.metadata_path:
                with open(self.metadata_path, 'rb') as f:
                    unpickler = ModelUnpickler(f)
                    self.metadata = unpickler.load()
            
            self.is_loaded = True
            
            logging.getLogger().info(f"Model loaded successfully from {self.model_path}")
            
            if self.metadata:
                logging.getLogger().info(f"Metadata loaded successfully from {self.metadata_path}")

            if hasattr(self.model, 'get_model_summary'):
                summary = self.model.get_model_summary()
                logging.getLogger().info(f"Model summary: {summary}")
            
        except Exception as e:
            logging.getLogger().error(f"Error loading model: {e}")
            raise Exception(f"Error loading model: {e}")
    
    def predict_reason_solution(self, dataset_key: str, features: List[float], top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán reasons và solutions cho một dataset cụ thể
        
        Args:
            dataset_key: Tên dataset (VD: 'clo_attendance', 'teaching_methods', ...)
            features: Danh sách features (thường là [score_normalized])
            top_k: Số lượng reasons/solutions trả về
            
        Returns:
            Dictionary chứa kết quả dự đoán
            
        Raises:
            Exception: Nếu model chưa được load hoặc có lỗi trong quá trình dự đoán
        """
        if not self.is_loaded or self.model is None:
            logging.getLogger().error("Model is not loaded! Call load() before predicting.")
            raise Exception("Model is not loaded!")
        
        try:
            return self.model.predict_reason_solution(dataset_key, features, top_k)
        except Exception as e:
            logging.getLogger().error(f"Error predicting reason and solution: {e}")
            raise Exception(f"Error predicting reason and solution: {e}")
    
    def get_model_info(self) -> Optional[Dict]:
        """
        Lấy thông tin về model đã load
        
        Returns:
            Dictionary chứa thông tin về model hoặc None nếu model chưa load
        """
        if not self.is_loaded or self.model is None:
            logging.getLogger().warning("Model is not loaded!")
            return None
        
        info = {
            'model_path': self.model_path,
            'metadata_path': self.metadata_path,
            'is_loaded': self.is_loaded,
            'has_metadata': self.metadata is not None
        }
        
        # Thêm thông tin từ model nếu có method get_model_summary
        if hasattr(self.model, 'get_model_summary'):
            try:
                info['model_summary'] = self.model.get_model_summary()
            except Exception as e:
                logging.getLogger().warning(f"Could not get model summary: {e}")
        
        return info
    
    def is_model_loaded(self) -> bool:
        """
        Kiểm tra xem model đã được load hay chưa
        
        Returns:
            True nếu model đã load, False nếu chưa
        """
        return self.is_loaded and self.model is not None


class ClassAnalyzer:
    """
    Class phân tích lớp học
    
    Model paths được hard code trong package, không cần truyền tham số.
    
    Hỗ trợ 2 cách sử dụng:
    1. analyze() - Truyền student_list và scores riêng (legacy)
    2. analyze_v2() - Truyền students_data dạng list of dict (khuyến nghị cho Backend)
    
    Ví dụ:
        analyzer = ClassAnalyzer()  # Không cần truyền path
        result = analyzer.analyze_v2(...)
    """
    
    __loader: ModelLoader = None
    
    def __init__(self):
        """Khởi tạo ClassAnalyzer với model path đã hard code trong package"""
        self.__loader = ModelLoader(_CLASS_MODEL_PATH, _CLASS_METADATA_PATH)
        self.__loader.load()
    
    def analyze_v2(self, subject_id: str, lecturer_name: str,
                   students_data: List[Dict], top_k: int = 3, 
                   display: bool = False) -> Optional[Dict]:
        """
        Phân tích lớp học - Nhận data dạng list of dict (KHUYẾN NGHỊ)
        
        Args:
            subject_id: Mã môn học (VD: "INF1383")
            lecturer_name: Tên giảng viên
            students_data: List of dict chứa thông tin sinh viên
                [
                    {"mssv": "SV001", "ho_ten": "Nguyễn Văn A", "diem_clo": 5.5},
                    {"mssv": "SV002", "ho_ten": "Trần Văn B", "diem_clo": 4.8},
                    ...
                ]
                Các key có thể dùng:
                - mssv / student_id / MSSV / Student_ID
                - ho_ten / hoten / name / full_name / HoTen
                - diem_clo / clo_score / score / DiemCLO
            top_k: Số lượng reasons/solutions trả về
            display: Có hiển thị kết quả ra console hay không
            
        Returns:
            Dictionary chứa kết quả phân tích lớp
        """
        # Validate input
        if not students_data or not isinstance(students_data, list):
            logging.getLogger().error("students_data phải là một list không rỗng!")
            raise Exception("students_data phải là một list không rỗng!")
        
        if len(students_data) == 0:
            logging.getLogger().error("Danh sách sinh viên trống!")
            raise Exception("Danh sách sinh viên trống!")
        
        # Extract và chuẩn hóa dữ liệu
        student_list = []
        scores = []
        student_names = {}  # Map MSSV -> Tên
        
        # Các key có thể có trong dict
        mssv_keys = ['mssv', 'student_id', 'MSSV', 'Student_ID', 'ma_sv', 'MaSV']
        name_keys = ['ho_ten', 'hoten', 'name', 'full_name', 'HoTen', 'FullName', 'ten', 'Ten']
        score_keys = ['diem_clo', 'clo_score', 'score', 'DiemCLO', 'CLO_Score', 'diem', 'Diem']
        
        for idx, student in enumerate(students_data):
            if not isinstance(student, dict):
                logging.getLogger().warning(f"Bỏ qua sinh viên thứ {idx+1}: không phải dict")
                continue
            
            # Tìm MSSV
            mssv = None
            for key in mssv_keys:
                if key in student:
                    mssv = str(student[key]).strip()
                    break
            
            if not mssv:
                logging.getLogger().warning(f"Bỏ qua sinh viên thứ {idx+1}: không tìm thấy MSSV")
                continue
            
            # Tìm điểm CLO
            score = None
            for key in score_keys:
                if key in student:
                    try:
                        score = float(student[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            if score is None:
                logging.getLogger().warning(f"Bỏ qua sinh viên {mssv}: không tìm thấy điểm CLO")
                continue
            
            # Validate điểm
            if not (0 <= score <= 6):
                logging.getLogger().warning(f"Bỏ qua sinh viên {mssv}: điểm không hợp lệ ({score})")
                continue
            
            # Tìm tên (optional)
            name = None
            for key in name_keys:
                if key in student:
                    name = str(student[key]).strip()
                    break
            
            # Thêm vào list
            student_list.append(mssv)
            scores.append(score)
            if name:
                student_names[mssv] = name
        
        if len(student_list) == 0:
            logging.getLogger().error("Không có sinh viên hợp lệ nào sau khi xử lý!")
            raise Exception("Không có sinh viên hợp lệ nào sau khi xử lý!")
        
        logging.getLogger().info(f"Đã xử lý {len(student_list)}/{len(students_data)} sinh viên")
        
        # Gọi hàm analyze cũ với data đã tách
        result = self.analyze(subject_id, lecturer_name, student_list, scores, top_k, display=False)
        
        if result is None:
            return None
        
        # Thêm tên sinh viên vào kết quả
        if student_names and 'students_need_attention' in result:
            for student in result['students_need_attention']:
                sid = student['student_id']
                if sid in student_names:
                    student['student_name'] = student_names[sid]
        
        # Thêm thông tin tổng số sinh viên đã xử lý
        result['total_students_processed'] = len(student_list)
        result['total_students_input'] = len(students_data)
        
        # Hiển thị kết quả nếu display=True
        if display:
            self._display_class_analysis(result)
        
        return result
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_list: List[str], scores: List[float],
                top_k: int = 3) -> Optional[Dict]:
        """
        Phân tích lớp học
        
        Args:
            subject_id: Mã môn học (VD: "INF1383")
            lecturer_name: Tên giảng viên
            student_list: Danh sách mã sinh viên
            scores: Danh sách điểm CLO (0-6)
            top_k: Số lượng reasons/solutions trả về
            display: Có hiển thị kết quả ra console hay không
            
        Returns:
            Dictionary chứa kết quả phân tích lớp
        """
        if not self.__loader.is_loaded:
            logging.getLogger().error("Model is not loaded! Call load() before analyzing.")
            raise Exception("Model is not loaded!")
        
        if len(student_list) != len(scores):
            logging.getLogger().error(f"Number of students ({len(student_list)}) does not match the number of scores ({len(scores)})")
            raise Exception(f"Number of students ({len(student_list)}) does not match the number of scores ({len(scores)})")
        
        avg_score = sum(scores) / len(scores)
        avg_score_normalized = avg_score / 6.0
        
        class_analysis = self.__loader.predict_reason_solution(
            'clo_attendance', 
            [avg_score_normalized], 
            top_k
        )
        
        result = {
            'mode': 'class',
            'subject_id': subject_id,
            'lecturer_name': lecturer_name,
            'statistics': {
                'total_students': len(student_list),
                'average_score': avg_score,
                'min_score': min(scores),
                'max_score': max(scores),
                'pass_rate': sum(1 for s in scores if s >= 3.0) / len(scores) * 100
            },
            'class_general_analysis': class_analysis,
            'students_need_attention': [
                {'student_id': sid, 'clo_score': score, 'performance_level': self._classify_performance(score)}
                for sid, score in zip(student_list, scores) if score < 3.0
            ]
        }
        
        return result
    
    def _classify_performance(self, score: float) -> str:
        """Phân loại mức độ học lực"""
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'

class IndividualAnalyzer:
    """
    Class phân tích cá nhân sinh viên
    
    Model paths được hard code trong package, không cần truyền tham số.
    
    Ví dụ:
        analyzer = IndividualAnalyzer()  # Không cần truyền path
        result = analyzer.analyze(...)
    """
    
    __loader: ModelLoader = None
    
    def __init__(self):
        """Khởi tạo IndividualAnalyzer với model path đã hard code trong package"""
        self.__loader = ModelLoader(_INDIVIDUAL_MODEL_PATH, _INDIVIDUAL_METADATA_PATH)
        self.__loader.load()
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_id: str, top_k: int = 5) -> Optional[Dict]:
        """
        Phân tích cá nhân sinh viên
        
        Args:
            subject_id: Mã môn học (VD: "INF1383")
            lecturer_name: Tên giảng viên (được sử dụng như lecturer_id)
            student_id: Mã sinh viên
            top_k: Số lượng reasons/solutions trả về
            
        Returns:
            Dictionary chứa kết quả phân tích cá nhân
        """
        # Sử dụng hàm analyze_individual từ unified_integration
        if analyze_individual is None:
            logging.getLogger().error("analyze_individual is not available!")
            raise ImportError(
                "Cannot import analyze_individual. "
                "Please ensure unified_integration.py is available in the package."
            )
        
        # Lưu ý: lecturer_name được sử dụng như lecturer_id
        result = analyze_individual(
            subject_id=subject_id,
            lecturer_id=lecturer_name,  # Sử dụng lecturer_name như lecturer_id
            student_id=student_id,
            top_k=top_k
        )
        
        return result
   
    def _classify_performance(self, score: float) -> str:
        """Phân loại mức độ học lực"""
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'

class PredictionTools:
    """
    Công cụ dự đoán reasons & solutions cho các dataset khác nhau
    
    Model paths được hard code trong package, không cần truyền tham số.
    
    Hỗ trợ dự đoán cho:
    - Teaching Methods (Phương pháp giảng dạy)
    - Evaluation Methods (Phương pháp đánh giá)
    - Student Conduct (Điểm rèn luyện)
    - Academic Midterm (Điểm giữa kỳ)
    - CLO Attendance (Chuyên cần CLO)
    
    Ví dụ:
        tools = PredictionTools()  # Không cần truyền path
        result = tools.predict_teaching_methods(0.6, top_k=3)
    """
    
    __loader: ModelLoader = None
    
    def __init__(self):
        """Khởi tạo PredictionTools với model path đã hard code trong package"""
        self.__loader = ModelLoader(_INDIVIDUAL_MODEL_PATH, _INDIVIDUAL_METADATA_PATH)
        self.__loader.load()
    
    def predict_teaching_methods(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán nguyên nhân và giải pháp cho Phương pháp giảng dạy (PPGD)
        
        Args:
            score: Điểm đánh giá PPGD (normalized 0-1)
            top_k: Số lượng reasons/solutions trả về (default: 3)
            
        Returns:
            Dictionary chứa kết quả dự đoán hoặc None nếu lỗi
        """
        return self.__loader.predict_reason_solution('teaching_methods', [score], top_k)
    
    def predict_evaluation_methods(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán nguyên nhân và giải pháp cho Phương pháp đánh giá (PPDG)
        
        Args:
            score: Điểm đánh giá PPDG (normalized 0-1)
            top_k: Số lượng reasons/solutions trả về (default: 3)
            
        Returns:
            Dictionary chứa kết quả dự đoán hoặc None nếu lỗi
        """
        return self.__loader.predict_reason_solution('evaluation_methods', [score], top_k)
    
    def predict_student_conduct(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán nguyên nhân và giải pháp cho Điểm rèn luyện
        
        Args:
            score: Điểm rèn luyện (normalized 0-1)
            top_k: Số lượng reasons/solutions trả về (default: 3)
            
        Returns:
            Dictionary chứa kết quả dự đoán hoặc None nếu lỗi
        """
        return self.__loader.predict_reason_solution('student_conduct', [score], top_k)
    
    def predict_academic_midterm(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán nguyên nhân và giải pháp cho Điểm giữa kỳ
        
        Args:
            score: Điểm giữa kỳ (normalized 0-1)
            top_k: Số lượng reasons/solutions trả về (default: 3)
            
        Returns:
            Dictionary chứa kết quả dự đoán hoặc None nếu lỗi
        """
        return self.__loader.predict_reason_solution('academic_midterm', [score], top_k)
    
    def predict_clo_attendance(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán nguyên nhân và giải pháp cho CLO Attendance (Chuyên cần CLO)
        
        Args:
            score: Điểm CLO attendance (normalized 0-1)
            top_k: Số lượng reasons/solutions trả về (default: 3)
            
        Returns:
            Dictionary chứa kết quả dự đoán hoặc None nếu lỗi
        """
        return self.__loader.predict_reason_solution('clo_attendance', [score], top_k)
    
    def predict_comprehensive(self, scores: Dict[str, float], top_k: int = 3) -> Dict[str, Optional[Dict]]:
        """
        Dự đoán toàn diện cho nhiều dataset cùng lúc
        
        Args:
            scores: Dictionary chứa các điểm số
                {
                    'teaching_methods': 0.6,
                    'evaluation_methods': 0.7,
                    'student_conduct': 0.5,
                    'academic_midterm': 0.65,
                    'clo_attendance': 0.55
                }
            top_k: Số lượng reasons/solutions trả về cho mỗi dataset
            
        Returns:
            Dictionary chứa kết quả dự đoán cho từng dataset
        """
        results = {}
        
        if 'teaching_methods' in scores:
            results['teaching_methods'] = self.predict_teaching_methods(scores['teaching_methods'], top_k)
        
        if 'evaluation_methods' in scores:
            results['evaluation_methods'] = self.predict_evaluation_methods(scores['evaluation_methods'], top_k)
        
        if 'student_conduct' in scores:
            results['student_conduct'] = self.predict_student_conduct(scores['student_conduct'], top_k)
        
        if 'academic_midterm' in scores:
            results['academic_midterm'] = self.predict_academic_midterm(scores['academic_midterm'], top_k)
        
        if 'clo_attendance' in scores:
            results['clo_attendance'] = self.predict_clo_attendance(scores['clo_attendance'], top_k)
        
        return results