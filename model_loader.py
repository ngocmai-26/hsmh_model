import pickle
import os
from typing import Dict, List, Optional, Any
import logging

class ModelLoader:
    model_path: str = ""
    metadata_path: str = ""
    model: Any = None
    metadata: Any = None
    is_loaded: bool = False
    
    def __init__(self, model_path, metadata_path = None):
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
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            if self.metadata_path:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
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
        if not self.is_loaded or self.model is None:
            logging.getLogger().error("Model is not loaded! Call load() before predicting.")
            raise Exception("Model is not loaded!")
        
        try:
            return self.model.predict_reason_solution(dataset_key, features, top_k)
        except Exception as e:
            logging.getLogger().error(f"Error predicting reason and solution: {e}")
            raise Exception(f"Error predicting reason and solution: {e}")


class ClassAnalyzer:
    __loader: ModelLoader = None
    
    def __init__(self, model_path, metadata_path = None):
        self.__loader = ModelLoader(model_path, metadata_path)
        self.__loader.load()
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_list: List[str], scores: List[float],
                top_k: int = 3) -> Optional[Dict]:
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
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'

class IndividualAnalyzer:
    __loader: ModelLoader = None
    
    def __init__(self, model_path, metadata_path = None):
        self.__loader = ModelLoader(model_path, metadata_path)
        self.__loader.load()
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_id: str, clo_score: float,
                top_k: int = 5) -> Optional[Dict]:
        if not self.__loader.is_loaded:
            logging.getLogger().error("Model is not loaded! Call load() before analyzing.")
            raise Exception("Model is not loaded!")
        
        if not self.loader.is_loaded:
            logging.getLogger().error("Model is not loaded! Call load() before analyzing.")
            raise Exception("Model is not loaded!")
        
        clo_score_normalized = clo_score / 6.0
        
        clo_analysis = self.__loader.predict_reason_solution(
            'clo_attendance',
            [clo_score_normalized],
            top_k
        )
        
        result = {
            'mode': 'individual',
            'subject_id': subject_id,
            'lecturer_name': lecturer_name,
            'student_id': student_id,
            'clo_score': clo_score,
            'performance_level': self._classify_performance(clo_score),
            'clo_analysis': clo_analysis
        }
        
        return result
    
    def _classify_performance(self, score: float) -> str:
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'

class PredictionTools:
    __loader: ModelLoader = None
    
    def __init__(self, model_path, metadata_path = None):
        self.__loader = ModelLoader(model_path, metadata_path)
        self.__loader.load()
    
    def predict_teaching_methods(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """Dự đoán cho Phương pháp giảng dạy"""
        return self.loader.predict_reason_solution('teaching_methods', [score], top_k)
    
    def predict_evaluation_methods(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """Dự đoán cho Phương pháp đánh giá"""
        return self.loader.predict_reason_solution('evaluation_methods', [score], top_k)
    
    def predict_student_conduct(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """Dự đoán cho Điểm rèn luyện"""
        return self.loader.predict_reason_solution('student_conduct', [score], top_k)
    
    def predict_academic_midterm(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """Dự đoán cho Điểm giữa kỳ"""
        return self.loader.predict_reason_solution('academic_midterm', [score], top_k)
    
    def predict_clo_attendance(self, score: float, top_k: int = 3) -> Optional[Dict]:
        """Dự đoán cho CLO Attendance"""
        return self.loader.predict_reason_solution('clo_attendance', [score], top_k)