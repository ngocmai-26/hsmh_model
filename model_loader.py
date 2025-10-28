#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Loader - Load và sử dụng các trained models từ file pickle
DEMO ĐÚNG: Load model đã train sẵn, KHÔNG train lại
"""

import pickle
import os
import pandas as pd
from typing import Dict, List, Optional


class ModelLoader:
    """
    Class để load và sử dụng trained model từ file pickle
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo ModelLoader
        
        Args:
            model_path: Đường dẫn tới file model pickle. 
                       Nếu None, sẽ tự động tìm model mới nhất
        """
        self.model = None
        self.metadata = None
        self.model_path = model_path
        self.is_loaded = False
        
        # Tự động detect model path nếu không được cung cấp
        if model_path is None:
            self._auto_detect_model()
    
    def _auto_detect_model(self):
        """Tự động tìm model mới nhất"""
        # Ưu tiên tìm class_model
        class_model = "trained_models/class_model/class_model.pkl"
        individual_model = "trained_models/individual_model/individual_model.pkl"
        
        if os.path.exists(class_model):
            self.model_path = class_model
            self.metadata_path = "trained_models/class_model/metadata.pkl"
            print(f"🔍 Tự động chọn: {class_model}")
        elif os.path.exists(individual_model):
            self.model_path = individual_model
            self.metadata_path = "trained_models/individual_model/metadata.pkl"
            print(f"🔍 Tự động chọn: {individual_model}")
        else:
            print("⚠️  Không tìm thấy model nào!")
            print("   Vui lòng train model trước:")
            print("   - python train_class_model.py")
            print("   - python train_individual_model.py")
    
    def load(self):
        """
        Load model từ file pickle
        
        Returns:
            True nếu load thành công, False nếu thất bại
        """
        if self.model_path is None or not os.path.exists(self.model_path):
            print(f"❌ Không tìm thấy model: {self.model_path}")
            return False
        
        try:
            print(f"\n{'=' * 80}")
            print(f"📦 ĐANG LOAD MODEL TỪ FILE PICKLE")
            print(f"{'=' * 80}")
            print(f"📁 File: {self.model_path}")
            
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load metadata nếu có
            if hasattr(self, 'metadata_path') and os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            
            self.is_loaded = True
            
            print(f"✅ LOAD MODEL THÀNH CÔNG!")
            
            # Hiển thị metadata
            if self.metadata:
                print(f"\n📋 THÔNG TIN MODEL:")
                print(f"   - Loại: {self.metadata.get('model_type', 'N/A')}")
                print(f"   - Ngày train: {self.metadata.get('trained_date', 'N/A')}")
                print(f"   - Số datasets: {self.metadata.get('num_datasets', 'N/A')}")
                print(f"   - Tổng records: {self.metadata.get('total_records', 'N/A'):,}")
            
            # Hiển thị model summary
            if hasattr(self.model, 'get_model_summary'):
                summary = self.model.get_model_summary()
                print(f"\n📊 SUMMARY:")
                print(f"   - Datasets: {summary['total_datasets']}")
                print(f"   - Models trained: {summary['total_models']}")
                
                if summary['models']:
                    print(f"\n   Models:")
                    for key, info in summary['models'].items():
                        print(f"      • {info['description']:30} | {info['type']:20} | Acc: {info['accuracy']:.4f}")
            
            print(f"{'=' * 80}\n")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_reason_solution(self, dataset_key: str, features: List[float], top_k: int = 3) -> Optional[Dict]:
        """
        Dự đoán reasons & solutions cho một dataset
        
        Args:
            dataset_key: Tên dataset (teaching_methods, evaluation_methods, etc.)
            features: List các features (thường là [score])
            top_k: Số lượng reasons/solutions trả về
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        if not self.is_loaded or self.model is None:
            print("❌ Model chưa được load! Gọi load() trước.")
            return None
        
        try:
            return self.model.predict_reason_solution(dataset_key, features, top_k)
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {e}")
            return None


class ClassAnalyzer:
    """
    Analyzer cho chế độ lớp học
    Sử dụng model đã load từ file pickle
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo ClassAnalyzer
        
        Args:
            model_path: Đường dẫn tới model pickle. Nếu None, dùng class_model
        """
        if model_path is None:
            model_path = "trained_models/class_model/class_model.pkl"
        
        self.loader = ModelLoader(model_path)
        
        # Load model ngay khi khởi tạo
        if not self.loader.load():
            print("⚠️  ClassAnalyzer: Không load được model!")
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_list: List[str], scores: List[float],
                top_k: int = 3, display: bool = True) -> Optional[Dict]:
        """
        Phân tích cho cả lớp học
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên giảng viên
            student_list: Danh sách mã sinh viên
            scores: Danh sách điểm CLO (0-6)
            top_k: Số lượng reasons/solutions
            display: Hiển thị kết quả
            
        Returns:
            Dictionary chứa kết quả phân tích
        """
        if not self.loader.is_loaded:
            print("❌ Model chưa được load!")
            return None
        
        # Validate input
        if len(student_list) != len(scores):
            print(f"❌ Số sinh viên ({len(student_list)}) không khớp với số điểm ({len(scores)})")
            return None
        
        # Tính toán thống kê lớp
        avg_score = sum(scores) / len(scores)
        avg_score_normalized = avg_score / 6.0
        
        # Dự đoán reasons & solutions chung cho lớp dựa trên điểm trung bình
        class_analysis = self.loader.predict_reason_solution(
            'clo_attendance', 
            [avg_score_normalized], 
            top_k
        )
        
        # Tạo kết quả
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
        
        if display:
            self._display_result(result)
        
        return result
    
    def _classify_performance(self, score: float) -> str:
        """Phân loại mức độ"""
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'
    
    def _display_result(self, result: Dict):
        """Hiển thị kết quả phân tích lớp"""
        print("\n" + "=" * 80)
        print(f"📚 PHÂN TÍCH LỚP HỌC")
        print("=" * 80)
        print(f"Môn học: {result['subject_id']}")
        print(f"Giảng viên: {result['lecturer_name']}")
        
        stats = result['statistics']
        print(f"\n📊 THỐNG KÊ:")
        print(f"   - Tổng số SV: {stats['total_students']}")
        print(f"   - Điểm TB: {stats['average_score']:.2f}/6")
        print(f"   - Điểm min: {stats['min_score']:.2f}/6")
        print(f"   - Điểm max: {stats['max_score']:.2f}/6")
        print(f"   - Tỉ lệ đạt: {stats['pass_rate']:.1f}%")
        
        # Nhận xét chung
        if result['class_general_analysis']:
            analysis = result['class_general_analysis']
            print(f"\n💡 NHẬN XÉT CHUNG CHO LỚP:")
            print(f"   Mức độ: {analysis['severity_level']} (Tin cậy: {analysis['severity_confidence']:.3f})")
            
            if 'results' in analysis:
                print(f"\n🎯 CÁC VẤN ĐỀ CHUNG:")
                for i, item in enumerate(analysis['results'], 1):
                    print(f"\n{i}. Vấn đề:")
                    print(f"   {item['reason']}")
                    print(f"   → Giải pháp:")
                    print(f"   {item['solution']}")
        
        # Sinh viên cần can thiệp
        if result['students_need_attention']:
            print(f"\n⚠️  SINH VIÊN CẦN CAN THIỆP: {len(result['students_need_attention'])} sinh viên")
            for student in result['students_need_attention']:
                print(f"   • {student['student_id']}: {student['clo_score']:.2f}/6 ({student['performance_level']})")


class IndividualAnalyzer:
    """
    Analyzer cho chế độ cá nhân
    Sử dụng model đã load từ file pickle
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo IndividualAnalyzer
        
        Args:
            model_path: Đường dẫn tới model pickle. Nếu None, dùng individual_model
        """
        if model_path is None:
            model_path = "trained_models/individual_model/individual_model.pkl"
            # Fallback to class_model nếu individual_model không tồn tại
            if not os.path.exists(model_path):
                model_path = "trained_models/class_model/class_model.pkl"
        
        self.loader = ModelLoader(model_path)
        
        # Load model ngay khi khởi tạo
        if not self.loader.load():
            print("⚠️  IndividualAnalyzer: Không load được model!")
    
    def analyze(self, subject_id: str, lecturer_name: str,
                student_id: str, clo_score: float,
                top_k: int = 5, display: bool = True) -> Optional[Dict]:
        """
        Phân tích cho 1 sinh viên
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên giảng viên
            student_id: Mã sinh viên
            clo_score: Điểm CLO (0-6)
            top_k: Số lượng reasons/solutions
            display: Hiển thị kết quả
            
        Returns:
            Dictionary chứa kết quả phân tích
        """
        if not self.loader.is_loaded:
            print("❌ Model chưa được load!")
            return None
        
        # Normalize score
        clo_score_normalized = clo_score / 6.0
        
        # Dự đoán reasons & solutions cho sinh viên
        clo_analysis = self.loader.predict_reason_solution(
            'clo_attendance',
            [clo_score_normalized],
            top_k
        )
        
        # Tạo kết quả
        result = {
            'mode': 'individual',
            'subject_id': subject_id,
            'lecturer_name': lecturer_name,
            'student_id': student_id,
            'clo_score': clo_score,
            'performance_level': self._classify_performance(clo_score),
            'clo_analysis': clo_analysis
        }
        
        if display:
            self._display_result(result)
        
        return result
    
    def _classify_performance(self, score: float) -> str:
        """Phân loại mức độ"""
        if score >= 5.5: return 'Xuất sắc'
        elif score >= 5.0: return 'Giỏi'
        elif score >= 4.0: return 'Khá'
        elif score >= 3.0: return 'Trung bình'
        elif score >= 2.0: return 'Yếu'
        else: return 'Kém'
    
    def _display_result(self, result: Dict):
        """Hiển thị kết quả phân tích cá nhân"""
        print("\n" + "=" * 80)
        print(f"👤 PHÂN TÍCH CÁ NHÂN")
        print("=" * 80)
        print(f"Môn học: {result['subject_id']}")
        print(f"Giảng viên: {result['lecturer_name']}")
        print(f"Sinh viên: {result['student_id']}")
        print(f"Điểm CLO: {result['clo_score']:.2f}/6")
        print(f"Xếp loại: {result['performance_level']}")
        
        # Phân tích CLO
        if result['clo_analysis']:
            analysis = result['clo_analysis']
            print(f"\n🎯 PHÂN TÍCH CLO:")
            print(f"   Mức độ: {analysis['severity_level']} (Tin cậy: {analysis['severity_confidence']:.3f})")
            
            if 'results' in analysis:
                print(f"\n💡 REASONS & SOLUTIONS:")
                for i, item in enumerate(analysis['results'], 1):
                    print(f"\n{i}. Nguyên nhân:")
                    print(f"   {item['reason']}")
                    print(f"   → Giải pháp:")
                    print(f"   {item['solution']}")


class PredictionTools:
    """
    Các công cụ dự đoán sử dụng model đã load
    """
    
    def __init__(self, model_path=None):
        """
        Khởi tạo PredictionTools
        
        Args:
            model_path: Đường dẫn tới model pickle
        """
        self.loader = ModelLoader(model_path)
        
        # Load model ngay
        if not self.loader.load():
            print("⚠️  PredictionTools: Không load được model!")
    
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


# ============================================================================
# DEMO USAGE
# ============================================================================

def demo_load_model():
    """Demo load model từ file pickle"""
    print("\n" + "🎯" * 40)
    print("DEMO 1: LOAD MODEL TỪ FILE PICKLE")
    print("🎯" * 40)
    
    # Tạo loader
    loader = ModelLoader()
    
    # Load model
    success = loader.load()
    
    if success:
        print("\n✅ Model đã được load thành công!")
        print("   Bây giờ có thể sử dụng model để dự đoán")
    else:
        print("\n❌ Không thể load model!")


def demo_class_analysis():
    """Demo phân tích lớp với model đã load"""
    print("\n" + "🎯" * 40)
    print("DEMO 2: PHÂN TÍCH LỚP HỌC (Sử dụng model đã load)")
    print("🎯" * 40)
    
    # Khởi tạo analyzer (sẽ tự động load model)
    analyzer = ClassAnalyzer()
    
    # Dữ liệu mẫu
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_list=["SV001", "SV002", "SV003", "SV004", "SV005"],
        scores=[5.5, 4.8, 3.2, 2.5, 4.5],
        top_k=3,
        display=True
    )


def demo_individual_analysis():
    """Demo phân tích cá nhân với model đã load"""
    print("\n" + "🎯" * 40)
    print("DEMO 3: PHÂN TÍCH CÁ NHÂN (Sử dụng model đã load)")
    print("🎯" * 40)
    
    # Khởi tạo analyzer (sẽ tự động load model)
    analyzer = IndividualAnalyzer()
    
    # Phân tích sinh viên
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_id="SV001",
        clo_score=3.5,
        top_k=5,
        display=True
    )


def demo_prediction_tools():
    """Demo các công cụ dự đoán với model đã load"""
    print("\n" + "🎯" * 40)
    print("DEMO 4: PREDICTION TOOLS (Sử dụng model đã load)")
    print("🎯" * 40)
    
    # Khởi tạo tools (sẽ tự động load model)
    tools = PredictionTools()
    
    # Dự đoán Teaching Methods
    print("\n1️⃣ Phương pháp giảng dạy (PPGD):")
    tm = tools.predict_teaching_methods(0.6, top_k=2)
    if tm:
        print(f"   Mức độ: {tm['severity_level']}")
        for i, item in enumerate(tm['results'], 1):
            print(f"   {i}. {item['reason'][:80]}...")
    
    # Dự đoán CLO
    print("\n2️⃣ CLO Attendance:")
    clo = tools.predict_clo_attendance(0.5, top_k=2)
    if clo:
        print(f"   Mức độ: {clo['severity_level']}")
        for i, item in enumerate(clo['results'], 1):
            print(f"   {i}. {item['reason'][:80]}...")


def main():
    """Main demo"""
    print("=" * 80)
    print("🔥 MODEL LOADER - LOAD VÀ SỬ DỤNG TRAINED MODEL")
    print("=" * 80)
    print("\n📚 Hệ thống này LOAD model từ file pickle đã train sẵn")
    print("   KHÔNG train lại model mỗi lần chạy")
    print("\n🎯 Các chức năng:")
    print("   1. Load model từ file pickle")
    print("   2. Phân tích lớp học với model đã load")
    print("   3. Phân tích cá nhân với model đã load")
    print("   4. Prediction tools với model đã load")
    
    try:
        # Chạy demos
        demo_load_model()
        demo_class_analysis()
        demo_individual_analysis()
        demo_prediction_tools()
        
        print("\n" + "=" * 80)
        print("✅ TẤT CẢ DEMO HOÀN THÀNH!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
