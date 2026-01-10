#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Input Handler - Xử lý đầu vào cho cả chế độ lớp và cá nhân
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional


class UnifiedInputHandler:
    """Xử lý đầu vào cho cả chế độ lớp và cá nhân"""
    
    def __init__(self):
        """Khởi tạo handler"""
        self.mode = None  # 'class' hoặc 'individual'
        
    def validate_class_input(self, subject_id: str, lecturer_name: str, 
                            student_list: List[str], scores: List[float]) -> bool:
        """
        Validate đầu vào cho chế độ lớp
        
        Args:
            subject_id: Mã môn học (VD: INF1383)
            lecturer_name: Tên giảng viên
            student_list: Danh sách mã sinh viên
            scores: Danh sách điểm CLO (scale 0-6)
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        if not subject_id or not lecturer_name:
            print("❌ Mã môn học và tên giảng viên không được rỗng!")
            return False
        
        if not student_list or not scores:
            print("❌ Danh sách sinh viên và điểm không được rỗng!")
            return False
        
        if len(student_list) != len(scores):
            print(f"❌ Số lượng sinh viên ({len(student_list)}) và số điểm ({len(scores)}) không khớp!")
            return False
        
        # Kiểm tra điểm: user input là điểm CLO (có thể là bất kỳ giá trị nào) hoặc đã normalized (0-1)
        for i, score in enumerate(scores):
            if score < 0:
                print(f"❌ Điểm CLO của sinh viên {student_list[i]} không hợp lệ: {score} (phải >= 0)")
                return False
        
        return True
    
    def validate_individual_input(self, subject_id: str, lecturer_name: str,
                                 student_id: str, clo_score: float) -> bool:
        """
        Validate đầu vào cho chế độ cá nhân
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên giảng viên
            student_id: Mã sinh viên
            clo_score: Điểm CLO (scale 0-6)
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        if not subject_id or not lecturer_name or not student_id:
            print("❌ Mã môn học, tên giảng viên và mã sinh viên không được rỗng!")
            return False
        
        # Chấp nhận điểm CLO (có thể là bất kỳ giá trị nào >= 0) hoặc đã normalized (0-1)
        if clo_score < 0:
            print(f"❌ Điểm CLO không hợp lệ: {clo_score} (phải >= 0)")
            return False
        
        return True
    
    def _normalize_score(self, score: float) -> tuple[float, float]:
        """
        Chuẩn hóa điểm về khoảng 0-1 để input vào model
        - User input là điểm CLO (không nhất thiết 0-6, có thể là bất kỳ giá trị nào)
        - Nếu điểm đã normalized (0-1): giữ nguyên để input model
        - Nếu điểm chưa normalized: chia cho 6.0 để normalize về 0-1 (vì model được train với điểm đã chia 6)
        
        Returns:
            (score_normalized_0_1, score_original): điểm normalized để input model và điểm CLO gốc để trả về
        """
        if score is None:
            return None, None
        
        # Nếu điểm đã được chuẩn hóa (0-1), giữ nguyên để input model
        if 0 <= score <= 1:
            # Trả về điểm gốc (đã normalized, không cần chuyển đổi)
            return score, score
        
        # Nếu điểm chưa normalized: chia cho 6.0 để normalize về 0-1
        # (vì model được train với điểm đã chia 6)
        score_normalized = score / 6.0
        return score_normalized, score
    
    def prepare_class_data(self, subject_id: str, lecturer_name: str,
                          student_list: List[str], scores: List[float]) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho chế độ lớp
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên giảng viên
            student_list: Danh sách mã sinh viên
            scores: Danh sách điểm CLO (0-6 hoặc đã normalized 0-1)
            
        Returns:
            DataFrame chứa dữ liệu lớp
        """
        if not self.validate_class_input(subject_id, lecturer_name, student_list, scores):
            return None
        
        # Tự động phát hiện và chuẩn hóa điểm
        normalized_scores = []
        scores_original = []
        for score in scores:
            norm_score, score_orig = self._normalize_score(score)
            if norm_score is not None:
                normalized_scores.append(norm_score)
                scores_original.append(score_orig)
            else:
                # Fallback: chia cho 6 để normalize
                normalized_scores.append(score / 6.0)
                scores_original.append(score)
        
        df = pd.DataFrame({
            'subject_id': [subject_id] * len(student_list),
            'lecturer_name': [lecturer_name] * len(student_list),
            'student_id': student_list,
            'clo_score': scores_original,  # Điểm CLO gốc để trả về
            'clo_score_normalized': normalized_scores  # Điểm normalized để input model
        })
        
        # Phân loại mức độ (dựa trên điểm hệ 6)
        df['performance_level'] = df['clo_score'].apply(self._classify_performance)
        
        self.mode = 'class'
        return df
    
    def prepare_individual_data(self, subject_id: str, lecturer_name: str,
                               student_id: str, clo_score: float) -> Dict:
        """
        Chuẩn bị dữ liệu cho chế độ cá nhân
        
        Args:
            subject_id: Mã môn học
            lecturer_name: Tên giảng viên
            student_id: Mã sinh viên
            clo_score: Điểm CLO (0-6 hoặc đã normalized 0-1)
            
        Returns:
            Dictionary chứa dữ liệu sinh viên
        """
        if not self.validate_individual_input(subject_id, lecturer_name, student_id, clo_score):
            return None
        
        # Tự động phát hiện và chuẩn hóa điểm
        clo_score_normalized, clo_score_original = self._normalize_score(clo_score)
        if clo_score_normalized is None:
            # Fallback: chia cho 6 để normalize
            clo_score_normalized = clo_score / 6.0
            clo_score_original = clo_score
        
        data = {
            'subject_id': subject_id,
            'lecturer_name': lecturer_name,
            'student_id': student_id,
            'clo_score': clo_score_original,  # Điểm CLO gốc để trả về
            'clo_score_normalized': clo_score_normalized,  # Điểm normalized để input model
            'performance_level': self._classify_performance(clo_score_original)
        }
        
        self.mode = 'individual'
        return data
    
    def _classify_performance(self, clo_score: float) -> str:
        """
        Phân loại mức độ thành tích dựa trên điểm CLO
        
        Args:
            clo_score: Điểm CLO (0-6)
            
        Returns:
            Mức độ: 'Xuất sắc', 'Giỏi', 'Khá', 'Trung bình', 'Yếu', 'Kém'
        """
        if clo_score >= 5.5:
            return 'Xuất sắc'
        elif clo_score >= 5.0:
            return 'Giỏi'
        elif clo_score >= 4.0:
            return 'Khá'
        elif clo_score >= 3.0:
            return 'Trung bình'
        elif clo_score >= 2.0:
            return 'Yếu'
        else:
            return 'Kém'
    
    def get_class_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Tính toán thống kê cho lớp
        
        Args:
            df: DataFrame chứa dữ liệu lớp
            
        Returns:
            Dictionary chứa thống kê
        """
        if df is None or df.empty:
            return None
        
        stats = {
            'total_students': len(df),
            'average_score': df['clo_score'].mean(),
            'median_score': df['clo_score'].median(),
            'std_score': df['clo_score'].std(),
            'min_score': df['clo_score'].min(),
            'max_score': df['clo_score'].max(),
            'performance_distribution': df['performance_level'].value_counts().to_dict(),
            'pass_rate': (df['clo_score'] >= 3.0).sum() / len(df) * 100,
            'excellent_rate': (df['clo_score'] >= 5.0).sum() / len(df) * 100
        }
        
        return stats
    
    def display_class_statistics(self, stats: Dict):
        """
        Hiển thị thống kê lớp
        
        Args:
            stats: Dictionary chứa thống kê
        """
        if not stats:
            print("❌ Không có dữ liệu thống kê!")
            return
        
        print("\n" + "=" * 80)
        print("📊 THỐNG KÊ LỚP HỌC")
        print("=" * 80)
        print(f"Tổng số sinh viên: {stats['total_students']}")
        print(f"\n📈 Điểm số:")
        print(f"  - Trung bình: {stats['average_score']:.2f}/6")
        print(f"  - Trung vị:   {stats['median_score']:.2f}/6")
        print(f"  - Độ lệch chuẩn: {stats['std_score']:.2f}")
        print(f"  - Điểm thấp nhất: {stats['min_score']:.2f}/6")
        print(f"  - Điểm cao nhất:  {stats['max_score']:.2f}/6")
        
        print(f"\n📊 Phân bố mức độ:")
        for level, count in sorted(stats['performance_distribution'].items()):
            percentage = (count / stats['total_students']) * 100
            print(f"  - {level:15}: {count:3} sinh viên ({percentage:.1f}%)")
        
        print(f"\n✅ Tỷ lệ:")
        print(f"  - Đạt (≥3.0):      {stats['pass_rate']:.1f}%")
        print(f"  - Giỏi trở lên (≥5.0): {stats['excellent_rate']:.1f}%")
    
    def get_students_by_performance(self, df: pd.DataFrame, level: str) -> List[str]:
        """
        Lấy danh sách sinh viên theo mức độ
        
        Args:
            df: DataFrame chứa dữ liệu lớp
            level: Mức độ ('Xuất sắc', 'Giỏi', 'Khá', 'Trung bình', 'Yếu', 'Kém')
            
        Returns:
            Danh sách mã sinh viên
        """
        if df is None or df.empty:
            return []
        
        return df[df['performance_level'] == level]['student_id'].tolist()
    
    def get_students_need_attention(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Lấy danh sách sinh viên cần chú ý (điểm dưới ngưỡng)
        
        Args:
            df: DataFrame chứa dữ liệu lớp
            threshold: Ngưỡng điểm (mặc định 3.0)
            
        Returns:
            DataFrame chứa sinh viên cần chú ý
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        need_attention = df[df['clo_score'] < threshold].copy()
        need_attention = need_attention.sort_values('clo_score')
        
        return need_attention


def demo_class_mode():
    """Demo chế độ lớp"""
    print("=" * 80)
    print("DEMO: CHẾ ĐỘ LỚP HỌC")
    print("=" * 80)
    
    handler = UnifiedInputHandler()
    
    # Dữ liệu mẫu cho lớp
    subject_id = "INF1383"
    lecturer_name = "Nguyễn Văn A"
    student_list = ["SV001", "SV002", "SV003", "SV004", "SV005", "SV006", "SV007", "SV008"]
    scores = [5.5, 4.8, 3.2, 5.0, 2.5, 4.5, 5.8, 3.8]
    
    # Chuẩn bị dữ liệu
    df = handler.prepare_class_data(subject_id, lecturer_name, student_list, scores)
    
    if df is not None:
        print(f"\n✅ Đã chuẩn bị dữ liệu cho {len(df)} sinh viên")
        print("\n📋 Dữ liệu lớp:")
        print(df[['student_id', 'clo_score', 'performance_level']].to_string(index=False))
        
        # Thống kê
        stats = handler.get_class_statistics(df)
        handler.display_class_statistics(stats)
        
        # Sinh viên cần chú ý
        need_attention = handler.get_students_need_attention(df, threshold=3.5)
        if not need_attention.empty:
            print("\n" + "=" * 80)
            print("⚠️  SINH VIÊN CẦN CHÚ Ý (Điểm < 3.5)")
            print("=" * 80)
            for _, row in need_attention.iterrows():
                print(f"  - {row['student_id']}: {row['clo_score']:.2f}/6 ({row['performance_level']})")


def demo_individual_mode():
    """Demo chế độ cá nhân"""
    print("\n" + "=" * 80)
    print("DEMO: CHẾ ĐỘ CÁ NHÂN")
    print("=" * 80)
    
    handler = UnifiedInputHandler()
    
    # Dữ liệu mẫu cho cá nhân
    subject_id = "INF1383"
    lecturer_name = "Nguyễn Văn A"
    student_id = "SV001"
    clo_score = 4.5
    
    # Chuẩn bị dữ liệu
    data = handler.prepare_individual_data(subject_id, lecturer_name, student_id, clo_score)
    
    if data is not None:
        print(f"\n✅ Đã chuẩn bị dữ liệu cho sinh viên {student_id}")
        print("\n📋 Thông tin sinh viên:")
        print(f"  - Mã môn học:     {data['subject_id']}")
        print(f"  - Giảng viên:     {data['lecturer_name']}")
        print(f"  - Mã sinh viên:   {data['student_id']}")
        print(f"  - Điểm CLO:       {data['clo_score']:.2f}/6")
        print(f"  - Mức độ:         {data['performance_level']}")
        print(f"  - Chuẩn hóa:      {data['clo_score_normalized']:.3f}")


def main():
    """Main demo"""
    demo_class_mode()
    demo_individual_mode()
    
    print("\n✅ Demo hoàn thành!")


if __name__ == "__main__":
    main()

