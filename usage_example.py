#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USAGE EXAMPLE - Ví dụ sử dụng model_loader.py
Model paths đã được hard code trong package, không cần truyền tham số
"""

from model_loader import ClassAnalyzer, IndividualAnalyzer, PredictionTools

class_model_path = "trained_models/class_model/class_model.pkl"
class_metadata_path = "trained_models/class_model/metadata.pkl"
individual_model_path = "trained_models/individual_model/individual_model.pkl"
individual_metadata_path = "trained_models/individual_model/metadata.pkl"
# ============================================================================
# VÍ DỤ 1: Phân tích lớp học với API v2
# ============================================================================
def example_1_analyze_class():
    """Ví dụ 1: Phân tích lớp học"""
    print("=" * 80)
    print("VÍ DỤ 1: PHÂN TÍCH LỚP HỌC")
    print("=" * 80)
    
    # Khởi tạo analyzer - Không cần truyền model path!
    analyzer = ClassAnalyzer()
    
    # Dữ liệu sinh viên (list of dict - tiện cho Backend)
    students_data = [
        {"mssv": "SV001", "ho_ten": "Nguyễn Văn A", "diem_clo": 5.5},
        {"mssv": "SV002", "ho_ten": "Trần Thị B", "diem_clo": 4.8},
        {"mssv": "SV003", "ho_ten": "Lê Văn C", "diem_clo": 2.5},
    ]
    
    # Phân tích lớp với API mới (analyze_v2)
    result = analyzer.analyze_v2(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        students_data=students_data,
        top_k=2
    )
    
    if result:
        print(f"\n📊 KẾT QUẢ:")
        print(f"   Môn học: {result['subject_id']}")
        print(f"   Điểm TB: {result['statistics']['average_score']:.2f}/6")
        print(f"   SV cần can thiệp: {len(result['students_need_attention'])}")
        
        if result['students_need_attention']:
            print(f"\n⚠️  Sinh viên cần can thiệp:")
            for student in result['students_need_attention']:
                name = student.get('student_name', 'N/A')
                print(f"   • {student['student_id']} - {name}: {student['clo_score']:.2f}/6")


# ============================================================================
# VÍ DỤ 2: Phân tích lớp - API cũ (legacy)
# ============================================================================
def example_2_analyze_class_legacy():
    """Ví dụ 2: Phân tích lớp với API cũ (legacy)"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 2: PHÂN TÍCH LỚP - API CŨ (LEGACY)")
    print("=" * 80)
    
    # Khởi tạo analyzer
    analyzer = ClassAnalyzer()
    
    # API cũ - Phải tách student_list và scores riêng
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_list=["SV001", "SV002", "SV003"],
        scores=[5.5, 4.8, 2.5],
        top_k=2
    )
    
    if result:
        print(f"\n📊 KẾT QUẢ:")
        print(f"   Điểm TB: {result['statistics']['average_score']:.2f}/6")
        print(f"   SV cần can thiệp: {len(result['students_need_attention'])}")


# ============================================================================
# VÍ DỤ 3: Phân tích cá nhân sinh viên
# ============================================================================
def example_3_analyze_individual():
    """Ví dụ 3: Phân tích cá nhân"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 3: PHÂN TÍCH CÁ NHÂN SINH VIÊN")
    print("=" * 80)
    
    # Khởi tạo analyzer - Không cần truyền path!
    analyzer = IndividualAnalyzer()
    
    # Phân tích 1 sinh viên
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_id="SV001",
        top_k=3
    )
    
    if result:
        print(f"\n📊 KẾT QUẢ:")
        print(f"   Sinh viên: {result['student_id']}")
        if 'predicted_clo_score' in result:
            print(f"   Điểm CLO dự đoán: {result['predicted_clo_score']:.2f}/6")
        elif 'clo_score' in result:
            print(f"   Điểm CLO: {result['clo_score']:.2f}/6")
        if 'performance_level' in result:
            print(f"   Xếp loại: {result['performance_level']}")
        
        if result.get('comprehensive_analysis'):
            comp = result['comprehensive_analysis']
            print(f"   Phân tích toàn diện: {len(comp.get('analyses', {}))} khía cạnh")
        elif result.get('clo_analysis'):
            analysis = result['clo_analysis']
            print(f"   Mức độ: {analysis.get('severity_level', 'N/A')}")
            print(f"   Số reasons: {len(analysis.get('results', []))}")


# ============================================================================
# VÍ DỤ 4: Sử dụng PredictionTools
# ============================================================================
def example_4_prediction_tools():
    """Ví dụ 4: Sử dụng PredictionTools"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 4: PREDICTION TOOLS")
    print("=" * 80)
    
    # Khởi tạo tools - Không cần truyền path!
    tools = PredictionTools()
    
    # 1. Dự đoán Teaching Methods
    print("\n1️⃣ Phương pháp giảng dạy:")
    tm = tools.predict_teaching_methods(0.6, top_k=2)
    if tm:
        print(f"   Mức độ: {tm['severity_level']}")
        print(f"   Reasons: {len(tm['results'])}")
    
    # 2. Dự đoán CLO Attendance
    print("\n2️⃣ CLO Attendance:")
    clo = tools.predict_clo_attendance(0.5, top_k=2)
    if clo:
        print(f"   Mức độ: {clo['severity_level']}")
        for i, item in enumerate(clo['results'], 1):
            print(f"   {i}. {item['reason'][:60]}...")


# ============================================================================
# VÍ DỤ 5: Dự đoán toàn diện
# ============================================================================
def example_5_comprehensive():
    """Ví dụ 5: Dự đoán toàn diện nhiều dataset"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 5: DỰ ĐOÁN TOÀN DIỆN")
    print("=" * 80)
    
    tools = PredictionTools()
    
    # Dự đoán nhiều dataset cùng lúc
    results = tools.predict_comprehensive({
        'teaching_methods': 0.6,
        'evaluation_methods': 0.7,
        'clo_attendance': 0.55
    }, top_k=2)
    
    print(f"\n📊 Đã dự đoán {len(results)} datasets:")
    for key, result in results.items():
        if result:
            print(f"   • {key}: {result['severity_level']}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Chạy tất cả ví dụ"""
    print("\n" + "🎯" * 40)
    print("USAGE EXAMPLES - HƯỚNG DẪN SỬ DỤNG MODEL_LOADER")
    print("Model paths đã được hard code - không cần truyền tham số!")
    print("🎯" * 40)
    
    try:
        example_1_analyze_class()
        example_2_analyze_class_legacy()
        example_3_analyze_individual()
        example_4_prediction_tools()
        example_5_comprehensive()
        
        print("\n" + "=" * 80)
        print("✅ TẤT CẢ VÍ DỤ HOÀN THÀNH!")
        print("=" * 80)
        
        print("\n💡 ĐIỂM KHÁC BIỆT:")
        print("   ✅ Không cần truyền model_path, metadata_path")
        print("   ✅ Model paths đã được hard code trong package")
        print("   ✅ API đơn giản hơn: ClassAnalyzer(), IndividualAnalyzer(), PredictionTools()")
        print("   ✅ Sử dụng analyze_v2() cho phân tích lớp (list of dict)")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

