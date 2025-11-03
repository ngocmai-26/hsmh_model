#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USAGE EXAMPLE - Ví dụ sử dụng model_loader.py
"""

from model_loader import ModelLoader, ClassAnalyzer, IndividualAnalyzer, PredictionTools

class_model_path = "/Users/thuan.nguyen.trong/Desktop/workspace/thuannt/hsmh_model/trained_models/class_model/class_model.pkl"
class_metadata_path = "/Users/thuan.nguyen.trong/Desktop/workspace/thuannt/hsmh_model/trained_models/class_model/metadata.pkl"
individual_model_path = "trained_models/individual_model/individual_model.pkl"
individual_metadata_path = "trained_models/individual_model/metadata.pkl"
# ============================================================================
# VÍ DỤ 1: Load model trực tiếp và sử dụng
# ============================================================================
def example_1_basic_load():
    """Ví dụ 1: Load model cơ bản"""
    print("=" * 80)
    print("VÍ DỤ 1: LOAD MODEL CƠ BẢN")
    print("=" * 80)
    
    # Bước 1: Tạo ModelLoader
    loader = ModelLoader(class_model_path, class_metadata_path)
    
    # Bước 2: Load model từ file pickle
    loader.load()
    print("\n✅ Model đã load xong, có thể sử dụng!")
    
    # Bước 3: Sử dụng model để dự đoán
    result = loader.predict_reason_solution(
        dataset_key='clo_attendance',
        features=[0.6],  # Điểm CLO = 0.6 (normalized)
        top_k=3
    )
    
    if result:
        print(f"\n📊 Kết quả dự đoán:")
        print(f"   Dataset: {result['dataset']}")
        print(f"   Mức độ: {result['severity_level']}")
        print(f"   Số reasons: {len(result['results'])}")


# ============================================================================
# VÍ DỤ 2: Phân tích lớp học (đơn giản nhất)
# ============================================================================
def example_2_analyze_class_simple():
    """Ví dụ 2: Phân tích lớp - đơn giản"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 2: PHÂN TÍCH LỚP HỌC (ĐƠN GIẢN)")
    print("=" * 80)
    
    # Khởi tạo analyzer (tự động load model)
    analyzer = ClassAnalyzer(class_model_path, class_metadata_path)
    
    # Phân tích lớp
    analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_list=["SV001", "SV002", "SV003"],
        scores=[5.5, 4.0, 2.5],
        top_k=2,
        display=True
    )


# ============================================================================
# VÍ DỤ 3: Phân tích lớp học (lấy kết quả về xử lý)
# ============================================================================
def example_3_analyze_class_with_result():
    """Ví dụ 3: Phân tích lớp - lấy kết quả"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 3: PHÂN TÍCH LỚP - LẤY KẾT QUẢ")
    print("=" * 80)
    
    analyzer = ClassAnalyzer(class_model_path, class_metadata_path)
    
    # Phân tích (không display, lấy kết quả về)
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_list=["SV001", "SV002", "SV003", "SV004", "SV005"],
        scores=[5.5, 4.8, 3.2, 2.5, 4.5],
        top_k=3,
        display=False  # Không hiển thị, chỉ lấy kết quả
    )
    
    # Xử lý kết quả
    if result:
        print(f"📊 Thống kê lớp:")
        print(f"   - Tổng SV: {result['statistics']['total_students']}")
        print(f"   - Điểm TB: {result['statistics']['average_score']:.2f}")
        print(f"   - Tỉ lệ đạt: {result['statistics']['pass_rate']:.1f}%")
        
        print(f"\n⚠️  Sinh viên cần can thiệp: {len(result['students_need_attention'])}")
        for student in result['students_need_attention']:
            print(f"   • {student['student_id']}: {student['clo_score']:.2f}/6")
        
        print(f"\n💡 Reasons & Solutions:")
        if result['class_general_analysis']:
            for i, item in enumerate(result['class_general_analysis']['results'], 1):
                print(f"   {i}. {item['reason'][:60]}...")


# ============================================================================
# VÍ DỤ 4: Phân tích cá nhân
# ============================================================================
def example_4_analyze_individual():
    """Ví dụ 4: Phân tích cá nhân"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 4: PHÂN TÍCH CÁ NHÂN")
    print("=" * 80)
    
    analyzer = IndividualAnalyzer(individual_model_path, individual_metadata_path)
    
    # Phân tích 1 sinh viên
    result = analyzer.analyze(
        subject_id="INF1383",
        lecturer_name="Nguyễn Văn A",
        student_id="SV001",
        clo_score=3.5,
        top_k=5,
        display=True
    )


# ============================================================================
# VÍ DỤ 5: Sử dụng PredictionTools
# ============================================================================
def example_5_prediction_tools():
    """Ví dụ 5: Sử dụng PredictionTools"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 5: PREDICTION TOOLS")
    print("=" * 80)
    
    # Khởi tạo tools
    tools = PredictionTools(individual_model_path, individual_metadata_path)
    
    # 1. Dự đoán Teaching Methods
    print("\n1️⃣ Phương pháp giảng dạy:")
    tm = tools.predict_teaching_methods(0.6, top_k=2)
    if tm:
        print(f"   Mức độ: {tm['severity_level']}")
        print(f"   Reasons: {len(tm['results'])}")
    
    # 2. Dự đoán Evaluation Methods
    print("\n2️⃣ Phương pháp đánh giá:")
    em = tools.predict_evaluation_methods(0.5, top_k=2)
    if em:
        print(f"   Mức độ: {em['severity_level']}")
        print(f"   Reasons: {len(em['results'])}")
    
    # 3. Dự đoán CLO
    print("\n3️⃣ CLO Attendance:")
    clo = tools.predict_clo_attendance(0.75, top_k=2)
    if clo:
        print(f"   Mức độ: {clo['severity_level']}")
        for i, item in enumerate(clo['results'], 1):
            print(f"   {i}. {item['reason'][:70]}...")


# ============================================================================
# VÍ DỤ 6: Load model từ path cụ thể
# ============================================================================
def example_6_custom_model_path():
    """Ví dụ 6: Load model từ đường dẫn cụ thể"""
    print("\n" + "=" * 80)
    print("VÍ DỤ 6: LOAD MODEL TỪ PATH CỤ THỂ")
    print("=" * 80)
    
    # Load model từ path cụ thể
    loader = ModelLoader(individual_model_path, individual_metadata_path)
    
    if loader.load():
        print("\n✅ Đã load individual model!")
        
        # Sử dụng model
        result = loader.predict_reason_solution('clo_attendance', [0.5], top_k=2)
        if result:
            print(f"   Severity: {result['severity_level']}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Chạy tất cả ví dụ"""
    print("\n" + "🎯" * 40)
    print("USAGE EXAMPLES - HƯỚNG DẪN SỬ DỤNG MODEL_LOADER")
    print("🎯" * 40)
    
    print("\n📚 Các ví dụ:")
    print("   1. Load model cơ bản")
    print("   2. Phân tích lớp học (đơn giản)")
    print("   3. Phân tích lớp (lấy kết quả)")
    print("   4. Phân tích cá nhân")
    print("   5. Prediction tools")
    print("   6. Load model từ path cụ thể")
    
    try:
        example_1_basic_load()
        # example_2_analyze_class_simple()
        # example_3_analyze_class_with_result()
        # example_4_analyze_individual()
        # example_5_prediction_tools()
        # example_6_custom_model_path()
        
        print("\n" + "=" * 80)
        print("✅ TẤT CẢ VÍ DỤ HOÀN THÀNH!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

