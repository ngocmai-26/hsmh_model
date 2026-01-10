#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Integration - Tích hợp Unified Reasons & Solutions Model vào hệ thống chính
Hỗ trợ 2 chế độ:
1. Chế độ lớp: Phân tích cho cả lớp (nhiều sinh viên)
2. Chế độ cá nhân: Phân tích cho 1 sinh viên
"""

# Import tương thích với package structure
try:
    # Khi chạy từ package đã cài đặt
    from hsmh_model.model.unified_reasons_solutions_model import UnifiedReasonsSolutionsModel
    from hsmh_model.model.unified_input_handler import UnifiedInputHandler
except ImportError:
    # Khi chạy từ source code
    from model.unified_reasons_solutions_model import UnifiedReasonsSolutionsModel
    from model.unified_input_handler import UnifiedInputHandler
import pandas as pd
import pickle
import os
import traceback

# Global model instance
_unified_model = None
_input_handler = None

# Đường dẫn đến model đã train
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_models", "individual_model", "individual_model.pkl")


def get_unified_model():
    """Lấy hoặc khởi tạo unified model - LOAD từ file đã train sẵn"""
    global _unified_model
    
    if _unified_model is None:
        print("🔄 Đang tải Unified Reasons & Solutions Model...")
        try:
            # Thử LOAD model đã train sẵn từ file
            if os.path.exists(_MODEL_PATH):
                print(f"📂 Tìm thấy model đã train tại: {_MODEL_PATH}")
                print("🔄 Đang load model từ file...")
                
                with open(_MODEL_PATH, 'rb') as f:
                    _unified_model = pickle.load(f)
                
                print("✅ Đã load model thành công từ file!")
                return _unified_model
            else:
                # Nếu không tìm thấy model đã train, báo lỗi và hướng dẫn train
                print(f"❌ Không tìm thấy model đã train tại: {_MODEL_PATH}")
                print("\n💡 HƯỚNG DẪN:")
                print("   Để sử dụng hệ thống, bạn cần train model trước:")
                print("   python3 train_all_models.py")
                print("\n   Hoặc train riêng:")
                print("   python3 train_individual_model.py  # Cho phân tích cá nhân")
                print("   python3 train_class_model.py      # Cho phân tích lớp")
                print("\n⚠️  Hệ thống sẽ không train tự động để tránh tốn thời gian mỗi lần chạy!")
                return None
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            traceback.print_exc()
            print("\n💡 Nếu lỗi do model không tương thích, hãy train lại:")
            print("   python3 train_all_models.py")
            return None
    
    return _unified_model


def get_input_handler():
    """Lấy hoặc khởi tạo input handler"""
    global _input_handler
    
    if _input_handler is None:
        _input_handler = UnifiedInputHandler()
    
    return _input_handler


def predict_teaching_methods(teaching_method_score, top_k=3):
    """Dự đoán reasons & solutions cho Teaching Methods"""
    model = get_unified_model()
    if model is None:
        return None
    
    try:
        features = [teaching_method_score]
        return model.predict_reason_solution('teaching_methods', features, top_k)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán teaching methods: {e}")
        return None


def predict_evaluation_methods(evaluation_method_score, top_k=3):
    """Dự đoán reasons & solutions cho Evaluation Methods"""
    model = get_unified_model()
    if model is None:
        return None
    
    try:
        features = [evaluation_method_score]
        return model.predict_reason_solution('evaluation_methods', features, top_k)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán evaluation methods: {e}")
        return None


def predict_student_conduct(conduct_score, top_k=3):
    """Dự đoán reasons & solutions cho Student Conduct"""
    model = get_unified_model()
    if model is None:
        return None
    
    try:
        features = [conduct_score]
        return model.predict_reason_solution('student_conduct', features, top_k)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán student conduct: {e}")
        return None


def predict_academic_midterm(midterm_score, top_k=3):
    """Dự đoán reasons & solutions cho Academic Midterm"""
    model = get_unified_model()
    if model is None:
        return None
    
    try:
        features = [midterm_score]
        return model.predict_reason_solution('academic_midterm', features, top_k)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán academic midterm: {e}")
        return None


def predict_clo_attendance(clo_score, top_k=3):
    """Dự đoán reasons & solutions cho CLO Attendance"""
    model = get_unified_model()
    if model is None:
        return None
    
    try:
        features = [clo_score]
        return model.predict_reason_solution('clo_attendance', features, top_k)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán CLO attendance: {e}")
        return None


def predict_comprehensive_analysis(student_data, top_k=3):
    """Phân tích toàn diện tất cả các khía cạnh của sinh viên"""
    model = get_unified_model()
    if model is None:
        return None
    
    results = {
        'student_id': student_data.get('student_id', 'Unknown'),
        'analyses': {}
    }
    
    # Phân tích Teaching Methods
    if 'teaching_method_score' in student_data:
        tm_result = predict_teaching_methods(
            student_data['teaching_method_score'], top_k
        )
        if tm_result:
            results['analyses']['teaching_methods'] = tm_result
    
    # Phân tích Evaluation Methods
    if 'evaluation_method_score' in student_data:
        em_result = predict_evaluation_methods(
            student_data['evaluation_method_score'], top_k
        )
        if em_result:
            results['analyses']['evaluation_methods'] = em_result
    
    # Phân tích Student Conduct
    if 'conduct_score' in student_data:
        sc_result = predict_student_conduct(
            student_data['conduct_score'], top_k
        )
        if sc_result:
            results['analyses']['student_conduct'] = sc_result
    
    # Phân tích Academic Midterm
    if 'midterm_score' in student_data:
        am_result = predict_academic_midterm(
            student_data['midterm_score'], top_k
        )
        if am_result:
            results['analyses']['academic_midterm'] = am_result
    
    # Phân tích CLO Attendance
    if 'clo_score' in student_data:
        ca_result = predict_clo_attendance(
            student_data['clo_score'], top_k
        )
        if ca_result:
            results['analyses']['clo_attendance'] = ca_result
    
    return results


def display_comprehensive_analysis(results):
    """Hiển thị kết quả phân tích toàn diện"""
    if not results or 'analyses' not in results:
        print("❌ Không có kết quả phân tích!")
        return
    
    print("\n" + "=" * 80)
    print(f"PHÂN TÍCH TOÀN DIỆN CHO SINH VIÊN: {results['student_id']}")
    print("=" * 80)
    
    for key, analysis in results['analyses'].items():
        print(f"\n{'=' * 80}")
        print(f"📊 {analysis['dataset']}")
        print(f"{'=' * 80}")
        print(f"🎯 Mức độ: {analysis['severity_level']} (Độ tin cậy: {analysis['severity_confidence']:.3f})")
        
        for i, item in enumerate(analysis['results'], 1):
            print(f"\n   --- Phân tích #{i} ---")
            print(f"   📝 Nguyên nhân: {item['reason']}")
            print(f"   💡 Giải pháp: {item['solution']}")


def analyze_class(subject_id, lecturer_id, student_list, scores, top_k=3):
    """
    Phân tích cho cả lớp học - CHỈ NHẬN XÉT CHUNG (Legacy API)
    
    ⚠️ CẢNH BÁO: Hàm này yêu cầu tách student_list và scores riêng.
    🔥 KHUYẾN NGHỊ: Sử dụng analyze_class_v2() cho Backend - tiện hơn!
    
    Args:
        subject_id: Mã môn học
        lecturer_id: Mã giảng viên (VD: "GV001")
        student_list: Danh sách mã sinh viên
        scores: Danh sách điểm CLO (0-6)
        top_k: Số lượng reasons/solutions trả về
        
    Returns:
        Dictionary chứa kết quả phân tích lớp (chỉ nhận xét chung)
    """
    handler = get_input_handler()
    model = get_unified_model()
    
    if handler is None or model is None:
        return None
    
    # Chuẩn bị dữ liệu lớp
    df = handler.prepare_class_data(subject_id, lecturer_id, student_list, scores)
    if df is None:
        return None
    
    # Thống kê lớp
    stats = handler.get_class_statistics(df)
    
    # Phân tích chung cho lớp dựa trên điểm trung bình
    # Lấy điểm đã được normalize từ DataFrame (đã xử lý trong prepare_class_data)
    avg_score_normalized = df['clo_score_normalized'].mean()
    
    # Dự đoán reasons & solutions chung cho lớp
    class_analysis = predict_clo_attendance(avg_score_normalized, top_k)
    
    return {
        'mode': 'class',
        'subject_id': subject_id,
        'lecturer_id': lecturer_id,
        'statistics': stats,
        'class_general_analysis': class_analysis,  # Nhận xét chung cho cả lớp
        'students_need_attention': handler.get_students_need_attention(df).to_dict('records')
    }


def analyze_class_v2(subject_id, lecturer_id, students_data, top_k=3):
    """
    Phân tích cho cả lớp học - Nhận data dạng list of dict
    
    Args:
        subject_id: Mã môn học (VD: "INF1383")
        lecturer_id: Mã giảng viên (VD: "GV001")
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
        
    Returns:
        Dictionary chứa kết quả phân tích lớp
    """
    
    # Validate input
    if not students_data or not isinstance(students_data, list):
        print("❌ students_data phải là một list không rỗng!")
        return None
    
    if len(students_data) == 0:
        print("❌ Danh sách sinh viên trống!")
        return None
    
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
            print(f"⚠️ Bỏ qua sinh viên thứ {idx+1}: không phải dict")
            continue
        
        # Tìm MSSV
        mssv = None
        for key in mssv_keys:
            if key in student:
                mssv = str(student[key]).strip()
                break
        
        if not mssv:
            print(f"⚠️ Bỏ qua sinh viên thứ {idx+1}: không tìm thấy MSSV")
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
            print(f"⚠️ Bỏ qua sinh viên {mssv}: không tìm thấy điểm CLO")
            continue
        
        # Validate điểm
        if not (0 <= score <= 6):
            print(f"⚠️ Bỏ qua sinh viên {mssv}: điểm không hợp lệ ({score})")
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
        print("❌ Không có sinh viên hợp lệ nào sau khi xử lý!")
        return None
    
    print(f"✅ Đã xử lý {len(student_list)}/{len(students_data)} sinh viên")
    
    # Gọi hàm cũ với data đã tách
    result = analyze_class(subject_id, lecturer_id, student_list, scores, top_k)
    
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
    
    return result


def analyze_individual(subject_id, lecturer_id, student_id, top_k=3):
    """
    Phân tích cho 1 sinh viên cụ thể
    
    Args:
        subject_id: Mã môn học
        lecturer_id: Mã giảng viên (VD: "GV001")
        student_id: Mã sinh viên
        top_k: Số lượng reasons/solutions trả về
        
    Returns:
        Dictionary chứa kết quả phân tích cá nhân
    """
    handler = get_input_handler()
    model = get_unified_model()
    
    if handler is None or model is None:
        return None
    
    # DỰ ĐOÁN điểm CLO từ dữ liệu sinh viên
    try:
        # Import tương thích với package structure
        try:
            from hsmh_model.model.predictor import CLOPredictor
        except ImportError:
            from model.predictor import CLOPredictor
    except ImportError as e:
        print(f"❌ Không thể import CLOPredictor: {e}")
        return None
    
    try:
        # Load model từ file đã train sẵn (không train lại)
        predictor = CLOPredictor(optimize_params=False, load_from_file=True)
        predicted_score = predictor.predict(student_id, lecturer_id, subject_id)
        
        if predicted_score is None or predicted_score < 0:
            print(f"❌ Không thể dự đoán điểm CLO cho sinh viên {student_id}")
            return None
        
        print(f"✅ Đã dự đoán điểm CLO: {predicted_score:.2f}/6")
        clo_score = predicted_score
        
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán điểm CLO: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Chuẩn bị dữ liệu cá nhân
    data = handler.prepare_individual_data(subject_id, lecturer_id, student_id, clo_score)
    if data is None:
        return None
    
    # Phân tích toàn diện
    student_data = {
        'student_id': student_id,
        'clo_score': data['clo_score_normalized']
    }
    
    comprehensive = predict_comprehensive_analysis(student_data, top_k)
    
    return {
        'mode': 'individual',
        'subject_id': subject_id,
        'lecturer_id': lecturer_id,
        'student_id': student_id,
        'clo_score': clo_score,
        'predicted_clo_score': predicted_score,  # Thêm điểm dự đoán
        'performance_level': data['performance_level'],
        'comprehensive_analysis': comprehensive
    }


def display_class_analysis(result):
    """Hiển thị kết quả phân tích lớp - CHỈ NHẬN XÉT CHUNG"""
    if not result or result.get('mode') != 'class':
        print("❌ Không có kết quả phân tích lớp!")
        return
    
    print("\n" + "=" * 80)
    print(f"📚 PHÂN TÍCH LỚP HỌC: {result['subject_id']}")
    print(f"👨‍🏫 Mã giảng viên: {result['lecturer_id']}")
    print("=" * 80)
    
    # Hiển thị thống kê
    handler = get_input_handler()
    handler.display_class_statistics(result['statistics'])
    
    # NHẬN XÉT CHUNG CHO CẢ LỚP
    if result.get('class_general_analysis'):
        print("\n" + "=" * 80)
        print("💡 NHẬN XÉT CHUNG VÀ KHUYẾN NGHỊ CHO LỚP")
        print("=" * 80)
        
        analysis = result['class_general_analysis']
        stats = result['statistics']
        
        print(f"\n📊 Mức độ chung: {analysis.get('severity_level', 'N/A')}")
        print(f"   Độ tin cậy: {analysis.get('severity_confidence', 0):.3f}")
        
        # Nhận xét dựa trên điểm trung bình
        avg_score = stats['average_score']
        if avg_score >= 5.0:
            print("\n✨ Đánh giá tổng quan:")
            print("   Lớp học có kết quả XUẤT SẮC!")
            print("   - Phần lớn sinh viên đạt và vượt chuẩn đầu ra")
            print("   - Tiếp tục duy trì và phát huy")
        elif avg_score >= 4.0:
            print("\n👍 Đánh giá tổng quan:")
            print("   Lớp học có kết quả TỐT")
            print("   - Đa số sinh viên đạt chuẩn đầu ra")
            print("   - Cần tăng cường hỗ trợ một số sinh viên")
        elif avg_score >= 3.0:
            print("\n⚠️  Đánh giá tổng quan:")
            print("   Lớp học có kết quả TRUNG BÌNH")
            print("   - Cần cải thiện phương pháp giảng dạy")
            print("   - Tăng cường hỗ trợ sinh viên yếu")
        else:
            print("\n❌ Đánh giá tổng quan:")
            print("   Lớp học CẦN CẢI THIỆN KHẨN CẤP")
            print("   - Xem xét lại toàn bộ quy trình giảng dạy")
            print("   - Can thiệp ngay lập tức")
        
        # Hiển thị reasons & solutions chung
        if 'results' in analysis and analysis['results']:
            print(f"\n🎯 CÁC VẤN ĐỀ CHUNG CẦN LƯU Ý:")
            for i, item in enumerate(analysis['results'], 1):
                print(f"\n{i}. Vấn đề:")
                print(f"   {item['reason']}")
                print(f"   → Giải pháp:")
                print(f"   {item['solution']}")
    
    # Sinh viên cần chú ý
    if result['students_need_attention']:
        print("\n" + "=" * 80)
        print("⚠️  DANH SÁCH SINH VIÊN CẦN CAN THIỆP (Điểm < 3.0)")
        print("=" * 80)
        print(f"Có {len(result['students_need_attention'])} sinh viên cần hỗ trợ đặc biệt:")
        for student in result['students_need_attention']:
            print(f"  • {student['student_id']}: {student['clo_score']:.2f}/6 ({student['performance_level']})")


def display_individual_analysis(result):
    """Hiển thị kết quả phân tích cá nhân"""
    if not result or result.get('mode') != 'individual':
        print("❌ Không có kết quả phân tích cá nhân!")
        return
    
    print("\n" + "=" * 80)
    print(f"👤 PHÂN TÍCH CÁ NHÂN")
    print("=" * 80)
    print(f"📚 Môn học:       {result['subject_id']}")
    print(f"👨‍🏫 Mã giảng viên: {result['lecturer_id']}")
    print(f"🎓 Sinh viên:     {result['student_id']}")
    print(f"📊 Điểm CLO dự đoán: {result.get('predicted_clo_score', result['clo_score']):.2f}/6")
    print(f"🏆 Xếp loại:      {result['performance_level']}")
    
    # Phân tích toàn diện
    if result['comprehensive_analysis']:
        display_comprehensive_analysis(result['comprehensive_analysis'])


def main():
    """Demo integration với 2 chế độ"""
    print("=" * 80)
    print("UNIFIED INTEGRATION - DEMO 2 CHẾ ĐỘ")
    print("=" * 80)
    
    # Khởi tạo model
    model = get_unified_model()
    if model is None:
        print("❌ Không thể khởi tạo model!")
        return
    
    # Demo 1: Chế độ lớp
    print("\n" + "=" * 80)
    print("DEMO 1: CHẾ ĐỘ LỚP HỌC")
    print("=" * 80)
    
    class_result = analyze_class(
        subject_id="INF1383",
        lecturer_id="GV001",
        student_list=["SV001", "SV002", "SV003", "SV004", "SV005"],
        scores=[5.5, 4.8, 3.2, 2.5, 4.5],
        top_k=2
    )
    
    if class_result:
        display_class_analysis(class_result)
    
    # Demo 2: Chế độ cá nhân
    print("\n\n" + "=" * 80)
    print("DEMO 2: CHẾ ĐỘ CÁ NHÂN")
    print("=" * 80)
    
    individual_result = analyze_individual(
        subject_id="INF1383",
        lecturer_id="GV001",
        student_id="SV001",
        top_k=3
    )
    
    if individual_result:
        display_individual_analysis(individual_result)
    
    print("\n✅ Demo hoàn thành!")


if __name__ == "__main__":
    main()

