#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script interactive - Nhập file danh sách sinh viên
"""

from unified_integration import analyze_class, analyze_individual, display_class_analysis, display_individual_analysis
import pandas as pd
import os

def input_class_mode_from_file():
    """Chế độ nhập cho lớp học - Lấy dữ liệu từ file"""
    print("\n" + "=" * 80)
    print("📚 CHẾ ĐỘ PHÂN TÍCH LỚP HỌC - NHẬP TỪ FILE")
    print("=" * 80)
    
    # Nhập thông tin lớp
    subject_id = input("\n📖 Nhập mã môn học (VD: INF1383): ").strip()
    lecturer_id = input("👨‍🏫 Nhập mã giảng viên (VD: GV001): ").strip()
    
    if not subject_id or not lecturer_id:
        print("❌ Mã môn học và mã giảng viên không được rỗng!")
        return
    
    # Nhập đường dẫn file
    print("\n📁 Nhập đường dẫn file danh sách sinh viên:")
    print("   Định dạng file: Excel (.xlsx, .xls) hoặc CSV (.csv)")
    print("   File phải có các cột: MSSV, HoTen, DiemCLO")
    file_path = input("   Đường dẫn file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return
    
    # Đọc file
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print("❌ Định dạng file không được hỗ trợ! Chỉ hỗ trợ .xlsx, .xls, .csv")
            return
        
        print(f"\n✅ Đọc file thành công! Tìm thấy {len(df)} sinh viên")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    
    # Kiểm tra các cột bắt buộc
    required_cols = ['MSSV', 'HoTen', 'DiemCLO']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ File thiếu các cột: {', '.join(missing_cols)}")
        print(f"   Các cột hiện có: {', '.join(df.columns.tolist())}")
        
        # Thử tìm các cột tương tự
        print("\n💡 Gợi ý: Các cột trong file của bạn:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        # Cho phép map lại tên cột
        print("\n🔧 Bạn có thể nhập lại tên cột:")
        mssv_col = input(f"   Cột chứa MSSV (mặc định: MSSV): ").strip() or 'MSSV'
        hoten_col = input(f"   Cột chứa Họ tên (mặc định: HoTen): ").strip() or 'HoTen'
        diem_col = input(f"   Cột chứa Điểm CLO (mặc định: DiemCLO): ").strip() or 'DiemCLO'
        
        # Đổi tên cột
        try:
            df = df.rename(columns={
                mssv_col: 'MSSV',
                hoten_col: 'HoTen',
                diem_col: 'DiemCLO'
            })
        except Exception as e:
            print(f"❌ Lỗi khi đổi tên cột: {e}")
            return
    
    # Kiểm tra và làm sạch dữ liệu
    df = df[['MSSV', 'HoTen', 'DiemCLO']].copy()
    
    # Loại bỏ dòng trống
    df = df.dropna(subset=['MSSV', 'DiemCLO'])
    
    # Chuyển đổi kiểu dữ liệu
    df['MSSV'] = df['MSSV'].astype(str).str.strip()
    df['HoTen'] = df['HoTen'].fillna('').astype(str).str.strip()
    df['DiemCLO'] = pd.to_numeric(df['DiemCLO'], errors='coerce')
    
    # Kiểm tra điểm hợp lệ
    invalid_scores = df[(df['DiemCLO'] < 0) | (df['DiemCLO'] > 6)]
    if not invalid_scores.empty:
        print(f"\n⚠️  Cảnh báo: Có {len(invalid_scores)} sinh viên có điểm không hợp lệ (phải từ 0-6):")
        print(invalid_scores[['MSSV', 'HoTen', 'DiemCLO']].to_string(index=False))
        
        choice = input("\n   Loại bỏ các sinh viên này? (y/n): ").strip().lower()
        if choice == 'y':
            df = df[(df['DiemCLO'] >= 0) & (df['DiemCLO'] <= 6)]
            print(f"   ✅ Đã loại bỏ. Còn lại {len(df)} sinh viên")
        else:
            print("   ❌ Hủy phân tích!")
            return
    
    # Hiển thị danh sách
    print("\n" + "=" * 80)
    print("📋 DANH SÁCH SINH VIÊN TỪ FILE:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print(f"\n📊 Tóm tắt:")
    print(f"   - Số sinh viên: {len(df)}")
    print(f"   - Điểm trung bình: {df['DiemCLO'].mean():.2f}/6")
    print(f"   - Điểm thấp nhất: {df['DiemCLO'].min():.2f}/6")
    print(f"   - Điểm cao nhất: {df['DiemCLO'].max():.2f}/6")
    
    # Xác nhận
    confirm = input("\n✅ Xác nhận phân tích? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy phân tích!")
        return
    
    # Chuẩn bị dữ liệu
    student_list = df['MSSV'].tolist()
    scores = df['DiemCLO'].tolist()
    
    # Phân tích
    print("\n🔄 Đang phân tích lớp học...")
    result = analyze_class(
        subject_id=subject_id,
        lecturer_name=lecturer_id,
        student_list=student_list,
        scores=scores,
        top_k=3
    )
    
    if result:
        # Hiển thị kết quả
        display_class_analysis(result)
        
        # Hiển thị danh sách sinh viên cần can thiệp (với tên)
        if result['students_need_attention']:
            print("\n" + "=" * 80)
            print("📋 CHI TIẾT SINH VIÊN CẦN CAN THIỆP")
            print("=" * 80)
            for student in result['students_need_attention']:
                # Tìm tên sinh viên từ df
                student_info = df[df['MSSV'] == student['student_id']]
                if not student_info.empty:
                    hoten = student_info.iloc[0]['HoTen']
                    print(f"  • {hoten:30} ({student['student_id']}): {student['clo_score']:.2f}/6 - {student['performance_level']}")
        
        print("\n✅ Phân tích hoàn tất!")
    else:
        print("❌ Không thể phân tích lớp học!")


def input_individual_mode():
    """Chế độ nhập cho cá nhân"""
    print("\n" + "=" * 80)
    print("👤 CHẾ ĐỘ PHÂN TÍCH CÁ NHÂN")
    print("=" * 80)
    
    # Nhập thông tin
    subject_id = input("\n📖 Nhập mã môn học (VD: INF1383): ").strip()
    lecturer_id = input("👨‍🏫 Nhập mã giảng viên (VD: GV001): ").strip()
    student_id = input("🎓 Nhập mã sinh viên (VD: SV2021001): ").strip()
    
    if not subject_id or not lecturer_name or not student_id:
        print("❌ Thông tin không được rỗng!")
        return
    
    try:
        clo_score = float(input("📊 Nhập điểm CLO (0-6): "))
        if not (0 <= clo_score <= 6):
            print("❌ Điểm phải trong khoảng 0-6!")
            return
    except ValueError:
        print("❌ Điểm không hợp lệ!")
        return
    
    # Hiển thị thông tin vừa nhập
    print("\n" + "=" * 80)
    print("📋 THÔNG TIN VỪA NHẬP:")
    print("=" * 80)
    print(f"Môn học:     {subject_id}")
    print(f"Mã giảng viên: {lecturer_id}")
    print(f"Sinh viên:   {student_id}")
    print(f"Điểm CLO:    {clo_score:.2f}/6")
    
    # Xác nhận
    confirm = input("\n✅ Xác nhận phân tích? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy phân tích!")
        return
    
    # Phân tích
    print("\n🔄 Đang phân tích sinh viên...")
    result = analyze_individual(
        subject_id=subject_id,
        lecturer_name=lecturer_id,
        student_id=student_id,
        clo_score=clo_score,
        top_k=5  # Cá nhân thì lấy nhiều reasons hơn
    )
    
    if result:
        display_individual_analysis(result)
        print("\n✅ Phân tích hoàn tất!")
    else:
        print("❌ Không thể phân tích sinh viên!")


def main():
    """Main interactive"""
    print("=" * 80)
    print("🎯 HỆ THỐNG PHÂN TÍCH CLO")
    print("=" * 80)
    print("\nHệ thống hỗ trợ 2 chế độ phân tích:")
    print("  1. Phân tích LỚP HỌC - Nhập từ FILE Excel/CSV")
    print("  2. Phân tích CÁ NHÂN - Phân tích chi tiết 1 sinh viên")
    
    while True:
        print("\n" + "=" * 80)
        choice = input("\n🔢 Chọn chế độ (1: Lớp, 2: Cá nhân, 0: Thoát): ").strip()
        
        if choice == '1':
            input_class_mode_from_file()
        elif choice == '2':
            input_individual_mode()
        elif choice == '0':
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ! Vui lòng chọn 1, 2 hoặc 0.")
        
        # Hỏi có muốn tiếp tục
        if choice in ['1', '2']:
            continue_choice = input("\n🔄 Tiếp tục phân tích? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
                break


if __name__ == "__main__":
    main()

