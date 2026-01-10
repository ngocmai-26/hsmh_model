#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLO Prediction System - Main Application
Hệ thống dự đoán CLO với tích hợp mô hình reasons và solutions CẢI TIẾN
"""

import sys
import os
from model.data_loader import DataLoader
from model.data_integration import DataIntegration
from model.feature_engineering import FeatureEngineering
from model.model_trainer import ModelTrainer
from model.predictor import CLOPredictor
# Thay thế mô hình cũ bằng mô hình cải tiến
from enhanced_integration import get_enhanced_reasons_predictor, predict_multiple_reasons
import traceback

def main():
    """Main application function"""
    print("=== CLO PREDICTION SYSTEM (ENHANCED VERSION) ===")
    
    # Load model từ file đã train sẵn (không train lại)
    print("🔄 Đang load CLOPredictor từ file đã train sẵn...")
    predictor = CLOPredictor(optimize_params=False, load_from_file=True)
    
    # Khởi tạo Enhanced Reasons & Solutions Predictor
    print("\n=== KHỞI TẠO ENHANCED REASONS & SOLUTIONS PREDICTOR ===")
    try:
        # Sử dụng mô hình cải tiến thay vì mô hình cũ
        reasons_predictor = get_enhanced_reasons_predictor()
        if reasons_predictor:
            print("✅ Đã khởi tạo Enhanced Reasons & Solutions Predictor")
            
            # Thêm vào CLOPredictor
            predictor.reasons_predictor = reasons_predictor
            print("✅ Đã tích hợp Enhanced Reasons & Solutions Predictor vào hệ thống chính")
        else:
            print("❌ Không thể khởi tạo Enhanced Reasons & Solutions Predictor")
            predictor.reasons_predictor = None
        
    except Exception as e:
        print(f"⚠️ Không thể khởi tạo Enhanced Reasons & Solutions Predictor: {e}")
        traceback.print_exc()
        print("Hệ thống sẽ hoạt động bình thường mà không có tính năng reasons & solutions")
        predictor.reasons_predictor = None
    
    # Main prediction loop
    while True:
        print("\n" + "="*50)
        print("CLO PREDICTION SYSTEM (ENHANCED)")
        print("="*50)
        
        # Nhập thông tin sinh viên
        try:
            student_id = input("Nhập mã sinh viên (hoặc 'quit' để thoát): ").strip()
            if student_id.lower() == 'quit':
                break
            
            lecturer = input("Nhập tên giảng viên: ").strip()
            subject_id = input("Nhập mã môn học: ").strip()
            
            if not all([student_id, lecturer, subject_id]):
                print("❌ Vui lòng nhập đầy đủ thông tin!")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        
        # Thực hiện dự đoán
        try:
            print(f"\n🔄 Đang dự đoán cho sinh viên {student_id}...")
            
            # Dự đoán điểm CLO (GIỮ NGUYÊN)
            predicted_score = predictor.predict(student_id, lecturer, subject_id)
            
            if predicted_score is not None:
                print(f"\n📊 KẾT QUẢ DỰ ĐOÁN CLO:")
                print(f"Điểm dự đoán: {predicted_score:.2f}/6")
                
                # Phân tích PPDG (GIỮ NGUYÊN)
                ppdg_analysis = predictor.analyze_ppdg_effectiveness(student_id, lecturer, subject_id)
                if ppdg_analysis:
                    print(f"\n📈 PHÂN TÍCH PPDG:")
                    print(f"Hiệu quả PPDG: {ppdg_analysis['effectiveness']:.2f}%")
                    print(f"Khuyến nghị: {ppdg_analysis['recommendations']}")
                
                # Sử dụng Enhanced Reasons & Solutions Predictor (MÔ HÌNH CẢI TIẾN)
                if predictor.reasons_predictor:
                    print(f"\n🎯 PHÂN TÍCH NHIỀU NGUYÊN NHÂN VÀ GIẢI PHÁP (MÔ HÌNH CẢI TIẾN):")
                    
                    # Tạo features nâng cao cho enhanced prediction
                    student_features = {
                        'reason_length': 120,
                        'solution_length': 180,
                        'reason_word_count': 25,
                        'solution_word_count': 35,
                        'reason_complexity': 0.15,
                        'solution_complexity': 0.12,
                        'has_technical_terms': 1 if predicted_score < 4.0 else 0,
                        'has_soft_skills': 1 if ppdg_analysis and ppdg_analysis.get('effectiveness', 100) < 70 else 0,
                        'has_assessment': 1 if predicted_score < 5.0 else 0,
                        'problem_type_frequency': 2 if predicted_score < 4.5 else 1,
                        'audience_numeric': 1,
                        'severity_problem_interaction': 2 if predicted_score < 4.0 else 1,
                        'subject_info': {
                            'id': subject_id,
                            'name': predictor.get_subject_name(subject_id)
                        }
                    }
                    
                    # Dự đoán NHIỀU nguyên nhân và giải pháp bằng mô hình cải tiến
                    print("🔄 Đang dự đoán nhiều nguyên nhân và giải pháp...")
                    reasons_result = predict_multiple_reasons(student_features, top_k=5)
                    
                    if reasons_result:
                        print(f"\n🎯 KẾT QUẢ DỰ ĐOÁN NHIỀU NGUYÊN NHÂN:")
                        print(f"📊 Mức độ nghiêm trọng: {reasons_result['severity_level']} (Độ tin cậy: {reasons_result['severity_confidence']:.3f})")
                        
                        print(f"\n🔍 {len(reasons_result['multiple_reasons'])} NGUYÊN NHÂN CHÍNH:")
                        for reason_data in reasons_result['multiple_reasons']:
                            print(f"\n--- Nguyên nhân #{reason_data['rank']} ---")
                            print(f"📝 Nguyên nhân: {reason_data['reason']}")
                            print(f"💡 Giải pháp: {reason_data['solution']}")
                            print(f"🏷️ Loại vấn đề: {reason_data['problem_type']}")
                            print(f"📊 Độ tin cậy: {reason_data['confidence']:.3f}")
                        
                        # Hiển thị tóm tắt phân tích
                        summary = reasons_result['analysis_summary']
                        print(f"\n📋 TÓM TẮT PHÂN TÍCH:")
                        print(f"  - Tổng số nguyên nhân: {summary['total_reasons']}")
                        print(f"  - Các lĩnh vực vấn đề chính: {', '.join(summary['main_problem_areas'])}")
                        print(f"  - Tác động dự kiến: {summary['expected_impact']}")
                        
                        print(f"\n⚡ HÀNH ĐỘNG ƯU TIÊN:")
                        for i, action in enumerate(summary['priority_actions'], 1):
                            print(f"  {i}. {action}")
                        
                        # Hiển thị recommendations nâng cao
                        if reasons_result['recommendations']:
                            print(f"\n📋 CÁC KHUYẾN NGHỊ NÂNG CAO:")
                            for i, rec in enumerate(reasons_result['recommendations'][:3], 1):
                                print(f"{i}. {rec['title']}: {rec['description']} (Ưu tiên: {rec['priority']})")
                    else:
                        print("❌ Không thể dự đoán nguyên nhân và giải pháp")
                else:
                    print("⚠️ Không có mô hình Enhanced Reasons & Solutions - chỉ hiển thị dự đoán điểm CLO")
                
            else:
                print("❌ Không thể dự đoán điểm cho sinh viên này!")
                
        except Exception as e:
            print(f"❌ Lỗi trong quá trình dự đoán: {e}")
            continue
        
        # Hỏi có muốn tiếp tục
        try:
            continue_choice = input("\nBạn có muốn dự đoán tiếp không? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
    
    print("\n👋 Cảm ơn bạn đã sử dụng CLO Prediction System (Enhanced Version)!")

if __name__ == "__main__":
    main() 