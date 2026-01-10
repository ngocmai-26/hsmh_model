#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đơn giản chuyển đổi dữ liệu lý do và cách khắc phục
"""

import pandas as pd
import json
import numpy as np

def create_sample_dataset(num_samples=1000):
    """Tạo dataset mẫu"""
    print(f"Đang tạo {num_samples} mẫu dữ liệu...")
    
    # Các mẫu lý do và cách khắc phục cơ bản
    base_reasons = [
        "Phương pháp đánh giá (PPDG) chưa đủ đa dạng",
        "Điểm đa dạng PPDG thấp",
        "Thiếu đánh giá quá trình",
        "Thiếu đánh giá tổng kết",
        "PPDG không tương thích với phương pháp giảng dạy",
        "Sinh viên có điểm tích lũy thấp",
        "Tỷ lệ pass môn học thấp",
        "Sinh viên có lịch sử vắng thi nhiều lần",
        "Điểm rèn luyện trung bình thấp",
        "Sinh viên tự học ít"
    ]
    
    base_solutions = [
        "Bổ sung thêm PPDG đa dạng",
        "Tăng cường đa dạng hóa PPDG",
        "Bổ sung đánh giá chuyên cần, bài tập cá nhân",
        "Bổ sung kiểm tra viết, trắc nghiệm",
        "Điều chỉnh PPDG để phù hợp với PPGD",
        "Tăng cường hỗ trợ học tập, phụ đạo",
        "Cải thiện phương pháp giảng dạy",
        "Tăng cường đánh giá quá trình",
        "Tăng cường hoạt động ngoại khóa",
        "Khuyến khích tự học, cung cấp tài liệu"
    ]
    
    problem_types = [
        "PPDG", "PPGD", "Điểm tích lũy", "Tỷ lệ pass", "Chuyên cần",
        "Điểm rèn luyện", "Tự học", "CLO", "Đánh giá", "Kỹ năng học tập"
    ]
    
    audiences = ["Sinh viên", "Giảng viên", "Nhà trường"]
    severity_levels = ["CAO", "TRUNG BÌNH", "THẤP"]
    
    data = []
    for i in range(num_samples):
        reason_idx = i % len(base_reasons)
        solution_idx = i % len(base_solutions)
        
        data.append({
            'id': i + 1,
            'reason_text': f"{base_reasons[reason_idx]} (mẫu {i+1})",
            'severity_level': severity_levels[i % len(severity_levels)],
            'solution_text': f"{base_solutions[solution_idx]} (mẫu {i+1})",
            'problem_type': problem_types[i % len(problem_types)],
            'target_audience': audiences[i % len(audiences)]
        })
    
    return data

def encode_data(data):
    """Mã hóa dữ liệu"""
    df = pd.DataFrame(data)
    
    # Mã hóa severity level
    severity_mapping = {'CAO': 3, 'TRUNG BÌNH': 2, 'THẤP': 1}
    df['severity_score'] = df['severity_level'].map(severity_mapping)
    
    # Mã hóa problem type
    problem_types = df['problem_type'].unique()
    problem_mapping = {pt: i for i, pt in enumerate(problem_types)}
    df['problem_encoded'] = df['problem_type'].map(problem_mapping)
    
    # Mã hóa target audience
    audiences = df['target_audience'].unique()
    audience_mapping = {aud: i for i, aud in enumerate(audiences)}
    df['audience_encoded'] = df['target_audience'].map(audience_mapping)
    
    # Mã hóa text (đơn giản: sử dụng độ dài)
    df['reason_encoded'] = df['reason_text'].str.len()
    df['solution_encoded'] = df['solution_text'].str.len()
    
    # Thêm effectiveness_score (placeholder)
    df['effectiveness_score'] = np.random.uniform(0.5, 1.0, len(df))
    
    return df, problem_mapping, audience_mapping

def create_metadata(df, problem_mapping, audience_mapping):
    """Tạo metadata"""
    metadata = {
        "encoding_mappings": {
            "severity_levels": {"CAO": 3, "TRUNG BÌNH": 2, "THẤP": 1},
            "problem_types": problem_mapping,
            "audiences": audience_mapping
        },
        "text_encoding": {
            "method": "Length-based",
            "description": "Sử dụng độ dài text làm feature"
        },
        "dataset_stats": {
            "total_records": len(df),
            "problem_type_distribution": df['problem_type'].value_counts().to_dict(),
            "severity_distribution": df['severity_level'].value_counts().to_dict(),
            "audience_distribution": df['target_audience'].value_counts().to_dict()
        }
    }
    
    return metadata

def save_data(df, metadata):
    """Lưu dữ liệu"""
    print("Đang lưu dữ liệu...")
    
    # Lưu CSV
    csv_path = "reasons_solutions_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Đã lưu CSV: {csv_path}")
    
    # Lưu metadata
    metadata_path = "reasons_solutions_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu metadata: {metadata_path}")
    
    # Lưu pickle
    pickle_path = "reasons_solutions_dataset.pkl"
    df.to_pickle(pickle_path)
    print(f"Đã lưu pickle: {pickle_path}")
    
    return csv_path, metadata_path, pickle_path

def main():
    """Hàm chính"""
    print("=== CHUYỂN ĐỔI DỮ LIỆU LÝ DO VÀ CÁCH KHẮC PHỤC ===")
    
    # 1. Tạo dữ liệu mẫu
    data = create_sample_dataset(1000)
    
    # 2. Mã hóa dữ liệu
    df, problem_mapping, audience_mapping = encode_data(data)
    
    # 3. Tạo metadata
    metadata = create_metadata(df, problem_mapping, audience_mapping)
    
    # 4. Lưu dữ liệu
    csv_path, metadata_path, pickle_path = save_data(df, metadata)
    
    # 5. In thống kê
    print("\n=== THỐNG KÊ DỮ LIỆU ===")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Phân bố mức độ nghiêm trọng:")
    print(df['severity_level'].value_counts())
    print(f"\nPhân bố loại vấn đề:")
    print(df['problem_type'].value_counts())
    print(f"\nPhân bố đối tượng:")
    print(df['target_audience'].value_counts())
    
    print("\n✅ Chuyển đổi thành công!")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Pickle: {pickle_path}")

if __name__ == "__main__":
    main() 