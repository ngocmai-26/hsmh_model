#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train và lưu MODEL CHO LỚP HỌC
Chỉ train model, không xử lý input/output
"""

import pickle
import os
from datetime import datetime
from model.unified_reasons_solutions_model import UnifiedReasonsSolutionsModel

def train_class_model():
    """Train model cho phân tích lớp - CHỈ TRAIN MODEL"""
    
    print("=" * 80)
    print("🤖 TRAIN MODEL CHO LỚP HỌC")
    print("=" * 80)
    
    # Tạo folder
    output_dir = "trained_models/class_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo và train model
    print("\n🔄 Khởi tạo model...")
    model = UnifiedReasonsSolutionsModel()
    
    print("📊 Load datasets...")
    model.load_all_datasets()
    
    print("\n🤖 Train models...")
    model.train_all_models()
    
    # Lưu model
    model_path = os.path.join(output_dir, "class_model.pkl")
    print(f"\n💾 Lưu model: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu metadata
    metadata = {
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'class',
        'num_datasets': len(model.datasets),
        'total_records': sum(len(df) for df in model.datasets.values())
    }
    
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Hoàn tất!")
    print(f"   Datasets: {metadata['num_datasets']}")
    print(f"   Records: {metadata['total_records']:,}")
    print(f"   Saved: {output_dir}")
    
    return model


if __name__ == "__main__":
    try:
        train_class_model()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
