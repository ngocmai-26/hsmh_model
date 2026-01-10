#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train và lưu CLOPredictor Model
Chỉ train model, không xử lý input/output
"""

import pickle
import os
from datetime import datetime
from model.predictor import CLOPredictor

def train_clo_predictor():
    """Train model cho CLO Predictor - CHỈ TRAIN MODEL"""
    
    print("=" * 80)
    print("🤖 TRAIN CLO PREDICTOR MODEL")
    print("=" * 80)
    
    # Tạo folder
    output_dir = "trained_models/clo_predictor"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo và train model
    print("\n🔄 Khởi tạo CLOPredictor...")
    print("⚠️  Lưu ý: Quá trình này sẽ mất thời gian do phải load và xử lý dữ liệu...")
    
    # Không tối ưu tham số để train nhanh hơn (đã tối ưu trong config.py)
    predictor = CLOPredictor(optimize_params=False)
    
    # Lưu toàn bộ predictor object (bao gồm data_loader, model_trainer, predictor)
    model_path = os.path.join(output_dir, "clo_predictor.pkl")
    print(f"\n💾 Lưu model: {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(predictor, f)
    
    # Lưu metadata
    metadata = {
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'clo_predictor',
        'has_data_loader': predictor.data_loader is not None,
        'has_model_trainer': predictor.model_trainer is not None,
        'has_predictor': predictor.predictor is not None,
        'optimize_params': predictor.optimize_params
    }
    
    with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Hoàn tất!")
    print(f"   Model đã lưu: {model_path}")
    print(f"   Metadata đã lưu: {os.path.join(output_dir, 'metadata.pkl')}")
    print(f"\n💡 Bây giờ bạn có thể sử dụng model này mà không cần train lại!")
    print(f"   Chạy: python3 run.py hoặc python3 run_interactive_with_file.py")
    
    return predictor


if __name__ == "__main__":
    try:
        train_clo_predictor()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

