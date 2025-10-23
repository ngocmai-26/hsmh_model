#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train cả 2 models: Lớp và Cá nhân
CHỈ TRAIN MODELS - Không xử lý input/output
"""

from datetime import datetime
from train_class_model import train_class_model
from train_individual_model import train_individual_model

def main():
    print("=" * 80)
    print("🤖 TRAIN TẤT CẢ MODELS")
    print("=" * 80)
    print(f"Bắt đầu: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Train model lớp
    print("1️⃣ MODEL CHO LỚP HỌC")
    print("-" * 80)
    try:
        train_class_model()
        print("✅ Model lớp: THÀNH CÔNG\n")
    except Exception as e:
        print(f"❌ Model lớp: LỖI - {e}\n")
        return
    
    # Train model cá nhân
    print("\n2️⃣ MODEL CHO CÁ NHÂN")
    print("-" * 80)
    try:
        train_individual_model()
        print("✅ Model cá nhân: THÀNH CÔNG\n")
    except Exception as e:
        print(f"❌ Model cá nhân: LỖI - {e}\n")
        return
    
    # Kết quả
    print("\n" + "=" * 80)
    print("✅ HOÀN TẤT TRAIN TẤT CẢ MODELS")
    print("=" * 80)
    print(f"Kết thúc: {datetime.now().strftime('%H:%M:%S')}")
    print("\n📁 Models đã lưu tại:")
    print("   - trained_models/class_model/class_model.pkl")
    print("   - trained_models/individual_model/individual_model.pkl")


if __name__ == "__main__":
    main()
