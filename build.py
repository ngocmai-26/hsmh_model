#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script build package HSMH Model
Sử dụng: python build.py
"""

import os
import sys
import shutil
import subprocess

def main():
    print("=" * 60)
    print("BUILD HSMH MODEL PACKAGE")
    print("=" * 60)
    
    # 1. Kiểm tra model files
    print("\n[1/7] Kiểm tra model files...")
    required_files = [
        "trained_models/class_model/class_model.pkl",
        "trained_models/class_model/metadata.pkl",
        "trained_models/individual_model/individual_model.pkl",
        "trained_models/individual_model/metadata.pkl",
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("❌ Thiếu files:")
        for f in missing:
            print(f"   - {f}")
        sys.exit(1)
    print("✅ OK")
    
    # 2. Clean build cũ
    print("\n[2/7] Clean build cũ...")
    for d in ['build', 'dist', 'hsmh_model.egg-info', 'hsmh_model']:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"   Đã xóa: {d}")
    print("✅ OK")
    
    # 3. Tạo cấu trúc package tạm thời
    print("\n[3/7] Tạo cấu trúc package...")
    os.makedirs('hsmh_model', exist_ok=True)
    
    # Copy model_loader.py
    shutil.copy2('model_loader.py', 'hsmh_model/model_loader.py')
    print("   ✓ Copied model_loader.py")
    
    # Copy unified_integration.py
    if os.path.exists('unified_integration.py'):
        shutil.copy2('unified_integration.py', 'hsmh_model/unified_integration.py')
        print("   ✓ Copied unified_integration.py")
    else:
        print("   ⚠️ Không tìm thấy unified_integration.py, bỏ qua")
    
    # Copy trained_models
    shutil.copytree('trained_models', 'hsmh_model/trained_models', dirs_exist_ok=True)
    print("   ✓ Copied trained_models/")

    # Copy thư mục model vào trong hsmh_model (để khi build có hsmh_model/model)
    if os.path.exists('model'):
        shutil.copytree('model', 'hsmh_model/model', dirs_exist_ok=True)
        print("   ✓ Copied model/")
    else:
        print("   ⚠️ Không tìm thấy thư mục 'model', bỏ qua copy model")
    
    # Tạo __init__.py
    with open('hsmh_model/__init__.py', 'w', encoding='utf-8') as f:
        f.write("""from .model_loader import ClassAnalyzer, IndividualAnalyzer, PredictionTools
__all__ = ['ClassAnalyzer', 'IndividualAnalyzer', 'PredictionTools']
""")
    print("   ✓ Created __init__.py")
    print("✅ OK")
    
    # 4. Upgrade build tools
    print("\n[4/7] Upgrade build tools...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', 
        '--upgrade', 'pip', 'setuptools', 'wheel', '--quiet'
    ])
    print("✅ OK")
    
    # 5. Build package
    print("\n[5/7] Build package...")
    result = subprocess.run([
        sys.executable, 'setup.py', 
        'sdist', 'bdist_wheel'
    ])
    
    if result.returncode != 0:
        print("❌ Build thất bại!")
        # Clean up
        if os.path.exists('hsmh_model'):
            shutil.rmtree('hsmh_model')
        sys.exit(1)
    print("✅ OK")
    
    # 6. Clean up thư mục tạm
    print("\n[6/7] Clean up...")
    if os.path.exists('hsmh_model'):
        shutil.rmtree('hsmh_model')
        print("   Đã xóa thư mục tạm hsmh_model/")
    print("✅ OK")
    
    # 7. Hiển thị kết quả
    print("\n[7/7] Kết quả:")
    print("=" * 60)
    print("✅ BUILD THÀNH CÔNG!")
    print("=" * 60)
    print("\nPackage files:")
    if os.path.exists('dist'):
        for f in os.listdir('dist'):
            size = os.path.getsize(os.path.join('dist', f)) / (1024*1024)
            print(f"  • {f} ({size:.2f} MB)")
    
    print("\n📥 Cài đặt:")
    print("   pip install dist/hsmh_model-0.0.1-py3-none-any.whl")
    
    print("\n✅ Sử dụng:")
    print("   from hsmh_model import ClassAnalyzer")
    print("   analyzer = ClassAnalyzer()")

if __name__ == "__main__":
    main()

