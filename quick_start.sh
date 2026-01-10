#!/bin/bash
# Quick Start Script - Hướng dẫn nhanh

echo "=========================================="
echo "🚀 HỆ THỐNG PHÂN TÍCH CLO - QUICK START"
echo "=========================================="
echo ""
echo "Chọn chức năng:"
echo "1. Train tất cả models (chỉ làm 1 lần hoặc khi có cập nhật)"
echo "2. Chạy hệ thống (sử dụng model đã train)"
echo "3. Test/Thử hệ thống"
echo "4. Kiểm tra model đã train chưa"
echo "5. Thoát"
echo ""
read -p "Nhập lựa chọn (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🔄 Đang train tất cả models..."
        echo "⏱️  Thời gian: 5-15 phút"
        python3 train_all_models.py
        ;;
    2)
        echo ""
        echo "🚀 Đang khởi động hệ thống..."
        python3 run_interactive_with_file.py
        ;;
    3)
        echo ""
        echo "🧪 Đang chạy test..."
        python3 usage_example.py
        ;;
    4)
        echo ""
        echo "🔍 Kiểm tra models..."
        if [ -f "trained_models/class_model/class_model.pkl" ]; then
            echo "✅ Model lớp: Đã có"
        else
            echo "❌ Model lớp: Chưa có"
        fi
        
        if [ -f "trained_models/individual_model/individual_model.pkl" ]; then
            echo "✅ Model cá nhân: Đã có"
        else
            echo "❌ Model cá nhân: Chưa có"
        fi
        
        if [ -f "trained_models/clo_predictor/clo_predictor.pkl" ]; then
            echo "✅ Model CLO Predictor: Đã có"
        else
            echo "❌ Model CLO Predictor: Chưa có"
        fi
        ;;
    5)
        echo "👋 Tạm biệt!"
        exit 0
        ;;
    *)
        echo "❌ Lựa chọn không hợp lệ!"
        exit 1
        ;;
esac

