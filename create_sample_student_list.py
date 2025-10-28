#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo file mẫu danh sách sinh viên
"""

import pandas as pd

# Tạo dữ liệu mẫu
data = {
    'MSSV': [
        'SV2021001', 'SV2021002', 'SV2021003', 'SV2021004', 'SV2021005',
        'SV2021006', 'SV2021007', 'SV2021008', 'SV2021009', 'SV2021010',
        'SV2021011', 'SV2021012', 'SV2021013', 'SV2021014', 'SV2021015'
    ],
    'HoTen': [
        'Nguyễn Văn An', 'Trần Thị Bình', 'Lê Văn Cường', 'Phạm Thị Dung', 'Hoàng Văn Em',
        'Vũ Thị Phương', 'Đỗ Văn Giang', 'Bùi Thị Hoa', 'Ngô Văn Ích', 'Đinh Thị Kim',
        'Mai Văn Long', 'Phan Thị Mai', 'Đặng Văn Nam', 'Võ Thị Oanh', 'Lý Văn Phúc'
    ],
    'DiemCLO': [
        5.8, 5.2, 4.5, 3.8, 2.5,
        4.8, 5.5, 3.2, 4.0, 2.8,
        5.0, 4.2, 3.5, 5.6, 4.8
    ]
}

df = pd.DataFrame(data)

# Lưu thành file Excel
excel_file = 'danh_sach_sinh_vien_mau.xlsx'
df.to_excel(excel_file, index=False)
print(f"✅ Đã tạo file mẫu: {excel_file}")
print(f"📊 Số sinh viên: {len(df)}")
print(f"\n📋 Xem trước:")
print(df.head(10).to_string(index=False))

# Lưu thành file CSV
csv_file = 'danh_sach_sinh_vien_mau.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"\n✅ Đã tạo file CSV: {csv_file}")

print("\n💡 Bạn có thể sử dụng file này để test:")
print(f"   python3 run_interactive_with_file.py")
print(f"   → Chọn chế độ 1 (File)")
print(f"   → Nhập đường dẫn: {excel_file} (hoặc {csv_file})")

