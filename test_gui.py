import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import tkinter as tk
from GUI.trang_chu import TrangChuGUI
from data.data_loader import load_all_sales_data

print("=" * 65)
print("KIỂM TRA GIAO DIỆN TKINTER VÀ CÁC CHỨC NĂNG TRÊN GIAO DIỆN")
print("=" * 65)

root = tk.Tk()
root.withdraw() # Ẩn cửa sổ để test tự động

app = TrangChuGUI(root)
print("  -> Khởi tạo TrangChuGUI thành công.")

# Tải dữ liệu tự động
df = load_all_sales_data()
app.data = df
app.sidebar.update_data(df)
print(f"  -> Nạp dữ liệu vào GUI: {len(app.data):,} dòng.")

# Kiểm tra duyệt qua tất cả các phím chức năng trên sidebar
keys_to_test = list(app.main_content.function_map.keys())
print(f"  -> Bắt đầu kiểm tra {len(keys_to_test)} chức năng trên giao diện:")

success_count = 0
for key in keys_to_test:
    try:
        app.handle_sidebar_select(key)
        success_count += 1
        print(f"     [OK] {key}")
    except Exception as e:
        print(f"     [ERROR] {key}: {e}")
        raise e

root.destroy()
print(f"\n✅ ĐÃ KIỂM TRA TOÀN BỘ {success_count}/{len(keys_to_test)} CHỨC NĂNG GIAO DIỆN THÀNH CÔNG!")
