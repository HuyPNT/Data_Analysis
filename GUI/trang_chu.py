import tkinter as tk
from tkinter import messagebox, filedialog
import os
from GUI.sidebar import SidebarGUI
from GUI.maincontent import MainContentGUI
from data import data_loader

class TrangChuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phân Tích Dữ Liệu Bán Hàng")
        self.root.geometry("1300x800")
        self.root.configure(bg='#f0f2f5')

        # Biến lưu trữ dữ liệu
        self.data = None

        # =====================================================
        #  Thanh tiêu đề + nút
        # =====================================================
        header_frame = tk.Frame(self.root, bg='#1a237e')
        header_frame.pack(side="top", fill="x")

        tk.Label(
            header_frame,
            text="📊 Phân Tích Dữ Liệu Bán Hàng",
            font=("Arial", 16, "bold"), fg="white", bg='#1a237e'
        ).pack(side="left", padx=20, pady=10)

        # Frame chứa các nút
        btn_frame = tk.Frame(header_frame, bg='#1a237e')
        btn_frame.pack(side="right", padx=10, pady=5)

        tk.Button(
            btn_frame, text="📂 Chọn File CSV",
            command=self.load_file,
            bg="#3498db", fg="white", font=("Arial", 10),
            relief='flat', padx=10
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="📁 Nạp Toàn Bộ Data",
            command=self.load_all_files,
            bg="#27ae60", fg="white", font=("Arial", 10),
            relief='flat', padx=10
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame, text="🧹 Làm Sạch Dữ Liệu",
            command=self.clean_data,
            bg="#e67e22", fg="white", font=("Arial", 10),
            relief='flat', padx=10
        ).pack(side="left", padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Chưa có dữ liệu - Vui lòng nạp file CSV")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Arial", 9), bg='#ecf0f1', fg='#2c3e50',
            anchor='w', padx=10
        )
        status_bar.pack(side="bottom", fill="x")

        # =====================================================
        #  Sidebar + Content
        # =====================================================
        # Sidebar frame
        self.sidebar_frame = tk.Frame(self.root, bg='#2c3e50', width=220)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)

        # Content frame
        self.content_frame = tk.Frame(self.root, bg='#e8f5e9')
        self.content_frame.pack(side="right", fill="both", expand=True)

        # Khởi tạo main content và sidebar
        self.main_content = MainContentGUI(self.content_frame)
        self.main_content.pack(fill="both", expand=True)

        self.sidebar = SidebarGUI(self.sidebar_frame, self.handle_sidebar_select)
        self.sidebar.pack(fill="both", expand=True)

        # Tự động nạp dữ liệu và mặc định hiển thị "Tổng hợp KPI" khi vào trang chủ
        try:
            auto_data = data_loader.load_all_sales_data()
            if auto_data is not None and not auto_data.empty:
                self.data = auto_data
                self.sidebar.update_data(self.data)
                self.status_var.set(
                    f"✅ Đã tự động nạp dữ liệu ({len(self.data):,} dòng) | 12 tháng"
                )
                self.main_content.display_result('kpi_dashboard', self.data)
            else:
                self.main_content.display_result('kpi_dashboard', None)
        except Exception:
            self.main_content.display_result('kpi_dashboard', None)

    def load_file(self):
        """Cho phép người dùng chọn nhiều file CSV."""
        file_paths = filedialog.askopenfilenames(
            title="Chọn các file dữ liệu",
            filetypes=[("CSV Files", "*.csv")]
        )

        if not file_paths:
            messagebox.showinfo("Thông báo", "Chưa chọn file nào.")
            return

        try:
            self.data = data_loader.load_sales_data(file_paths)

            if self.data.empty:
                self.data = None
                raise ValueError("Không có dữ liệu hợp lệ trong các file đã chọn.")

            self.sidebar.update_data(self.data)
            self.main_content.display_result('kpi_dashboard', self.data)

            file_names = ", ".join([os.path.basename(fp) for fp in file_paths])
            self.status_var.set(
                f"✅ Đã nạp {len(file_paths)} file ({len(self.data):,} dòng) | {file_names}"
            )
            messagebox.showinfo("Thành công", f"Đã nạp {len(file_paths)} file: {file_names}")

        except Exception as e:
            self.data = None
            self.sidebar.update_data(None)
            self.main_content.display_result(None, None)
            self.status_var.set(f"❌ Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể nạp dữ liệu: {str(e)}")

    def load_all_files(self):
        """Tự động nạp toàn bộ 12 file CSV từ thư mục sales_data."""
        try:
            self.data = data_loader.load_all_sales_data()

            if self.data is None or self.data.empty:
                self.data = None
                raise ValueError("Không tìm thấy hoặc không có dữ liệu hợp lệ.")

            self.sidebar.update_data(self.data)
            self.main_content.display_result('kpi_dashboard', self.data)

            self.status_var.set(
                f"✅ Đã nạp toàn bộ dữ liệu ({len(self.data):,} dòng) | 12 tháng"
            )
            messagebox.showinfo("Thành công",
                f"Đã nạp toàn bộ 12 tháng dữ liệu!\n"
                f"Tổng: {len(self.data):,} giao dịch")

        except Exception as e:
            self.data = None
            self.sidebar.update_data(None)
            self.main_content.display_result(None, None)
            self.status_var.set(f"❌ Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể nạp dữ liệu: {str(e)}")

    def clean_data(self):
        """Chạy pipeline làm sạch dữ liệu nâng cao."""
        if self.data is None or self.data.empty:
            messagebox.showwarning("Chưa có dữ liệu", "Vui lòng nạp file CSV trước!")
            return

        try:
            from FUNCTION.data_cleaning import full_clean_pipeline
            before_rows = len(self.data)
            self.data, report = full_clean_pipeline(self.data)
            after_rows = len(self.data)

            self.sidebar.update_data(self.data)

            self.status_var.set(
                f"🧹 Đã làm sạch: {before_rows:,} → {after_rows:,} dòng | "
                f"Xóa {before_rows - after_rows:,} dòng lỗi/trùng"
            )
            messagebox.showinfo("Làm sạch thành công",
                f"Trước: {before_rows:,} dòng\n"
                f"Sau:   {after_rows:,} dòng\n"
                f"Xóa:   {before_rows - after_rows:,} dòng\n\n"
                f"Outliers: {report.get('outliers_count', 0)}\n"
                f"Trùng lặp: {report.get('duplicates', 0)}")

            # Tự động hiển thị báo cáo chất lượng
            self.main_content.display_result('data_quality', self.data)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi làm sạch dữ liệu: {str(e)}")

    def handle_sidebar_select(self, key):
        self.main_content.display_result(key, self.data)