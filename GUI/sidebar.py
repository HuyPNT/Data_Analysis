import tkinter as tk
from tkinter import ttk

class SidebarGUI(tk.Frame):
    def __init__(self, parent, on_menu_click):
        super().__init__(parent, bg='#2c3e50')
        self.on_menu_click = on_menu_click
        self.data = None

        # Tạo canvas + scrollbar để sidebar cuộn được
        self.canvas = tk.Canvas(self, bg='#2c3e50', highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#2c3e50')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Hỗ trợ cuộn bằng chuột
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # =====================================================
        #  Danh sách chức năng theo nhóm
        # =====================================================
        menu_groups = [
            ("📊 PHÂN TÍCH CƠ BẢN", [
                ("Tổng Hợp KPI", "kpi_dashboard"),
                ("Doanh Thu Theo Tháng", "monthly"),
                ("Top 5 SP Bán Chạy", "top_products"),
                ("Top 5 SP Kém Bán", "bottom_products"),
                ("Doanh Thu Theo TP", "by_city"),
                ("Doanh Thu Theo Giờ", "by_hour"),
                ("Ngày Doanh Thu Cao", "top_days"),
                ("Top 5 SP DT Cao", "top_revenue_products"),
                ("Top 5 SP DT Thấp", "bottom_revenue_products"),
                ("PT Thanh Toán", "payment_method"),
            ]),
            ("🧹 CHẤT LƯỢNG DỮ LIỆU", [
                ("Báo Cáo Chất Lượng", "data_quality"),
                ("Phát Hiện Outlier", "outlier_detection"),
            ]),
            ("🔗 LUẬT KẾT HỢP", [
                ("Luật Kết Hợp", "association_rules"),
                ("SP Hay Mua Cùng", "frequent_items"),
                ("Gợi Ý Sản Phẩm", "product_recommendations"),
            ]),
            ("👥 KHÁCH HÀNG", [
                ("Phân Tích RFM", "rfm_analysis"),
                ("Phân Khúc KH", "customer_segments"),
                ("Giá Trị Vòng Đời KH", "customer_value"),
            ]),
            ("📍 PHÂN TÍCH VỊ TRÍ", [
                ("DT Theo Vùng Miền", "revenue_region"),
                ("SP Theo Vùng Miền", "products_region"),
                ("Thanh Toán Theo Vùng", "payment_region"),
            ]),
            ("📈 DỰ ĐOÁN (ML)", [
                ("Dự Đoán Doanh Thu", "predict_ml"),
                ("Xu Hướng Sản Phẩm", "product_trends"),
                ("Phân Tích Mùa Vụ", "seasonal"),
                ("Phát Hiện Bất Thường", "anomaly_detection"),
            ]),
            ("💡 HỖ TRỢ QUYẾT ĐỊNH", [
                ("Hiệu Suất Nhân Viên", "employee_perf"),
                ("Báo Cáo Tồn Kho", "inventory_alerts"),
                ("Gợi Ý Cross-Sell", "cross_sell"),
            ]),
        ]

        for group_title, buttons in menu_groups:
            self.create_section(group_title, buttons)

    def create_section(self, title, buttons):
        """Tạo một nhóm chức năng với tiêu đề."""
        # Separator
        sep = ttk.Separator(self.scrollable_frame, orient='horizontal')
        sep.pack(fill='x', pady=(8, 2), padx=5)

        # Section title
        tk.Label(
            self.scrollable_frame, text=title,
            font=('Arial', 10, 'bold'), bg='#2c3e50', fg='#1abc9c',
            anchor='w'
        ).pack(fill='x', padx=10, pady=(4, 2))

        # Buttons
        for text, key in buttons:
            self.create_button(text, key)

    def create_button(self, text, key):
        button = tk.Button(
            self.scrollable_frame,
            text=text,
            font=('Arial', 10),
            bg='#34495e', fg='white',
            activebackground='#388e3c', activeforeground='white',
            relief='flat',
            command=lambda: self.on_menu_click(key) if self.data is not None and not self.data.empty else self.show_warning()
        )
        button.pack(fill='x', padx=10, pady=2)
        # Hiệu ứng hover
        button.bind("<Enter>", lambda e: button.config(bg='#3f5a7a'))
        button.bind("<Leave>", lambda e: button.config(bg='#34495e'))

    def update_data(self, data):
        """Cập nhật trạng thái dữ liệu."""
        self.data = data

    def show_warning(self):
        """Hiển thị cảnh báo nếu chưa có dữ liệu."""
        from tkinter import messagebox
        messagebox.showwarning("Chưa có dữ liệu", "Vui lòng chọn file CSV trước!")