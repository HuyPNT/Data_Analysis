import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
from data import data_loader
from FUNCTION import analysis_functions
from CHARTS import chart_functions
from CHARTS import advanced_charts

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class MainContentGUI(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg='#e8f5e9')
        self.chart_canvas = None

        # Tiêu đề
        self.label = tk.Label(
            self,
            text="Kết quả phân tích doanh thu",
            font=('Arial', 14, 'bold'),
            bg='#e8f5e9', fg='#388e3c'
        )
        self.label.pack(pady=10, fill='x')

        # Khu vực hiển thị văn bản với thanh cuộn
        text_frame = tk.Frame(self, bg='#e8f5e9')
        text_frame.pack(padx=3, pady=10, fill='both', expand=True)

        self.result_text = tk.Text(
            text_frame, wrap=tk.WORD, width=75, height=10,
            bg='#f1f8e9', font=('Courier', 11), bd=1, relief='solid'
        )
        self.result_text.pack(side='left', fill='both', expand=True)

        scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=self.result_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.result_text.config(yscrollcommand=scrollbar.set)

        # =====================================================
        #  Ánh xạ chức năng: key -> {analysis, chart, format}
        # =====================================================
        self.function_map = {
            # ---- PHÂN TÍCH CƠ BẢN (giữ nguyên) ----
            "revenue": {
                "analysis": analysis_functions.calculate_total_revenue,
                "chart": chart_functions.plot_total_revenue_bar,
                "format": lambda x: f"Tổng doanh thu: {x:,.0f} VNĐ"
            },
            "monthly": {
                "analysis": analysis_functions.revenue_by_month,
                "chart": chart_functions.plot_monthly_revenue,
                "format": self.format_dataframe
            },
            "predict": {
                "analysis": analysis_functions.predict_revenue,
                "chart": chart_functions.plot_predicted_revenue,
                "format": lambda x: f"Doanh thu dự đoán (trung bình): {x:,.0f} VNĐ"
            },
            "top_products": {
                "analysis": analysis_functions.top_5_best_selling,
                "chart": chart_functions.plot_top_5_products,
                "format": self.format_dataframe
            },
            "bottom_products": {
                "analysis": analysis_functions.bottom_5_worst_selling,
                "chart": chart_functions.plot_bottom_5_products,
                "format": self.format_dataframe
            },
            "by_city": {
                "analysis": analysis_functions.revenue_by_city,
                "chart": chart_functions.plot_revenue_by_city,
                "format": self.format_dataframe
            },
            "by_hour": {
                "analysis": analysis_functions.revenue_by_hour,
                "chart": chart_functions.plot_revenue_by_hour,
                "format": self.format_dataframe
            },
            "top_days": {
                "analysis": analysis_functions.top_revenue_day,
                "chart": chart_functions.plot_top_revenue_days,
                "format": self.format_dataframe
            },
            "top_revenue_products": {
                "analysis": analysis_functions.top_5_highest_revenue_products,
                "chart": chart_functions.plot_top_revenue_products,
                "format": self.format_dataframe
            },
            "bottom_revenue_products": {
                "analysis": analysis_functions.bottom_5_lowest_revenue_products,
                "chart": chart_functions.plot_bottom_revenue_products,
                "format": self.format_dataframe
            },
            "payment_method": {
                "analysis": analysis_functions.most_used_payment_method,
                "chart": advanced_charts.plot_payment_methods,
                "format": self.format_dataframe
            },

            # ---- CHẤT LƯỢNG DỮ LIỆU ----
            "data_quality": {
                "analysis": self._analysis_data_quality,
                "chart": advanced_charts.plot_data_quality_report,
                "format": self._format_data_quality
            },
            "outlier_detection": {
                "analysis": self._analysis_outlier,
                "chart": advanced_charts.plot_outliers,
                "format": self._format_outlier
            },

            # ---- LUẬT KẾT HỢP ----
            "association_rules": {
                "analysis": self._analysis_association_rules,
                "chart": advanced_charts.plot_association_rules,
                "format": self.format_dataframe
            },
            "frequent_items": {
                "analysis": self._analysis_frequent_items,
                "chart": advanced_charts.plot_frequent_itemsets,
                "format": self.format_dataframe
            },
            "product_recommendations": {
                "analysis": self._analysis_product_recommendations,
                "chart": advanced_charts.plot_product_recommendations,
                "format": self.format_dataframe
            },

            # ---- KHÁCH HÀNG ----
            "rfm_analysis": {
                "analysis": self._analysis_rfm,
                "chart": advanced_charts.plot_rfm_analysis,
                "format": self.format_dataframe
            },
            "customer_segments": {
                "analysis": self._analysis_customer_segments,
                "chart": advanced_charts.plot_customer_segments,
                "format": self._format_customer_segments
            },
            "customer_value": {
                "analysis": self._analysis_customer_value,
                "chart": advanced_charts.plot_customer_value,
                "format": self.format_dataframe
            },

            # ---- PHÂN TÍCH VỊ TRÍ ----
            "revenue_region": {
                "analysis": self._analysis_revenue_region,
                "chart": advanced_charts.plot_revenue_by_region,
                "format": self.format_dataframe
            },
            "products_region": {
                "analysis": self._analysis_products_region,
                "chart": advanced_charts.plot_products_by_region,
                "format": self.format_dataframe
            },
            "payment_region": {
                "analysis": self._analysis_payment_region,
                "chart": advanced_charts.plot_payment_by_region,
                "format": self.format_dataframe
            },

            # ---- DỰ ĐOÁN (ML) ----
            "predict_ml": {
                "analysis": self._analysis_predict_ml,
                "chart": advanced_charts.plot_predict_ml,
                "format": self._format_predict_ml
            },
            "product_trends": {
                "analysis": self._analysis_product_trends,
                "chart": advanced_charts.plot_product_trends,
                "format": self.format_dataframe
            },
            "seasonal": {
                "analysis": self._analysis_seasonal,
                "chart": advanced_charts.plot_seasonal_analysis,
                "format": self.format_dataframe
            },
            "anomaly_detection": {
                "analysis": self._analysis_anomaly,
                "chart": advanced_charts.plot_anomaly_detection,
                "format": self.format_dataframe
            },

            # ---- HỖ TRỢ QUYẾT ĐỊNH ----
            "employee_perf": {
                "analysis": self._analysis_employee_perf,
                "chart": advanced_charts.plot_employee_performance,
                "format": self.format_dataframe
            },
            "inventory_alerts": {
                "analysis": self._analysis_inventory,
                "chart": advanced_charts.plot_inventory_alerts,
                "format": self.format_dataframe
            },
            "cross_sell": {
                "analysis": self._analysis_cross_sell,
                "chart": advanced_charts.plot_cross_sell,
                "format": self.format_dataframe
            },
            "kpi_dashboard": {
                "analysis": self._analysis_kpi,
                "chart": advanced_charts.plot_kpi_dashboard,
                "format": self._format_kpi
            },
        }

    # =====================================================
    #  Wrapper analysis functions - gọi module tương ứng
    # =====================================================

    def _analysis_data_quality(self, df):
        from FUNCTION.data_cleaning import data_quality_report
        return data_quality_report(df)

    def _analysis_outlier(self, df):
        from FUNCTION.data_cleaning import detect_outliers_iqr
        outliers, lower, upper = detect_outliers_iqr(df, 'Thành tiền')
        return {'outliers': outliers, 'lower': lower, 'upper': upper}

    def _analysis_association_rules(self, df):
        from FUNCTION.association_rules import generate_rules
        return generate_rules(df)

    def _analysis_frequent_items(self, df):
        from FUNCTION.association_rules import get_top_product_pairs
        return get_top_product_pairs(df, top_n=15)

    def _analysis_product_recommendations(self, df):
        from FUNCTION.association_rules import generate_rules
        rules = generate_rules(df)
        if rules is not None and not rules.empty:
            return rules.head(15)
        import pandas as pd
        return pd.DataFrame(columns=['antecedents', 'consequents', 'confidence', 'lift'])

    def _analysis_rfm(self, df):
        from FUNCTION.customer_analysis import calculate_rfm
        return calculate_rfm(df)

    def _analysis_customer_segments(self, df):
        from FUNCTION.customer_analysis import segment_customers
        rfm_seg, summary = segment_customers(df)
        return {'segments': rfm_seg, 'summary': summary}

    def _analysis_customer_value(self, df):
        from FUNCTION.customer_analysis import customer_lifetime_value
        return customer_lifetime_value(df)

    def _analysis_revenue_region(self, df):
        from FUNCTION.location_analysis import revenue_by_region
        return revenue_by_region(df)

    def _analysis_products_region(self, df):
        from FUNCTION.location_analysis import popular_products_by_region
        return popular_products_by_region(df, top_n=5)

    def _analysis_payment_region(self, df):
        from FUNCTION.location_analysis import payment_by_region
        return payment_by_region(df)

    def _analysis_predict_ml(self, df):
        from FUNCTION.prediction import predict_revenue_linear
        predictions, metrics = predict_revenue_linear(df, months_ahead=3)
        return {'predictions': predictions, 'metrics': metrics}

    def _analysis_product_trends(self, df):
        from FUNCTION.prediction import product_trend_analysis
        return product_trend_analysis(df, top_n=10)

    def _analysis_seasonal(self, df):
        from FUNCTION.prediction import seasonal_analysis
        return seasonal_analysis(df)

    def _analysis_anomaly(self, df):
        from FUNCTION.prediction import detect_anomalies
        return detect_anomalies(df)

    def _analysis_employee_perf(self, df):
        from FUNCTION.decision_support import employee_performance
        return employee_performance(df)

    def _analysis_inventory(self, df):
        from FUNCTION.decision_support import inventory_alerts
        return inventory_alerts(df)

    def _analysis_cross_sell(self, df):
        from FUNCTION.decision_support import cross_sell_recommendations
        return cross_sell_recommendations(df, top_n=10)

    def _analysis_kpi(self, df):
        from FUNCTION.decision_support import kpi_summary
        return kpi_summary(df)

    # =====================================================
    #  Format functions
    # =====================================================

    def _format_data_quality(self, report):
        if not report:
            return "Không có dữ liệu"
        lines = [
            "═══════════════════════════════════════════════",
            "           BÁO CÁO CHẤT LƯỢNG DỮ LIỆU",
            "═══════════════════════════════════════════════",
            f"  📊 Tổng số dòng:         {report.get('total_rows', 0):>10,}",
            f"  📋 Tổng số cột:          {report.get('total_columns', 0):>10}",
            f"  🔁 Giao dịch trùng:      {report.get('duplicates', 0):>10,}",
            f"  ⚠️  Outliers (Thành tiền): {report.get('outliers_count', 0):>10,}",
            "",
            "  📉 Giá trị thiếu theo cột:",
        ]
        for col, count in report.get('missing_values', {}).items():
            pct = report.get('missing_percentage', {}).get(col, 0)
            status = "✅" if count == 0 else "❌"
            lines.append(f"     {status} {col:<25} {count:>5} ({pct:.1f}%)")

        lines.append("")
        lines.append("  🔢 Số giá trị duy nhất:")
        for col, count in report.get('unique_values', {}).items():
            lines.append(f"     • {col:<25} {count:>5}")

        return "\n".join(lines)

    def _format_outlier(self, result):
        if not result:
            return "Không có dữ liệu"
        outliers = result['outliers']
        lower = result['lower']
        upper = result['upper']
        lines = [
            "═══════════════════════════════════════════════",
            "         PHÁT HIỆN GIÁ TRỊ NGOẠI LỆ",
            "═══════════════════════════════════════════════",
            f"  📏 Ngưỡng dưới (IQR): {lower:>15,.0f} VNĐ",
            f"  📏 Ngưỡng trên (IQR): {upper:>15,.0f} VNĐ",
            f"  ⚠️  Số outlier:        {len(outliers):>15,}",
            "",
        ]
        if not outliers.empty:
            lines.append("  Top 10 giao dịch bất thường:")
            top = outliers.nlargest(10, 'Thành tiền')
            for _, row in top.iterrows():
                lines.append(
                    f"    • {row.get('Mã giao dịch', 'N/A')} | "
                    f"{row.get('Tên mặt hàng', 'N/A')} | "
                    f"{row.get('Thành tiền', 0):,.0f} VNĐ"
                )
        else:
            lines.append("  ✅ Không phát hiện outlier nào!")
        return "\n".join(lines)

    def _format_customer_segments(self, result):
        if not result:
            return "Không có dữ liệu"
        segments = result.get('segments')
        summary = result.get('summary')
        lines = [
            "═══════════════════════════════════════════════",
            "         PHÂN KHÚC KHÁCH HÀNG (K-MEANS)",
            "═══════════════════════════════════════════════",
        ]
        if summary is not None and not summary.empty:
            lines.append("")
            lines.append("  Tổng hợp theo phân khúc:")
            lines.append(self.format_dataframe(summary))
        if segments is not None and not segments.empty:
            lines.append("")
            seg_counts = segments['Phân khúc'].value_counts()
            lines.append("  Phân bổ:")
            for seg, count in seg_counts.items():
                lines.append(f"    • {seg}: {count} khách hàng")
        return "\n".join(lines)

    def _format_predict_ml(self, result):
        if not result:
            return "Không có dữ liệu"
        predictions = result.get('predictions')
        metrics = result.get('metrics', {})
        lines = [
            "═══════════════════════════════════════════════",
            "       DỰ ĐOÁN DOANH THU (LINEAR REGRESSION)",
            "═══════════════════════════════════════════════",
            f"  📊 R² Score:    {metrics.get('r2_score', 0):.4f}",
            f"  📉 MAE:         {metrics.get('mae', 0):,.0f} VNĐ",
            f"  📈 Xu hướng:    {metrics.get('trend', 'N/A')}",
            "",
        ]
        if predictions is not None and not predictions.empty:
            lines.append("  Dự đoán theo tháng:")
            lines.append(self.format_dataframe(predictions))
        return "\n".join(lines)

    def _format_kpi(self, kpi):
        if not kpi:
            return "Không có dữ liệu"
        lines = [
            "═══════════════════════════════════════════════",
            "              TỔNG HỢP KPI",
            "═══════════════════════════════════════════════",
            f"  💰 Tổng doanh thu:       {kpi.get('tong_doanh_thu', 0):>15,.0f} VNĐ",
            f"  🧾 Số giao dịch:         {kpi.get('so_giao_dich', 0):>15,}",
            f"  💵 Giá trị TB/GD:        {kpi.get('gia_tri_tb_gd', 0):>15,.0f} VNĐ",
            f"  👥 Số khách hàng:        {kpi.get('so_khach_hang', 0):>15,}",
            f"  📦 Số sản phẩm:          {kpi.get('so_san_pham', 0):>15}",
            f"  🏆 SP bán chạy nhất:     {kpi.get('sp_ban_chay_nhat', 'N/A')}",
            f"  💎 SP doanh thu cao nhất:{kpi.get('sp_doanh_thu_cao_nhat', 'N/A')}",
            f"  🏙️  TP doanh thu cao nhất:{kpi.get('thanh_pho_dt_cao_nhat', 'N/A')}",
            f"  💳 PT thanh toán phổ biến:{kpi.get('pt_thanh_toan_pho_bien', 'N/A')}",
            f"  🕐 Giờ bán chạy nhất:   {kpi.get('gio_ban_chay_nhat', 'N/A')}",
            f"  📅 Tháng DT cao nhất:    {kpi.get('thang_dt_cao_nhat', 'N/A')}",
            f"  📈 Tăng trưởng DT:       {kpi.get('tang_truong_dt', 0):>+14.1f}%",
            f"  🔄 Tỷ lệ KH quay lại:   {kpi.get('ty_le_kh_quay_lai', 0):>14.1f}%",
        ]
        return "\n".join(lines)

    def format_dataframe(self, df):
        import pandas as pd
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return "Không có dữ liệu"
        if not isinstance(df, pd.DataFrame):
            return str(df)

        # Thêm định dạng số với dấu phẩy để ngăn cách
        formatted_df = df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype in ['int64', 'float64']:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")

        # Đổi tên cột
        if 'Product' in formatted_df.columns:
            formatted_df = formatted_df.rename(columns={'Product': 'Sản phẩm'})

        COLUMN_PADDING = 4  # Khoảng cách giữa các cột

        # Xác định độ rộng max cho từng cột
        col_widths = {}
        for col in formatted_df.columns:
            max_width = max(
                len(str(col)),
                max(len(str(val)) for val in formatted_df[col]) if not formatted_df.empty else 0
            )
            col_widths[col] = max_width

        # Tạo chuỗi header
        header = []
        for col in formatted_df.columns:
            header.append(str(col).ljust(col_widths[col] + COLUMN_PADDING))
        header_str = ''.join(header)

        # Tạo separator
        sep_str = '-' * len(header_str)

        # Tạo các dòng dữ liệu
        rows = []
        for _, row in formatted_df.iterrows():
            row_str = []
            for col in formatted_df.columns:
                cell_value = str(row[col])
                row_str.append(cell_value.ljust(col_widths[col] + COLUMN_PADDING))
            rows.append(''.join(row_str))

        result = header_str + '\n' + sep_str + '\n' + '\n'.join(rows)
        return result

    def display_result(self, key, data=None):
        """Hiển thị kết quả văn bản và biểu đồ dựa trên key."""
        logging.info(f"Gọi display_result với key: {key}, data: {'rỗng' if data is None else len(data)} dòng")
        self.result_text.delete(1.0, tk.END)
        if data is None or (hasattr(data, 'empty') and data.empty):
            self.result_text.insert(tk.END, "Vui lòng chọn file dữ liệu hợp lệ trước.")
            if key is not None:
                messagebox.showwarning("Chưa có dữ liệu", "Vui lòng chọn file dữ liệu hợp lệ trước.")
            return

        try:
            if key not in self.function_map:
                self.result_text.insert(tk.END, "Chức năng chưa được hỗ trợ.")
                return

            title_map = {
                "kpi_dashboard": "📊 Tổng Hợp KPI Toàn Cảnh Hoạt Động",
                "monthly": "📅 Doanh Thu Theo Tháng",
                "predict": "🔮 Dự Đoán Doanh Thu (Trung Bình)",
                "top_products": "🏆 Top 5 Sản Phẩm Bán Chạy",
                "bottom_products": "⚠️ Top 5 Sản Phẩm Bán Chậm",
                "by_city": "🏙️ Doanh Thu Theo Thành Phố",
                "by_hour": "⏰ Doanh Thu Theo Giờ",
                "top_days": "🗓️ Ngày Doanh Thu Cao Nhất",
                "top_revenue_products": "💎 Top 5 Sản Phẩm Doanh Thu Cao",
                "bottom_revenue_products": "📉 Top 5 Sản Phẩm Doanh Thu Thấp",
                "payment_method": "💳 Phân Tích Phương Thức Thanh Toán",
                "data_quality": "📋 Báo Cáo Chất Lượng Dữ Liệu",
                "outlier_detection": "⚠️ Phát Hiện Giá Trị Ngoại Lệ (Outliers)",
                "association_rules": "🔗 Phân Tích Luật Kết Hợp (Apriori)",
                "frequent_items": "🛒 Sản Phẩm Hay Mua Cùng Nhau",
                "product_recommendations": "💡 Gợi Ý Sản Phẩm Bán Kèm",
                "rfm_analysis": "👥 Phân Tích Khách Hàng (RFM)",
                "customer_segments": "🎯 Phân Khúc Khách Hàng (K-Means)",
                "customer_value": "💰 Giá Trị Vòng Đời Khách Hàng (CLV)",
                "revenue_region": "🗺️ Doanh Thu Theo Vùng Miền",
                "products_region": "📦 Sản Phẩm Phổ Biến Theo Vùng Miền",
                "payment_region": "💳 Phương Thức Thanh Toán Theo Vùng Miền",
                "predict_ml": "📈 Dự Đoán Doanh Thu Bằng Machine Learning",
                "product_trends": "📊 Phân Tích Xu Hướng Mặt Hàng",
                "seasonal": "☀️ Phân Tích Mùa Vụ Kinh Doanh",
                "anomaly_detection": "🚨 Phát Hiện Biến Động Doanh Thu Bất Thường",
                "employee_perf": "👨‍💼 Đánh Giá Hiệu Suất Nhân Viên",
                "inventory_alerts": "📦 Báo Cáo Tồn Kho",
                "cross_sell": "🛍️ Gợi Ý Bán Chéo (Cross-Sell)",
            }
            if key in title_map:
                self.label.config(text=title_map[key])

            # Lấy hàm phân tích và định dạng
            func_info = self.function_map[key]
            result = func_info["analysis"](data)
            logging.info(f"Kết quả phân tích cho {key}: hoàn thành")
            formatted_result = func_info["format"](result)

            # Hiển thị văn bản
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, formatted_result)

            # Hiển thị biểu đồ
            chart_func = func_info.get("chart")
            if chart_func:
                fig = chart_func(data)
                if fig:
                    self.display_chart(fig)
                else:
                    logging.warning(f"Không tạo được biểu đồ cho key: {key}")

        except Exception as e:
            logging.error(f"Lỗi trong display_result ({key}): {str(e)}")
            import traceback
            traceback.print_exc()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Đã xảy ra lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

    def display_chart(self, fig):
        """Hiển thị biểu đồ trong giao diện."""
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(pady=10, fill='both', expand=True)
