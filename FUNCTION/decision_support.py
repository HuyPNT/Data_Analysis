"""
Module hỗ trợ ra quyết định kinh doanh (Decision Support System).
Cung cấp các phân tích thực tế về bán chéo, cảnh báo tồn kho, hiệu suất nhân viên,
đề xuất chương trình khuyến mãi, bảng điều khiển KPI và các nhận định kinh doanh tự động.
"""

import logging
from typing import List, Dict, Any
from itertools import permutations
from collections import defaultdict
import numpy as np
import pandas as pd

# Thiết lập logging
logger = logging.getLogger(__name__)


def cross_sell_recommendations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Phân tích và gợi ý các cặp sản phẩm bán chéo (Cross-selling) dựa trên hành vi
    mua sắm của cùng một khách hàng.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Mã khách hàng', 'Tên mặt hàng'.
        top_n (int): Số lượng cặp gợi ý hàng đầu cần trả về (mặc định là 10).

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            ['Sản phẩm chính', 'Sản phẩm gợi ý', 'Số KH mua cùng', 'Tỷ lệ mua cùng (%)']
    """
    empty_df = pd.DataFrame(columns=['Sản phẩm chính', 'Sản phẩm gợi ý', 'Số KH mua cùng', 'Tỷ lệ mua cùng (%)'])

    if df is None or df.empty:
        logger.warning("cross_sell_recommendations: DataFrame rỗng.")
        return empty_df

    required_cols = {'Mã khách hàng', 'Tên mặt hàng'}
    if not required_cols.issubset(df.columns):
        logger.error(f"cross_sell_recommendations: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        # Lọc các dòng có mã khách hàng hợp lệ (loại bỏ khách vãng lai/NaN)
        valid_df = df.dropna(subset=['Mã khách hàng', 'Tên mặt hàng']).copy()
        valid_df['Mã khách hàng'] = valid_df['Mã khách hàng'].astype(str).str.strip()
        valid_df['Tên mặt hàng'] = valid_df['Tên mặt hàng'].astype(str).str.strip()
        valid_df = valid_df[~valid_df['Mã khách hàng'].isin(['', 'nan', 'None', 'null'])]

        if valid_df.empty:
            logger.warning("cross_sell_recommendations: Không tìm thấy khách hàng định danh.")
            return empty_df

        # Gom danh sách các sản phẩm phân biệt mà mỗi khách hàng đã mua
        cust_prods = valid_df.groupby('Mã khách hàng')['Tên mặt hàng'].unique()
        # Chỉ giữ lại khách hàng mua từ 2 mặt hàng khác nhau trở lên
        multi_item_custs = [set(prods) for prods in cust_prods if len(prods) >= 2]

        if not multi_item_custs:
            logger.info("Không có khách hàng nào mua từ 2 mặt hàng trở lên.")
            return empty_df

        product_cust_count = defaultdict(int)
        pair_counts = defaultdict(int)

        for prods_set in multi_item_custs:
            for p in prods_set:
                product_cust_count[p] += 1
            for p1, p2 in permutations(prods_set, 2):
                pair_counts[(p1, p2)] += 1

        if not pair_counts:
            return empty_df

        records = []
        for (p1, p2), co_count in pair_counts.items():
            base_count = product_cust_count.get(p1, 1)
            co_rate = (co_count / base_count) * 100.0
            records.append({
                'Sản phẩm chính': p1,
                'Sản phẩm gợi ý': p2,
                'Số KH mua cùng': co_count,
                'Tỷ lệ mua cùng (%)': round(co_rate, 2)
            })

        rec_df = pd.DataFrame(records)
        rec_df = rec_df.sort_values(by=['Số KH mua cùng', 'Tỷ lệ mua cùng (%)'], ascending=[False, False])
        return rec_df.head(top_n).reset_index(drop=True)

    except Exception as e:
        logger.error(f"Lỗi trong cross_sell_recommendations: {e}")
        return empty_df


def inventory_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán tốc độ bán hàng theo tháng và phân loại mức cảnh báo hàng tồn kho:
    - 'Bán chạy - Cần nhập thêm'
    - 'Bán chậm - Giảm nhập'
    - 'Ổn định'

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Tên mặt hàng', 'Số lượng', 'Ngày bán'.

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            ['Tên mặt hàng', 'SL bán/tháng TB', 'Xu hướng', 'Cảnh báo']
    """
    empty_df = pd.DataFrame(columns=['Tên mặt hàng', 'SL bán/tháng TB', 'Xu hướng', 'Cảnh báo'])

    if df is None or df.empty:
        logger.warning("inventory_alerts: DataFrame rỗng.")
        return empty_df

    required_cols = {'Tên mặt hàng', 'Số lượng', 'Ngày bán'}
    if not required_cols.issubset(df.columns):
        logger.error(f"inventory_alerts: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        work_df = df[['Tên mặt hàng', 'Số lượng', 'Ngày bán']].copy()
        work_df['Ngày bán'] = pd.to_datetime(work_df['Ngày bán'], errors='coerce')
        work_df['Số lượng'] = pd.to_numeric(work_df['Số lượng'], errors='coerce')
        work_df = work_df.dropna(subset=['Tên mặt hàng', 'Số lượng', 'Ngày bán'])

        if work_df.empty:
            return empty_df

        work_df['Period'] = work_df['Ngày bán'].dt.to_period('M')
        all_periods = sorted(work_df['Period'].unique())
        n_periods = len(all_periods)

        # Pivot số lượng bán theo sản phẩm và tháng
        pivot_qty = work_df.pivot_table(
            index='Tên mặt hàng',
            columns='Period',
            values='Số lượng',
            aggfunc='sum',
            fill_value=0
        )

        if pivot_qty.empty:
            return empty_df

        mean_velocity = pivot_qty.mean(axis=1)
        current_velocity = pivot_qty.iloc[:, -1]

        trends = []
        alerts = []

        if n_periods >= 2:
            x = np.arange(1, n_periods + 1)
            x_diff = x - np.mean(x)
            x_denom = np.sum(x_diff ** 2)

            for prod, row in pivot_qty.iterrows():
                y = row.values.astype(float)
                y_mean = np.mean(y)
                y_diff = y - y_mean
                slope = float(np.sum(x_diff * y_diff) / x_denom) if x_denom != 0 else 0.0

                rel_slope = (slope / y_mean) * 100.0 if y_mean > 0 else 0.0
                curr_v = float(current_velocity.loc[prod])
                avg_v = float(mean_velocity.loc[prod])

                # Xác định xu hướng
                if rel_slope >= 5.0 or (curr_v >= avg_v * 1.25 and curr_v >= 5):
                    trend = 'Tăng'
                elif rel_slope <= -5.0 or (curr_v <= avg_v * 0.75):
                    trend = 'Giảm'
                else:
                    trend = 'Ổn định'
                trends.append(trend)

                # Phân loại cảnh báo tồn kho
                if trend == 'Tăng' and curr_v >= avg_v:
                    alert = 'Bán chạy - Cần nhập thêm'
                elif trend == 'Giảm' and curr_v <= avg_v:
                    alert = 'Bán chậm - Giảm nhập'
                elif curr_v >= avg_v * 1.3:
                    alert = 'Bán chạy - Cần nhập thêm'
                elif curr_v <= avg_v * 0.6:
                    alert = 'Bán chậm - Giảm nhập'
                else:
                    alert = 'Ổn định'
                alerts.append(alert)
        else:
            trends = ['Ổn định'] * len(pivot_qty)
            alerts = ['Ổn định'] * len(pivot_qty)

        result_df = pd.DataFrame({
            'Tên mặt hàng': pivot_qty.index,
            'SL bán/tháng TB': np.round(mean_velocity.values, 2),
            'Xu hướng': trends,
            'Cảnh báo': alerts
        })

        # Sắp xếp ưu tiên: Cần nhập thêm -> Giảm nhập -> Ổn định
        alert_order = {'Bán chạy - Cần nhập thêm': 0, 'Bán chậm - Giảm nhập': 1, 'Ổn định': 2}
        result_df['_order'] = result_df['Cảnh báo'].map(alert_order).fillna(3)
        result_df = result_df.sort_values(by=['_order', 'SL bán/tháng TB'], ascending=[True, False])
        result_df = result_df.drop(columns=['_order']).reset_index(drop=True)
        return result_df

    except Exception as e:
        logger.error(f"Lỗi trong inventory_alerts: {e}")
        return empty_df


def employee_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đánh giá hiệu suất nhân viên bán hàng: doanh thu, số giao dịch, giá trị TB đơn hàng,
    mặt hàng bán nhiều nhất, xếp hạng và xếp loại khen thưởng.

    Hạng đánh giá:
    - 'Xuất sắc': top 20%
    - 'Tốt': tiếp theo 30%
    - 'Trung bình': tiếp theo 30%
    - 'Cần cải thiện': 20% còn lại

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng chứa các cột 'Mã nhân viên', 'Thành tiền',
                           'Số lượng', 'Tên mặt hàng'.

    Returns:
        pd.DataFrame: DataFrame gồm các cột:
            ['Mã nhân viên', 'Doanh thu', 'Số GD', 'Giá trị TB', 'SP bán nhiều nhất', 'Xếp hạng', 'Đánh giá']
    """
    empty_df = pd.DataFrame(columns=[
        'Mã nhân viên', 'Doanh thu', 'Số GD', 'Giá trị TB', 'SP bán nhiều nhất', 'Xếp hạng', 'Đánh giá'
    ])

    if df is None or df.empty:
        logger.warning("employee_performance: DataFrame rỗng.")
        return empty_df

    required_cols = {'Mã nhân viên', 'Thành tiền', 'Số lượng', 'Tên mặt hàng'}
    if not required_cols.issubset(df.columns):
        logger.error(f"employee_performance: Thiếu cột bắt buộc: {required_cols - set(df.columns)}")
        return empty_df

    try:
        work_df = df.dropna(subset=['Mã nhân viên']).copy()
        work_df['Mã nhân viên'] = work_df['Mã nhân viên'].astype(str).str.strip()
        work_df = work_df[~work_df['Mã nhân viên'].isin(['', 'nan', 'None'])]

        if work_df.empty:
            return empty_df

        work_df['Thành tiền'] = pd.to_numeric(work_df['Thành tiền'], errors='coerce').fillna(0.0)
        work_df['Số lượng'] = pd.to_numeric(work_df['Số lượng'], errors='coerce').fillna(0)

        # Tổng hợp doanh thu và số giao dịch theo từng nhân viên
        emp_stats = work_df.groupby('Mã nhân viên').agg(
            Doanh_thu=('Thành tiền', 'sum'),
            So_GD=('Mã nhân viên', 'count')
        ).reset_index()

        emp_stats['Gia_tri_TB'] = emp_stats['Doanh_thu'] / emp_stats['So_GD']

        # Tìm sản phẩm bán được nhiều nhất của từng nhân viên
        emp_prod = work_df.groupby(['Mã nhân viên', 'Tên mặt hàng'])['Số lượng'].sum().reset_index()
        emp_top_prod = emp_prod.sort_values(by=['Mã nhân viên', 'Số lượng'], ascending=[True, False]).drop_duplicates(subset=['Mã nhân viên'])
        prod_map = dict(zip(emp_top_prod['Mã nhân viên'], emp_top_prod['Tên mặt hàng']))
        emp_stats['SP bán nhiều nhất'] = emp_stats['Mã nhân viên'].map(prod_map).fillna('N/A')

        # Xếp hạng theo doanh thu giảm dần
        emp_stats = emp_stats.sort_values(by='Doanh_thu', ascending=False).reset_index(drop=True)
        emp_stats['Xếp hạng'] = emp_stats.index + 1
        n_emp = len(emp_stats)

        # Phân loại hiệu suất: top 20%, 30%, 30%, 20%
        def assign_tier(rank: int, total: int) -> str:
            if total <= 1:
                return 'Xuất sắc'
            percentile = (rank - 1) / total
            if percentile < 0.20:
                return 'Xuất sắc'
            elif percentile < 0.50:
                return 'Tốt'
            elif percentile < 0.80:
                return 'Trung bình'
            else:
                return 'Cần cải thiện'

        emp_stats['Đánh giá'] = [assign_tier(r, n_emp) for r in emp_stats['Xếp hạng']]

        result_df = pd.DataFrame({
            'Mã nhân viên': emp_stats['Mã nhân viên'],
            'Doanh thu': np.round(emp_stats['Doanh_thu'], 2),
            'Số GD': emp_stats['So_GD'],
            'Giá trị TB': np.round(emp_stats['Gia_tri_TB'], 2),
            'SP bán nhiều nhất': emp_stats['SP bán nhiều nhất'],
            'Xếp hạng': emp_stats['Xếp hạng'],
            'Đánh giá': emp_stats['Đánh giá']
        })
        return result_df

    except Exception as e:
        logger.error(f"Lỗi trong employee_performance: {e}")
        return empty_df


def promotion_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Phân tích dữ liệu thực tế để đưa ra các gợi ý chương trình khuyến mãi:
    a. Giảm giá sản phẩm bán chậm để kích cầu giải phóng tồn kho.
    b. Tạo combo bán kèm dựa trên quy luật khách hàng cùng mua.
    c. Khuyến mãi theo khung giờ vắng khách (Happy Hour).
    d. Khuyến mãi kích cầu theo khu vực / thành phố có doanh thu thấp.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng.

    Returns:
        List[Dict[str, str]]: Danh sách từ điển với các khóa:
            ['loai', 'mo_ta', 'san_pham', 'ly_do']
    """
    suggestions = []

    if df is None or df.empty:
        logger.warning("promotion_suggestions: DataFrame rỗng.")
        return suggestions

    try:
        # a. Sản phẩm bán chậm (cần giảm giá kích cầu)
        if {'Tên mặt hàng', 'Số lượng'}.issubset(df.columns):
            qty_by_prod = df.groupby('Tên mặt hàng')['Số lượng'].sum().sort_values()
            if not qty_by_prod.empty:
                slow_prod = qty_by_prod.index[0]
                slow_qty = float(qty_by_prod.iloc[0])
                suggestions.append({
                    'loai': 'Giảm giá xả kho',
                    'mo_ta': f"Áp dụng giảm giá 15% - 20% cho '{slow_prod}' nhằm kích cầu tiêu dùng và giải phóng hàng tồn.",
                    'san_pham': str(slow_prod),
                    'ly_do': f"Mặt hàng có sản lượng tiêu thụ thấp nhất trong kỳ (chỉ bán được {slow_qty:,.0f} sản phẩm)."
                })

        # b. Combo ưu đãi dựa trên cặp sản phẩm mua cùng
        rec_df = cross_sell_recommendations(df, top_n=1)
        if not rec_df.empty:
            p1 = rec_df['Sản phẩm chính'].iloc[0]
            p2 = rec_df['Sản phẩm gợi ý'].iloc[0]
            co_rate = rec_df['Tỷ lệ mua cùng (%)'].iloc[0]
            co_cust = rec_df['Số KH mua cùng'].iloc[0]
            suggestions.append({
                'loai': 'Combo ưu đãi',
                'mo_ta': f"Tạo gói combo mua kèm '{p1} + {p2}' giảm ngay 10% trên tổng giá trị combo.",
                'san_pham': f"{p1} & {p2}",
                'ly_do': f"Khách hàng mua '{p1}' có tỷ lệ mua kèm '{p2}' lên đến {co_rate:.1f}% ({co_cust} khách hàng cùng mua)."
            })
        elif {'Tên mặt hàng', 'Số lượng'}.issubset(df.columns):
            # Fallback nếu không có khách hàng định danh: ghép 2 sản phẩm bán chạy nhất
            top2 = df.groupby('Tên mặt hàng')['Số lượng'].sum().nlargest(2).index.tolist()
            if len(top2) >= 2:
                suggestions.append({
                    'loai': 'Combo ưu đãi',
                    'mo_ta': f"Tạo combo bán chạy '{top2[0]} + {top2[1]}' với mức chiết khấu 5% khi mua kèm.",
                    'san_pham': f"{top2[0]} & {top2[1]}",
                    'ly_do': "Hai sản phẩm được khách hàng yêu thích và lựa chọn nhiều nhất trong cửa hàng."
                })

        # c. Khuyến mãi theo khung giờ vắng khách (Happy Hour)
        if {'Ngày bán', 'Thành tiền'}.issubset(df.columns):
            dates = pd.to_datetime(df['Ngày bán'], errors='coerce')
            valid_time_df = df[dates.notna()].copy()
            valid_time_df['Giờ'] = dates[dates.notna()].dt.hour
            hour_rev = valid_time_df.groupby('Giờ')['Thành tiền'].sum()

            # Xét các khung giờ mở cửa chính (8h đến 21h)
            day_hours = hour_rev[(hour_rev.index >= 8) & (hour_rev.index <= 21)]
            if not day_hours.empty:
                off_peak_h = int(day_hours.idxmin())
                off_peak_val = float(day_hours.loc[off_peak_h])
            elif not hour_rev.empty:
                off_peak_h = int(hour_rev.idxmin())
                off_peak_val = float(hour_rev.loc[off_peak_h])
            else:
                off_peak_h = None

            if off_peak_h is not None:
                suggestions.append({
                    'loai': 'Khung giờ vàng (Happy Hour)',
                    'mo_ta': f"Triển khai chương trình 'Happy Hour' giảm giá 10% vào khung giờ {off_peak_h:02d}:00 - {off_peak_h+1:02d}:00 hàng ngày.",
                    'san_pham': 'Toàn bộ cửa hàng',
                    'ly_do': f"Khung giờ {off_peak_h:02d}:00 ghi nhận lượng giao dịch thấp nhất ban ngày (doanh thu {off_peak_val:,.0f} VNĐ)."
                })

        # d. Khuyến mãi theo khu vực / thành phố
        if {'Thành phố', 'Thành tiền'}.issubset(df.columns):
            city_rev = df.groupby('Thành phố')['Thành tiền'].sum().sort_values()
            if not city_rev.empty:
                low_city = city_rev.index[0]
                low_rev = float(city_rev.iloc[0])
                suggestions.append({
                    'loai': 'Kích cầu theo khu vực',
                    'mo_ta': f"Phát hành mã voucher giảm giá và chương trình tích điểm độc quyền cho thị trường {low_city}.",
                    'san_pham': 'Toàn bộ danh mục sản phẩm',
                    'ly_do': f"Thành phố {low_city} có doanh số thấp nhất toàn hệ thống ({low_rev:,.0f} VNĐ), cần tăng độ phủ nhận diện thương hiệu."
                })

        return suggestions

    except Exception as e:
        logger.error(f"Lỗi trong promotion_suggestions: {e}")
        return suggestions


def kpi_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tổng hợp toàn diện các chỉ số KPI cốt lõi của hoạt động kinh doanh.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng.

    Returns:
        Dict[str, Any]: Từ điển chứa các chỉ số KPI:
            - 'tong_doanh_thu': Tổng doanh thu (float).
            - 'so_giao_dich': Tổng số lượt giao dịch (int).
            - 'gia_tri_tb_gd': Giá trị đơn hàng trung bình (float).
            - 'so_khach_hang': Số khách hàng định danh (int).
            - 'so_san_pham': Số lượng mặt hàng đang kinh doanh (int).
            - 'sp_ban_chay_nhat': Tên sản phẩm bán chạy nhất theo số lượng (str).
            - 'sp_doanh_thu_cao_nhat': Tên sản phẩm có doanh thu cao nhất (str).
            - 'thanh_pho_dt_cao_nhat': Thành phố dẫn đầu về doanh thu (str).
            - 'pt_thanh_toan_pho_bien': Phương thức thanh toán phổ biến nhất (str).
            - 'gio_ban_chay_nhat': Khung giờ có doanh thu cao nhất (int).
            - 'thang_dt_cao_nhat': Tháng có doanh thu cao nhất (str).
            - 'tang_truong_dt': Tỷ lệ tăng trưởng doanh thu tháng gần nhất vs tháng trước (%) (float).
            - 'ty_le_kh_quay_lai': Tỷ lệ khách hàng quay lại mua sắm >1 lần (%) (float).
    """
    default_kpi = {
        'tong_doanh_thu': 0.0,
        'so_giao_dich': 0,
        'gia_tri_tb_gd': 0.0,
        'so_khach_hang': 0,
        'so_san_pham': 0,
        'sp_ban_chay_nhat': 'N/A',
        'sp_doanh_thu_cao_nhat': 'N/A',
        'thanh_pho_dt_cao_nhat': 'N/A',
        'pt_thanh_toan_pho_bien': 'N/A',
        'gio_ban_chay_nhat': 0,
        'thang_dt_cao_nhat': 'N/A',
        'tang_truong_dt': 0.0,
        'ty_le_kh_quay_lai': 0.0
    }

    if df is None or df.empty:
        logger.warning("kpi_summary: DataFrame rỗng.")
        return default_kpi

    try:
        # Doanh thu và số giao dịch
        tong_dt = float(df['Thành tiền'].sum()) if 'Thành tiền' in df.columns else 0.0
        so_gd = int(len(df))
        gia_tri_tb = (tong_dt / so_gd) if so_gd > 0 else 0.0

        # Khách hàng định danh
        so_kh = 0
        ty_le_quay_lai = 0.0
        if 'Mã khách hàng' in df.columns:
            valid_cust = df['Mã khách hàng'].dropna().astype(str).str.strip()
            valid_cust = valid_cust[~valid_cust.isin(['', 'nan', 'None'])]
            so_kh = int(valid_cust.nunique())
            if so_kh > 0:
                cust_counts = valid_cust.value_counts()
                repeat_custs = int((cust_counts > 1).sum())
                ty_le_quay_lai = (repeat_custs / so_kh) * 100.0

        # Số sản phẩm
        so_sp = int(df['Tên mặt hàng'].nunique()) if 'Tên mặt hàng' in df.columns else 0

        # Mặt hàng bán chạy nhất và doanh thu cao nhất
        sp_ban_chay = 'N/A'
        if {'Tên mặt hàng', 'Số lượng'}.issubset(df.columns):
            sp_qty = df.groupby('Tên mặt hàng')['Số lượng'].sum()
            if not sp_qty.empty:
                sp_ban_chay = str(sp_qty.idxmax())

        sp_dt_cao = 'N/A'
        if {'Tên mặt hàng', 'Thành tiền'}.issubset(df.columns):
            sp_rev = df.groupby('Tên mặt hàng')['Thành tiền'].sum()
            if not sp_rev.empty:
                sp_dt_cao = str(sp_rev.idxmax())

        # Thành phố dẫn đầu
        tp_cao = 'N/A'
        if {'Thành phố', 'Thành tiền'}.issubset(df.columns):
            tp_rev = df.groupby('Thành phố')['Thành tiền'].sum()
            if not tp_rev.empty:
                tp_cao = str(tp_rev.idxmax())

        # Phương thức thanh toán phổ biến nhất
        pt_tt = 'N/A'
        if 'Phương thức thanh toán' in df.columns:
            mode_s = df['Phương thức thanh toán'].dropna().mode()
            if not mode_s.empty:
                pt_tt = str(mode_s.iloc[0])

        # Thời gian bán hàng
        gio_top = 0
        thang_top = 'N/A'
        tang_truong = 0.0

        if {'Ngày bán', 'Thành tiền'}.issubset(df.columns):
            dates = pd.to_datetime(df['Ngày bán'], errors='coerce')
            valid_date_mask = dates.notna()

            if valid_date_mask.any():
                # Giờ bán chạy nhất
                hour_rev = df[valid_date_mask].groupby(dates[valid_date_mask].dt.hour)['Thành tiền'].sum()
                if not hour_rev.empty:
                    gio_top = int(hour_rev.idxmax())

                # Doanh thu theo tháng
                month_rev = df[valid_date_mask].groupby(dates[valid_date_mask].dt.to_period('M'))['Thành tiền'].sum()
                month_rev_sorted = month_rev.sort_index()

                if not month_rev_sorted.empty:
                    best_period = month_rev_sorted.idxmax()
                    thang_top = best_period.strftime('%m-%Y')

                    if len(month_rev_sorted) >= 2:
                        last_rev = float(month_rev_sorted.iloc[-1])
                        prev_rev = float(month_rev_sorted.iloc[-2])
                        tang_truong = ((last_rev - prev_rev) / prev_rev) * 100.0 if prev_rev > 0 else 0.0

        return {
            'tong_doanh_thu': round(tong_dt, 2),
            'so_giao_dich': so_gd,
            'gia_tri_tb_gd': round(gia_tri_tb, 2),
            'so_khach_hang': so_kh,
            'so_san_pham': so_sp,
            'sp_ban_chay_nhat': sp_ban_chay,
            'sp_doanh_thu_cao_nhat': sp_dt_cao,
            'thanh_pho_dt_cao_nhat': tp_cao,
            'pt_thanh_toan_pho_bien': pt_tt,
            'gio_ban_chay_nhat': gio_top,
            'thang_dt_cao_nhat': thang_top,
            'tang_truong_dt': round(tang_truong, 2),
            'ty_le_kh_quay_lai': round(ty_le_quay_lai, 2)
        }

    except Exception as e:
        logger.error(f"Lỗi trong kpi_summary: {e}")
        return default_kpi


def business_insights(df: pd.DataFrame) -> List[str]:
    """
    Tạo ra 10 nhận định kinh doanh sâu sắc bằng ngôn ngữ tự nhiên từ dữ liệu bán hàng.

    Args:
        df (pd.DataFrame): Dữ liệu bán hàng.

    Returns:
        List[str]: Danh sách 10 câu nhận định thực tế hỗ trợ ra quyết định.
    """
    if df is None or df.empty:
        logger.warning("business_insights: DataFrame rỗng.")
        return ["Chưa có dữ liệu giao dịch để phân tích nhận định kinh doanh."]

    try:
        insights = []
        kpi = kpi_summary(df)

        # 1. Tổng quan quy mô doanh thu và đơn hàng
        insights.append(
            f"1. Tổng doanh thu hệ thống đạt {kpi['tong_doanh_thu']:,.0f} VNĐ thông qua {kpi['so_giao_dich']:,} giao dịch thành công."
        )

        # 2. Giá trị trung bình đơn hàng (AOV)
        insights.append(
            f"2. Giá trị đơn hàng trung bình (AOV) đạt {kpi['gia_tri_tb_gd']:,.0f} VNĐ/giao dịch."
        )

        # 3. Sản phẩm có sản lượng lớn nhất
        if {'Tên mặt hàng', 'Số lượng'}.issubset(df.columns):
            qty_top = df.groupby('Tên mặt hàng')['Số lượng'].sum().max()
            insights.append(
                f"3. Mặt hàng có sản lượng tiêu thụ lớn nhất là '{kpi['sp_ban_chay_nhat']}' với tổng cộng {qty_top:,} sản phẩm được bán ra."
            )
        else:
            insights.append(f"3. Sản phẩm bán chạy nhất là '{kpi['sp_ban_chay_nhat']}'.")

        # 4. Sản phẩm đóng góp doanh thu cao nhất
        if {'Tên mặt hàng', 'Thành tiền'}.issubset(df.columns) and kpi['tong_doanh_thu'] > 0:
            rev_top = df.groupby('Tên mặt hàng')['Thành tiền'].sum().max()
            rev_pct = (rev_top / kpi['tong_doanh_thu']) * 100.0
            insights.append(
                f"4. Mặt hàng tạo doanh thu cao nhất là '{kpi['sp_doanh_thu_cao_nhat']}' đạt {rev_top:,.0f} VNĐ (chiếm {rev_pct:.1f}% tổng doanh thu)."
            )
        else:
            insights.append(f"4. Mặt hàng tạo doanh thu lớn nhất là '{kpi['sp_doanh_thu_cao_nhat']}'.")

        # 5. Tháng kinh doanh ấn tượng nhất
        if {'Ngày bán', 'Thành tiền'}.issubset(df.columns):
            dates = pd.to_datetime(df['Ngày bán'], errors='coerce')
            monthly_rev = df[dates.notna()].groupby(dates[dates.notna()].dt.to_period('M'))['Thành tiền'].sum()
            best_month_rev = monthly_rev.max() if not monthly_rev.empty else 0.0
            insights.append(
                f"5. Tháng có doanh số cao nhất là {kpi['thang_dt_cao_nhat']} với doanh thu ghi nhận {best_month_rev:,.0f} VNĐ."
            )
        else:
            insights.append(f"5. Tháng đỉnh điểm doanh thu là {kpi['thang_dt_cao_nhat']}.")

        # 6. Khung giờ mua sắm cao điểm
        if {'Ngày bán', 'Thành tiền'}.issubset(df.columns):
            dates = pd.to_datetime(df['Ngày bán'], errors='coerce')
            hour_rev = df[dates.notna()].groupby(dates[dates.notna()].dt.hour)['Thành tiền'].sum()
            peak_hour_rev = hour_rev.max() if not hour_rev.empty else 0.0
            insights.append(
                f"6. Khung giờ cao điểm mua sắm là {kpi['gio_ban_chay_nhat']:02d}:00 với doanh thu tích lũy đạt {peak_hour_rev:,.0f} VNĐ."
            )
        else:
            insights.append(f"6. Khung giờ mua sắm tốt nhất là {kpi['gio_ban_chay_nhat']} giờ.")

        # 7. Thị trường dẫn đầu theo địa lý
        if {'Thành phố', 'Thành tiền'}.issubset(df.columns) and kpi['tong_doanh_thu'] > 0:
            city_rev = df.groupby('Thành phố')['Thành tiền'].sum().max()
            city_pct = (city_rev / kpi['tong_doanh_thu']) * 100.0
            insights.append(
                f"7. Thị trường {kpi['thanh_pho_dt_cao_nhat']} dẫn đầu toàn quốc về doanh số, đạt {city_rev:,.0f} VNĐ ({city_pct:.1f}% toàn chuỗi)."
            )
        else:
            insights.append(f"7. Thành phố đạt doanh thu cao nhất là {kpi['thanh_pho_dt_cao_nhat']}.")

        # 8. Kênh thanh toán được ưa chuộng nhất
        if 'Phương thức thanh toán' in df.columns:
            pm_counts = df['Phương thức thanh toán'].value_counts()
            top_pm_count = pm_counts.iloc[0] if not pm_counts.empty else 0
            pm_pct = (top_pm_count / kpi['so_giao_dich']) * 100.0 if kpi['so_giao_dich'] > 0 else 0.0
            insights.append(
                f"8. Phương thức thanh toán '{kpi['pt_thanh_toan_pho_bien']}' chiếm tỷ trọng lớn nhất với {pm_pct:.1f}% tổng số đơn ({top_pm_count:,} giao dịch)."
            )
        else:
            insights.append(f"8. Phương thức thanh toán phổ biến nhất: {kpi['pt_thanh_toan_pho_bien']}.")

        # 9. Tỷ lệ trung thành và quay lại của khách hàng
        insights.append(
            f"9. Tỷ lệ khách hàng định danh quay lại mua sắm đạt {kpi['ty_le_kh_quay_lai']:.1f}% ({kpi['so_khach_hang']:,} khách hàng đã đăng ký)."
        )

        # 10. Cơ hội bán chéo hoặc xu hướng tăng trưởng
        recs = cross_sell_recommendations(df, top_n=1)
        if not recs.empty:
            p1 = recs['Sản phẩm chính'].iloc[0]
            p2 = recs['Sản phẩm gợi ý'].iloc[0]
            co_rate = recs['Tỷ lệ mua cùng (%)'].iloc[0]
            insights.append(
                f"10. Cơ hội bán chéo tiềm năng: Khách hàng mua '{p1}' có xu hướng mua kèm '{p2}' đạt tỷ lệ {co_rate:.1f}%."
            )
        else:
            trend_val = kpi['tang_truong_dt']
            direction = "tăng" if trend_val >= 0 else "giảm"
            insights.append(
                f"10. Tăng trưởng doanh thu tháng gần nhất đạt {trend_val:+.1f}% ({direction} so với tháng liền trước)."
            )

        return insights

    except Exception as e:
        logger.error(f"Lỗi trong business_insights: {e}")
        return ["Đã xảy ra lỗi khi trích xuất nhận định kinh doanh từ dữ liệu."]
