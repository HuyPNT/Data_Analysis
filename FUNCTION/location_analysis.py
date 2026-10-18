import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ============================================================
#  Bảng phân loại vùng miền Việt Nam
# ============================================================
REGION_MAP = {
    # Miền Bắc
    'Hà Nội': 'Miền Bắc', 'Hải Phòng': 'Miền Bắc', 'Hà Nam': 'Miền Bắc',
    'Thái Bình': 'Miền Bắc', 'Điện Biên': 'Miền Bắc', 'Quảng Ninh': 'Miền Bắc',
    'Bắc Ninh': 'Miền Bắc', 'Bắc Giang': 'Miền Bắc', 'Hải Dương': 'Miền Bắc',
    'Hưng Yên': 'Miền Bắc', 'Nam Định': 'Miền Bắc', 'Ninh Bình': 'Miền Bắc',
    'Vĩnh Phúc': 'Miền Bắc', 'Phú Thọ': 'Miền Bắc', 'Thái Nguyên': 'Miền Bắc',
    'Lạng Sơn': 'Miền Bắc', 'Cao Bằng': 'Miền Bắc', 'Hà Giang': 'Miền Bắc',
    'Lào Cai': 'Miền Bắc', 'Yên Bái': 'Miền Bắc', 'Tuyên Quang': 'Miền Bắc',
    'Sơn La': 'Miền Bắc', 'Lai Châu': 'Miền Bắc', 'Hòa Bình': 'Miền Bắc',
    'Bắc Kạn': 'Miền Bắc',
    # Miền Trung
    'Thanh Hóa': 'Miền Trung', 'Nghệ An': 'Miền Trung', 'Hà Tĩnh': 'Miền Trung',
    'Quảng Bình': 'Miền Trung', 'Quảng Trị': 'Miền Trung',
    'Thừa Thiên Huế': 'Miền Trung', 'Đà Nẵng': 'Miền Trung',
    'Quảng Nam': 'Miền Trung', 'Quảng Ngãi': 'Miền Trung',
    'Bình Định': 'Miền Trung', 'Phú Yên': 'Miền Trung',
    'Khánh Hòa': 'Miền Trung', 'Ninh Thuận': 'Miền Trung',
    'Bình Thuận': 'Miền Trung', 'Kon Tum': 'Miền Trung',
    'Gia Lai': 'Miền Trung', 'Đắk Lắk': 'Miền Trung',
    'Đắk Nông': 'Miền Trung', 'Lâm Đồng': 'Miền Trung',
    # Miền Nam
    'TP.HCM': 'Miền Nam', 'Hồ Chí Minh': 'Miền Nam',
    'Bà Rịa - Vũng Tàu': 'Miền Nam', 'Bình Dương': 'Miền Nam',
    'Bình Phước': 'Miền Nam', 'Tây Ninh': 'Miền Nam',
    'Đồng Nai': 'Miền Nam', 'Long An': 'Miền Nam',
    'Tiền Giang': 'Miền Nam', 'Bến Tre': 'Miền Nam',
    'Trà Vinh': 'Miền Nam', 'Vĩnh Long': 'Miền Nam',
    'Đồng Tháp': 'Miền Nam', 'An Giang': 'Miền Nam',
    'Kiên Giang': 'Miền Nam', 'Cần Thơ': 'Miền Nam',
    'Hậu Giang': 'Miền Nam', 'Sóc Trăng': 'Miền Nam',
    'Bạc Liêu': 'Miền Nam', 'Cà Mau': 'Miền Nam',
}


def classify_region(city):
    """Phân loại thành phố vào vùng miền."""
    if pd.isna(city):
        return 'Khác'
    city_str = str(city).strip()
    return REGION_MAP.get(city_str, 'Khác')


def revenue_by_region(df):
    """
    Tính doanh thu theo vùng miền (Bắc - Trung - Nam).
    Trả về DataFrame: Vùng miền, Doanh thu, Số giao dịch, Tỷ lệ (%).
    """
    if df.empty:
        return pd.DataFrame(columns=['Vùng miền', 'Doanh thu', 'Số giao dịch', 'Tỷ lệ (%)'])

    df_copy = df.copy()
    df_copy['Vùng miền'] = df_copy['Thành phố'].apply(classify_region)

    result = df_copy.groupby('Vùng miền').agg(
        doanh_thu=('Thành tiền', 'sum'),
        so_gd=('Mã giao dịch', 'nunique')
    ).reset_index()

    total_revenue = result['doanh_thu'].sum()
    result['ty_le'] = (result['doanh_thu'] / total_revenue * 100).round(1) if total_revenue > 0 else 0

    result.columns = ['Vùng miền', 'Doanh thu', 'Số giao dịch', 'Tỷ lệ (%)']
    result = result.sort_values('Doanh thu', ascending=False)

    logging.info(f"Phân tích doanh thu theo {len(result)} vùng miền")
    return result


def popular_products_by_city(df, top_n=5):
    """
    Tìm top N sản phẩm phổ biến nhất tại mỗi thành phố.
    Trả về DataFrame: Thành phố, Tên mặt hàng, Số lượng, Xếp hạng.
    """
    if df.empty:
        return pd.DataFrame(columns=['Thành phố', 'Tên mặt hàng', 'Số lượng', 'Xếp hạng'])

    product_by_city = df.groupby(['Thành phố', 'Tên mặt hàng'])['Số lượng'].sum().reset_index()

    result_rows = []
    for city in product_by_city['Thành phố'].unique():
        city_data = product_by_city[product_by_city['Thành phố'] == city]
        top = city_data.nlargest(top_n, 'Số lượng')
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            result_rows.append({
                'Thành phố': row['Thành phố'],
                'Tên mặt hàng': row['Tên mặt hàng'],
                'Số lượng': row['Số lượng'],
                'Xếp hạng': rank
            })

    result = pd.DataFrame(result_rows)
    logging.info(f"Đã phân tích SP phổ biến cho {df['Thành phố'].nunique()} thành phố")
    return result


def popular_products_by_region(df, top_n=5):
    """
    Tìm top N sản phẩm phổ biến nhất theo vùng miền.
    Trả về DataFrame: Vùng miền, Tên mặt hàng, Số lượng, Xếp hạng.
    """
    if df.empty:
        return pd.DataFrame(columns=['Vùng miền', 'Tên mặt hàng', 'Số lượng', 'Xếp hạng'])

    df_copy = df.copy()
    df_copy['Vùng miền'] = df_copy['Thành phố'].apply(classify_region)

    product_by_region = df_copy.groupby(['Vùng miền', 'Tên mặt hàng'])['Số lượng'].sum().reset_index()

    result_rows = []
    for region in product_by_region['Vùng miền'].unique():
        region_data = product_by_region[product_by_region['Vùng miền'] == region]
        top = region_data.nlargest(top_n, 'Số lượng')
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            result_rows.append({
                'Vùng miền': row['Vùng miền'],
                'Tên mặt hàng': row['Tên mặt hàng'],
                'Số lượng': row['Số lượng'],
                'Xếp hạng': rank
            })

    result = pd.DataFrame(result_rows)
    logging.info(f"Đã phân tích SP phổ biến cho {df_copy['Vùng miền'].nunique()} vùng miền")
    return result


def payment_by_region(df):
    """
    Phân tích phương thức thanh toán theo vùng miền.
    Trả về DataFrame crosstab: Vùng miền vs các phương thức thanh toán.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Vùng miền'] = df_copy['Thành phố'].apply(classify_region)

    # Crosstab
    ct = pd.crosstab(df_copy['Vùng miền'], df_copy['Phương thức thanh toán'])
    ct = ct.reset_index()

    logging.info(f"Phân tích PT thanh toán cho {len(ct)} vùng miền")
    return ct


def regional_comparison(df):
    """
    So sánh tổng hợp các vùng miền.
    Trả về DataFrame: Vùng miền, Doanh thu, Giá trị TB, Số KH, SP phổ biến, PT thanh toán phổ biến.
    """
    if df.empty:
        return pd.DataFrame()

    df_copy = df.copy()
    df_copy['Vùng miền'] = df_copy['Thành phố'].apply(classify_region)

    result_rows = []
    for region in df_copy['Vùng miền'].unique():
        region_data = df_copy[df_copy['Vùng miền'] == region]

        doanh_thu = region_data['Thành tiền'].sum()
        gia_tri_tb = region_data['Thành tiền'].mean()

        # Số khách hàng (bỏ NaN)
        kh_data = region_data.dropna(subset=['Mã khách hàng'])
        so_kh = kh_data['Mã khách hàng'].nunique() if not kh_data.empty else 0

        # SP phổ biến nhất
        sp_popular = region_data.groupby('Tên mặt hàng')['Số lượng'].sum()
        sp_top = sp_popular.idxmax() if not sp_popular.empty else 'N/A'

        # PT thanh toán phổ biến nhất
        pt_popular = region_data['Phương thức thanh toán'].mode()
        pt_top = pt_popular.iloc[0] if not pt_popular.empty else 'N/A'

        result_rows.append({
            'Vùng miền': region,
            'Doanh thu': doanh_thu,
            'Giá trị TB': gia_tri_tb,
            'Số KH': so_kh,
            'SP phổ biến': sp_top,
            'PT thanh toán phổ biến': pt_top
        })

    result = pd.DataFrame(result_rows)
    result = result.sort_values('Doanh thu', ascending=False)

    logging.info(f"So sánh tổng hợp {len(result)} vùng miền")
    return result
