import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ============================================================
#  Cấu hình chung
# ============================================================
plt.rcParams['font.size'] = 10
COLORS = {
    'primary': '#2196F3',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#F44336',
    'info': '#00BCD4',
    'purple': '#9C27B0',
    'teal': '#009688',
    'navy': '#1A237E',
}
PALETTE = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
           '#00BCD4', '#795548', '#607D8B', '#E91E63', '#3F51B5']


def _fmt_vnd(x, _=None):
    """Định dạng số tiền VNĐ."""
    return f'{x:,.0f}'


# ============================================================
#  1. BIỂU ĐỒ CHẤT LƯỢNG DỮ LIỆU
# ============================================================

def plot_data_quality_report(df):
    """Biểu đồ tổng quan chất lượng dữ liệu (đã tinh chỉnh chống đè chữ và bỏ thông tin trùng lặp)."""
    if df.empty:
        return None

    from FUNCTION.data_cleaning import data_quality_report
    report = data_quality_report(df)

    fig = plt.figure(figsize=(13.5, 6))
    fig.suptitle('Báo Cáo Chất Lượng Dữ Liệu', fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.38, wspace=0.32)

    # 1. Số giá trị duy nhất theo từng cột (Unique values - Horizontal bar chart bên trái)
    ax1 = fig.add_subplot(gs[:, 0])
    if report.get('unique_values'):
        uv_series = pd.Series(report['unique_values']).sort_values(ascending=True)
        colors_uv = [COLORS['primary'] if v > 10 else COLORS['teal'] for v in uv_series.values]
        bars1 = ax1.barh(uv_series.index, uv_series.values, color=colors_uv, alpha=0.85, height=0.6)
        ax1.set_title('Số giá trị duy nhất theo cột (Unique Values)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Số lượng giá trị duy nhất', fontsize=9)
        max_uv = uv_series.max() if not uv_series.empty else 1
        for bar, val in zip(bars1, uv_series.values):
            ax1.text(bar.get_width() + (max_uv * 0.02),
                     bar.get_y() + bar.get_height() / 2,
                     f'{val:,}',
                     va='center', fontsize=8.5, fontweight='bold', color='#2c3e50')
        ax1.set_xlim(0, max_uv * 1.18)
    else:
        ax1.text(0.5, 0.5, 'Không có dữ liệu unique values', ha='center', va='center')
        ax1.axis('off')

    # 2. Giá trị thiếu theo từng cột (Missing values - Top right)
    ax2 = fig.add_subplot(gs[0, 1])
    missing_dict = report.get('missing_values', {})
    if missing_dict:
        mv_series = pd.Series(missing_dict).sort_values(ascending=True)
        has_missing = any(v > 0 for v in mv_series.values)
        if not has_missing:
            ax2.barh(['Tất cả các cột'], [100], color=COLORS['success'], alpha=0.85, height=0.45)
            ax2.text(50, 0, '100% Đầy đủ (Không có giá trị thiếu)', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white')
            ax2.set_xlim(0, 105)
            ax2.set_xlabel('Tỷ lệ hoàn chỉnh (%)', fontsize=9)
            ax2.set_title('Tình trạng giá trị khuyết thiếu (Missing Values)', fontsize=11, fontweight='bold')
        else:
            colors_mv = [COLORS['danger'] if v > 0 else COLORS['success'] for v in mv_series.values]
            bars2 = ax2.barh(mv_series.index, mv_series.values, color=colors_mv, alpha=0.85, height=0.6)
            ax2.set_title('Số lượng giá trị thiếu theo cột', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Số lượng dòng thiếu', fontsize=9)
            max_mv = max(mv_series.max(), 1)
            for bar, val in zip(bars2, mv_series.values):
                ax2.text(bar.get_width() + (max_mv * 0.02),
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:,}',
                         va='center', fontsize=8, color='#c0392b' if val > 0 else '#27ae60')
            ax2.set_xlim(0, max_mv * 1.2)
    else:
        ax2.axis('off')

    # 3. Phân bổ kiểu dữ liệu (Data types - Bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    if report.get('data_types'):
        dtype_counts = {}
        for dt in report['data_types'].values():
            dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        colors_dt = PALETTE[:len(dtype_counts)]
        wedges, texts, autotexts = ax3.pie(
            dtype_counts.values(),
            labels=dtype_counts.keys(),
            autopct='%1.0f%%',
            colors=colors_dt,
            startangle=90,
            pctdistance=0.7,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
        )
        for at in autotexts:
            at.set_fontsize(8.5)
            at.set_fontweight('bold')
        ax3.set_title('Phân bổ kiểu dữ liệu (Data Types)', fontsize=11, fontweight='bold')
    else:
        ax3.axis('off')

    return fig


def plot_outliers(df):
    """Biểu đồ phát hiện outliers bằng IQR."""
    if df.empty or 'Thành tiền' not in df.columns:
        return None

    from FUNCTION.data_cleaning import detect_outliers_iqr
    outliers, lower, upper = detect_outliers_iqr(df, 'Thành tiền')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phát hiện Giá trị Ngoại lệ (Outliers)', fontsize=14, fontweight='bold')

    # Box plot
    ax = axes[0]
    ax.boxplot(df['Thành tiền'].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=COLORS['info'], alpha=0.7))
    ax.set_title('Box Plot - Thành tiền')
    ax.set_ylabel('VNĐ')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))

    # Scatter plot
    ax = axes[1]
    normal = df[~df.index.isin(outliers.index)]
    ax.scatter(range(len(normal)), normal['Thành tiền'], alpha=0.3,
               color=COLORS['primary'], label='Bình thường', s=10)
    if not outliers.empty:
        ax.scatter(outliers.index, outliers['Thành tiền'], alpha=0.7,
                   color=COLORS['danger'], label=f'Outlier ({len(outliers)})', s=20)
    ax.axhline(upper, color=COLORS['warning'], linestyle='--', label=f'Ngưỡng trên: {upper:,.0f}')
    ax.axhline(lower, color=COLORS['warning'], linestyle='--', label=f'Ngưỡng dưới: {lower:,.0f}')
    ax.set_title('Phân bố Thành tiền')
    ax.set_ylabel('VNĐ')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))

    fig.tight_layout()
    return fig


# ============================================================
#  2. BIỂU ĐỒ LUẬT KẾT HỢP (ASSOCIATION RULES)
# ============================================================

def plot_association_rules(df):
    """Biểu đồ luật kết hợp - scatter plot Support vs Confidence, sized by Lift."""
    if df.empty:
        return None
    try:
        from FUNCTION.association_rules import generate_rules
        rules = generate_rules(df)
        if rules is None or rules.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Không tìm thấy luật kết hợp nào.\nThử giảm min_support hoặc min_confidence.',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Luật Kết hợp (Association Rules)')
            ax.axis('off')
            return fig

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            rules['support'], rules['confidence'],
            s=rules['lift'] * 80, c=rules['lift'],
            cmap='YlOrRd', alpha=0.7, edgecolors='gray'
        )
        plt.colorbar(scatter, ax=ax, label='Lift')
        ax.set_xlabel('Support')
        ax.set_ylabel('Confidence')
        ax.set_title('Luật Kết hợp: Support vs Confidence (kích thước = Lift)')

        # Ghi chú top 5 rules
        top_rules = rules.nlargest(5, 'lift')
        for _, row in top_rules.iterrows():
            ax.annotate(
                f"{row['antecedents']} → {row['consequents']}",
                (row['support'], row['confidence']),
                fontsize=7, alpha=0.8,
                textcoords="offset points", xytext=(5, 5)
            )
        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ association rules: {e}")
        return None


def plot_frequent_itemsets(df):
    """Biểu đồ top sản phẩm hay mua cùng nhau."""
    if df.empty:
        return None
    try:
        from FUNCTION.association_rules import get_top_product_pairs
        pairs = get_top_product_pairs(df, top_n=10)
        if pairs is None or pairs.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Không tìm thấy cặp sản phẩm nào được mua cùng nhau.',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Sản phẩm hay mua cùng')
            ax.axis('off')
            return fig

        fig, ax = plt.subplots(figsize=(10, 7))
        pair_labels = [f"{row.iloc[0]}\n+ {row.iloc[1]}" for _, row in pairs.iterrows()]
        counts = pairs.iloc[:, 2]
        bars = ax.barh(range(len(pair_labels)), counts, color=PALETTE[:len(pair_labels)])
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_xlabel('Số lần mua cùng')
        ax.set_title('Top 10 Cặp sản phẩm hay mua cùng nhau')
        ax.invert_yaxis()

        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(int(val)), va='center', fontsize=9)
        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ frequent itemsets: {e}")
        return None


def plot_product_recommendations(df):
    """Biểu đồ gợi ý sản phẩm dựa trên luật kết hợp."""
    if df.empty:
        return None
    try:
        from FUNCTION.association_rules import generate_rules
        rules = generate_rules(df)
        if rules is None or rules.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Không có dữ liệu gợi ý sản phẩm.',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig

        top = rules.nlargest(10, 'lift')
        fig, ax = plt.subplots(figsize=(10, 7))
        labels = [f"{row['antecedents']} → {row['consequents']}" for _, row in top.iterrows()]
        lifts = top['lift'].values
        confs = top['confidence'].values

        y_pos = range(len(labels))
        bars = ax.barh(y_pos, lifts, color=COLORS['purple'], alpha=0.7, label='Lift')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Lift')
        ax.set_title('Top 10 Gợi ý Sản phẩm (theo Lift)')
        ax.invert_yaxis()

        # Thêm confidence label
        for i, (bar, conf) in enumerate(zip(bars, confs)):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'Conf: {conf:.0%}', va='center', fontsize=8, color=COLORS['info'])
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ recommendations: {e}")
        return None


def plot_payment_methods(df):
    """Biểu đồ phân tích phương thức thanh toán: số lượng giao dịch và doanh thu."""
    if df.empty or 'Phương thức thanh toán' not in df.columns:
        return None

    df_copy = df.dropna(subset=['Phương thức thanh toán']).copy()
    if df_copy.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Phân Tích Phương Thức Thanh Toán', fontsize=14, fontweight='bold')

    # 1. Tỷ lệ số lượt giao dịch (Donut / Pie chart)
    ax1 = axes[0]
    counts = df_copy['Phương thức thanh toán'].value_counts()
    colors = PALETTE[:len(counts)]
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax1.set_title('Tỷ lệ số lượt giao dịch (%)', fontsize=11, fontweight='bold')

    # 2. Doanh thu theo phương thức thanh toán (Triệu VNĐ)
    ax2 = axes[1]
    rev = df_copy.groupby('Phương thức thanh toán')['Thành tiền'].sum().reindex(counts.index)
    rev_million = rev / 1e6
    bars = ax2.barh(rev.index, rev_million, color=colors, alpha=0.85, edgecolor='gray', height=0.55)
    ax2.set_xlabel('Doanh thu (Triệu VNĐ)', fontsize=9)
    ax2.set_title('Doanh thu theo phương thức thanh toán (Triệu VNĐ)', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()

    max_val = rev_million.max() if not rev_million.empty else 1
    for bar, val in zip(bars, rev_million):
        ax2.text(bar.get_width() + (max_val * 0.02),
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:,.1f} Tr',
                 va='center', fontsize=9, fontweight='bold', color='#2c3e50')

    ax2.set_xlim(0, max_val * 1.2)
    fig.tight_layout()
    return fig


# ============================================================
#  3. BIỂU ĐỒ PHÂN TÍCH KHÁCH HÀNG
# ============================================================

def plot_rfm_analysis(df):
    """Biểu đồ phân bổ RFM (đã chuyển Monetary sang Triệu VNĐ chống đè chữ)."""
    if df.empty:
        return None
    try:
        from FUNCTION.customer_analysis import calculate_rfm
        rfm = calculate_rfm(df)
        if rfm is None or rfm.empty:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle('Phân tích RFM - Khách hàng', fontsize=14, fontweight='bold')

        # Recency
        axes[0].hist(rfm['Recency'], bins=20, color=COLORS['primary'], alpha=0.75, edgecolor='white')
        axes[0].set_title('Recency (Ngày mua gần nhất)', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Số ngày', fontsize=9)
        axes[0].set_ylabel('Số khách hàng', fontsize=9)

        # Frequency
        axes[1].hist(rfm['Frequency'], bins=20, color=COLORS['success'], alpha=0.75, edgecolor='white')
        axes[1].set_title('Frequency (Tần suất mua hàng)', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Số đơn hàng', fontsize=9)
        axes[1].set_ylabel('Số khách hàng', fontsize=9)

        # Monetary (Triệu VNĐ)
        monetary_m = rfm['Monetary'] / 1e6
        axes[2].hist(monetary_m, bins=20, color=COLORS['warning'], alpha=0.75, edgecolor='white')
        axes[2].set_title('Monetary (Triệu VNĐ)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Tổng chi tiêu (Triệu VNĐ)', fontsize=9)
        axes[2].set_ylabel('Số khách hàng', fontsize=9)
        axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ RFM: {e}")
        return None


def plot_customer_segments(df):
    """Biểu đồ phân khúc khách hàng (đã chuyển Monetary sang Triệu VNĐ)."""
    if df.empty:
        return None
    try:
        from FUNCTION.customer_analysis import segment_customers
        rfm_seg, summary = segment_customers(df)
        if rfm_seg is None or rfm_seg.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle('Phân khúc Khách hàng (K-Means)', fontsize=14, fontweight='bold')

        # Pie chart phân bổ
        ax = axes[0]
        seg_counts = rfm_seg['Phân khúc'].value_counts()
        ax.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
               colors=PALETTE[:len(seg_counts)], startangle=90)
        ax.set_title('Phân bổ khách hàng theo phân khúc', fontsize=11, fontweight='bold')

        # Bar chart so sánh monetary (Triệu VNĐ)
        ax = axes[1]
        if summary is not None and not summary.empty:
            seg_names = summary.iloc[:, 0]
            monetary_col = [c for c in summary.columns if 'Monetary' in c or 'monetary' in c.lower() or 'DT' in c]
            if monetary_col:
                vals = summary[monetary_col[0]]
                vals_m = vals / 1e6
                bars = ax.bar(seg_names, vals_m, color=PALETTE[:len(seg_names)], width=0.55)
                ax.set_title('Chi tiêu trung bình theo phân khúc (Triệu VNĐ)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Triệu VNĐ', fontsize=9)
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
                max_v = vals_m.max() if not vals_m.empty else 1
                for bar, val in zip(bars, vals_m):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (max_v * 0.02),
                            f'{val:.2f} Tr',
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
                ax.set_ylim(0, max_v * 1.18)
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.text(0.5, 0.5, 'Không có dữ liệu monetary', ha='center', va='center')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu tổng hợp', ha='center', va='center')
            ax.axis('off')

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ customer segments: {e}")
        return None


def plot_customer_value(df):
    """Biểu đồ giá trị vòng đời khách hàng (đã chuyển đơn vị sang Triệu VNĐ chống đè chữ)."""
    if df.empty:
        return None
    try:
        from FUNCTION.customer_analysis import customer_lifetime_value
        clv = customer_lifetime_value(df)
        if clv is None or clv.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle('Giá trị Vòng đời Khách hàng (CLV)', fontsize=14, fontweight='bold')

        # 1. Distribution (Triệu VNĐ)
        ax = axes[0]
        monetary_col = clv.columns[1]  # Tổng DT
        dt_trieu = clv[monetary_col] / 1e6
        ax.hist(dt_trieu, bins=25, color=COLORS['info'], alpha=0.75, edgecolor='white')
        ax.set_title('Phân bổ Tổng doanh thu KH (Triệu VNĐ)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Doanh thu (Triệu VNĐ)', fontsize=9)
        ax.set_ylabel('Số khách hàng', fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))

        # 2. Top 10 customers (Triệu VNĐ)
        ax = axes[1]
        top10 = clv.nlargest(10, monetary_col).copy()
        top10_m = top10[monetary_col] / 1e6
        bars = ax.barh(top10.iloc[:, 0].astype(str), top10_m,
                       color=COLORS['success'], alpha=0.85, edgecolor='gray', height=0.6)
        ax.set_title('Top 10 Khách hàng giá trị nhất', fontsize=11, fontweight='bold')
        ax.set_xlabel('Tổng doanh thu (Triệu VNĐ)', fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.invert_yaxis()

        max_val = top10_m.max() if not top10_m.empty else 1
        for bar, val in zip(bars, top10_m):
            ax.text(bar.get_width() + (max_val * 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f} Tr',
                    va='center', fontsize=8.5, fontweight='bold', color='#1b5e20')
        ax.set_xlim(0, max_val * 1.18)

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ CLV: {e}")
        return None


# ============================================================
#  4. BIỂU ĐỒ PHÂN TÍCH VỊ TRÍ
# ============================================================

def plot_revenue_by_region(df):
    """Biểu đồ doanh thu theo vùng miền."""
    if df.empty:
        return None
    try:
        from FUNCTION.location_analysis import revenue_by_region
        region_data = revenue_by_region(df)
        if region_data is None or region_data.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle('Doanh thu theo Vùng miền', fontsize=14, fontweight='bold')

        # Bar chart
        ax = axes[0]
        regions = region_data.iloc[:, 0]
        revenue = region_data['Doanh thu']
        bars = ax.bar(regions, revenue, color=[COLORS['primary'], COLORS['success'], COLORS['warning']][:len(regions)])
        ax.set_title('Doanh thu')
        ax.set_ylabel('VNĐ')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))
        for bar, val in zip(bars, revenue):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8)

        # Pie chart tỷ lệ
        ax = axes[1]
        if 'Tỷ lệ (%)' in region_data.columns:
            ax.pie(region_data['Tỷ lệ (%)'], labels=regions, autopct='%1.1f%%',
                   colors=[COLORS['primary'], COLORS['success'], COLORS['warning']][:len(regions)],
                   startangle=90)
        else:
            ax.pie(revenue, labels=regions, autopct='%1.1f%%',
                   colors=[COLORS['primary'], COLORS['success'], COLORS['warning']][:len(regions)],
                   startangle=90)
        ax.set_title('Tỷ lệ doanh thu')

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ revenue by region: {e}")
        return None


def plot_products_by_region(df):
    """Biểu đồ sản phẩm phổ biến theo vùng miền."""
    if df.empty:
        return None
    try:
        from FUNCTION.location_analysis import popular_products_by_region
        prod_data = popular_products_by_region(df, top_n=5)
        if prod_data is None or prod_data.empty:
            return None

        regions = prod_data.iloc[:, 0].unique()
        n_regions = len(regions)
        fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 6))
        if n_regions == 1:
            axes = [axes]
        fig.suptitle('Top 5 Sản phẩm phổ biến theo Vùng miền', fontsize=14, fontweight='bold')

        for ax, region, color in zip(axes, regions, [COLORS['primary'], COLORS['success'], COLORS['warning']]):
            region_prods = prod_data[prod_data.iloc[:, 0] == region]
            names = region_prods.iloc[:, 1]
            qty = region_prods.iloc[:, 2]
            ax.barh(range(len(names)), qty, color=color, alpha=0.8)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title(region)
            ax.set_xlabel('Số lượng')
            ax.invert_yaxis()

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ products by region: {e}")
        return None


def plot_payment_by_region(df):
    """Biểu đồ phương thức thanh toán theo vùng miền."""
    if df.empty:
        return None
    try:
        from FUNCTION.location_analysis import payment_by_region
        pay_data = payment_by_region(df)
        if pay_data is None or pay_data.empty:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        regions = pay_data.iloc[:, 0]
        payment_cols = pay_data.columns[1:]
        x = np.arange(len(regions))
        width = 0.8 / len(payment_cols)

        for i, col in enumerate(payment_cols):
            ax.bar(x + i * width, pay_data[col], width, label=col,
                   color=PALETTE[i % len(PALETTE)], alpha=0.8)

        ax.set_xticks(x + width * (len(payment_cols) - 1) / 2)
        ax.set_xticklabels(regions)
        ax.set_title('Phương thức Thanh toán theo Vùng miền', fontsize=14, fontweight='bold')
        ax.set_ylabel('Số giao dịch')
        ax.legend(title='PT Thanh toán', fontsize=8)
        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ payment by region: {e}")
        return None


# ============================================================
#  5. BIỂU ĐỒ DỰ ĐOÁN
# ============================================================

def plot_predict_ml(df):
    """Biểu đồ dự đoán doanh thu bằng ML."""
    if df.empty:
        return None
    try:
        from FUNCTION.prediction import predict_revenue_linear
        predictions, metrics = predict_revenue_linear(df, months_ahead=3)
        if predictions is None or predictions.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        months = predictions['Tháng'].astype(str)
        actual = predictions['Doanh thu thực tế']
        predicted = predictions['Doanh thu dự đoán']

        # Actual data
        mask_actual = actual.notna()
        ax.plot(months[mask_actual], actual[mask_actual], 'o-',
                color=COLORS['primary'], label='Thực tế', linewidth=2, markersize=6)

        # Predicted (all)
        ax.plot(months, predicted, 's--',
                color=COLORS['danger'], label='Dự đoán (Linear Regression)',
                linewidth=2, markersize=6, alpha=0.8)

        # Highlight future
        mask_future = actual.isna()
        if mask_future.any():
            ax.fill_between(months[mask_future], predicted[mask_future] * 0.9,
                            predicted[mask_future] * 1.1,
                            color=COLORS['danger'], alpha=0.1, label='Khoảng dự đoán (±10%)')

        ax.set_title(f"Dự đoán Doanh thu (R² = {metrics.get('r2_score', 0):.3f})",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Doanh thu (VNĐ)')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))

        # Trend annotation
        trend = metrics.get('trend', '')
        ax.annotate(f"Xu hướng: {trend}", xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=11, fontweight='bold',
                    color=COLORS['success'] if 'Tăng' in str(trend) else COLORS['danger'])

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ prediction: {e}")
        return None


def plot_product_trends(df):
    """Biểu đồ xu hướng sản phẩm."""
    if df.empty:
        return None
    try:
        from FUNCTION.prediction import product_trend_analysis
        trends = product_trend_analysis(df, top_n=10)
        if trends is None or trends.empty:
            return None

        fig, ax = plt.subplots(figsize=(11, 7))

        products = trends.iloc[:, 0]
        change_col = [c for c in trends.columns if 'thay đổi' in c.lower() or '%' in c]
        if change_col:
            changes = trends[change_col[0]]
        else:
            changes = trends.iloc[:, -1]

        colors = [COLORS['success'] if v >= 0 else COLORS['danger'] for v in changes]
        bars = ax.barh(range(len(products)), changes, color=colors, alpha=0.8)
        ax.set_yticks(range(len(products)))
        ax.set_yticklabels(products, fontsize=9)
        ax.set_xlabel('Độ thay đổi (%)')
        ax.set_title('Xu hướng Sản phẩm (Top 10 thay đổi nhiều nhất)', fontsize=14, fontweight='bold')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='-')
        ax.invert_yaxis()

        for bar, val in zip(bars, changes):
            ax.text(bar.get_width() + (1 if val >= 0 else -1),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}%', va='center', fontsize=9,
                    color=COLORS['success'] if val >= 0 else COLORS['danger'])

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ product trends: {e}")
        return None


def plot_seasonal_analysis(df):
    """Biểu đồ phân tích mùa vụ."""
    if df.empty:
        return None
    try:
        from FUNCTION.prediction import seasonal_analysis
        seasonal = seasonal_analysis(df)
        if seasonal is None or seasonal.empty:
            return None

        fig, ax = plt.subplots(figsize=(11, 6))

        months = seasonal.iloc[:, 0].astype(str)
        revenue = seasonal.iloc[:, 1]
        types = seasonal.iloc[:, 2] if len(seasonal.columns) > 2 else ['Bình thường'] * len(seasonal)

        color_map = {
            'Cao điểm': COLORS['success'],
            'Thấp điểm': COLORS['danger'],
            'Bình thường': COLORS['primary']
        }
        bar_colors = [color_map.get(str(t), COLORS['primary']) for t in types]

        bars = ax.bar(months, revenue, color=bar_colors, alpha=0.8, edgecolor='white')
        mean_val = revenue.mean()
        ax.axhline(mean_val, color='gray', linestyle='--', label=f'Trung bình: {mean_val:,.0f}')

        ax.set_title('Phân tích Mùa vụ - Doanh thu theo Tháng', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Doanh thu TB (VNĐ)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))
        ax.legend()

        # Legend cho colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['success'], label='Cao điểm'),
            Patch(facecolor=COLORS['primary'], label='Bình thường'),
            Patch(facecolor=COLORS['danger'], label='Thấp điểm'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ seasonal: {e}")
        return None


def plot_anomaly_detection(df):
    """Biểu đồ phát hiện bất thường."""
    if df.empty:
        return None
    try:
        from FUNCTION.prediction import detect_anomalies
        anomalies = detect_anomalies(df)
        if anomalies is None or anomalies.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Không phát hiện bất thường nào trong dữ liệu.',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Phát hiện Bất thường')
            ax.axis('off')
            return fig

        # Cần dữ liệu daily revenue đầy đủ để vẽ context
        df_copy = df.copy()
        df_copy['Ngày bán'] = pd.to_datetime(df_copy['Ngày bán'], errors='coerce')
        daily = df_copy.groupby(df_copy['Ngày bán'].dt.date)['Thành tiền'].sum().reset_index()
        daily.columns = ['Ngày', 'Doanh thu']

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(range(len(daily)), daily['Doanh thu'], '-', color=COLORS['primary'],
                alpha=0.5, linewidth=1, label='Doanh thu hàng ngày')

        # Mark anomalies
        anomaly_dates = set(anomalies.iloc[:, 0].astype(str).values)
        for i, row in daily.iterrows():
            if str(row['Ngày']) in anomaly_dates:
                ax.plot(i, row['Doanh thu'], 'o', color=COLORS['danger'],
                        markersize=8, zorder=5)

        ax.set_title(f'Phát hiện Bất thường ({len(anomalies)} ngày)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Doanh thu (VNĐ)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLORS['primary'], label='Bình thường'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['danger'],
                   markersize=8, label=f'Bất thường ({len(anomalies)})'),
        ]
        ax.legend(handles=legend_elements)
        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ anomaly: {e}")
        return None


# ============================================================
#  6. BIỂU ĐỒ HỖ TRỢ QUYẾT ĐỊNH
# ============================================================

def plot_employee_performance(df):
    """Biểu đồ hiệu suất nhân viên."""
    if df.empty:
        return None
    try:
        from FUNCTION.decision_support import employee_performance
        perf = employee_performance(df)
        if perf is None or perf.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Hiệu suất Nhân viên', fontsize=14, fontweight='bold')

        # Top 10 by revenue
        ax = axes[0]
        top10 = perf.nlargest(10, 'Doanh thu')
        eval_col = 'Đánh giá' if 'Đánh giá' in top10.columns else None
        color_map_emp = {'Xuất sắc': COLORS['success'], 'Tốt': COLORS['primary'],
                         'Trung bình': COLORS['warning'], 'Cần cải thiện': COLORS['danger']}
        if eval_col:
            bar_colors = [color_map_emp.get(str(e), COLORS['primary']) for e in top10[eval_col]]
        else:
            bar_colors = COLORS['primary']

        ax.barh(top10['Mã nhân viên'].astype(str), top10['Doanh thu'],
                color=bar_colors, alpha=0.8)
        ax.set_title('Top 10 Nhân viên Doanh thu cao nhất')
        ax.set_xlabel('Doanh thu (VNĐ)')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_vnd))
        ax.invert_yaxis()

        # Distribution by tier
        ax = axes[1]
        if eval_col:
            tier_counts = perf[eval_col].value_counts()
            tier_colors = [color_map_emp.get(str(t), COLORS['primary']) for t in tier_counts.index]
            ax.pie(tier_counts.values, labels=tier_counts.index, autopct='%1.0f%%',
                   colors=tier_colors, startangle=90)
            ax.set_title('Phân bổ đánh giá nhân viên')
        else:
            ax.text(0.5, 0.5, 'Không có dữ liệu đánh giá', ha='center', va='center')
            ax.axis('off')

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ employee performance: {e}")
        return None


def plot_inventory_alerts(df):
    """Biểu đồ cảnh báo tồn kho."""
    if df.empty:
        return None
    try:
        from FUNCTION.decision_support import inventory_alerts
        alerts = inventory_alerts(df)
        if alerts is None or alerts.empty:
            return None

        fig, ax = plt.subplots(figsize=(11, 7))

        alert_col = [c for c in alerts.columns if 'Cảnh báo' in c or 'cảnh báo' in c.lower()]
        alert_col = alert_col[0] if alert_col else alerts.columns[-1]

        color_map_inv = {
            'Bán chạy - Cần nhập thêm': COLORS['success'],
            'Bán chậm - Giảm nhập': COLORS['danger'],
            'Ổn định': COLORS['primary']
        }
        alert_counts = alerts[alert_col].value_counts()
        colors_inv = [color_map_inv.get(str(a), COLORS['info']) for a in alert_counts.index]
        ax.pie(alert_counts.values, labels=alert_counts.index, autopct='%1.0f%%',
               colors=colors_inv, startangle=90, textprops={'fontsize': 11})
        ax.set_title('Báo cáo tồn kho - Phân loại sản phẩm', fontsize=14, fontweight='bold')

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ inventory alerts: {e}")
        return None


def plot_cross_sell(df):
    """Biểu đồ gợi ý Cross-sell."""
    if df.empty:
        return None
    try:
        from FUNCTION.decision_support import cross_sell_recommendations
        recs = cross_sell_recommendations(df, top_n=10)
        if recs is None or recs.empty:
            return None

        fig, ax = plt.subplots(figsize=(11, 7))

        labels = [f"{row.iloc[0]} → {row.iloc[1]}" for _, row in recs.iterrows()]
        values = recs.iloc[:, 2]  # Số KH mua cùng

        ax.barh(range(len(labels)), values, color=COLORS['teal'], alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Số khách hàng mua cùng')
        ax.set_title('Top 10 Gợi ý Cross-sell', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ cross-sell: {e}")
        return None


def plot_kpi_dashboard(df):
    """Biểu đồ tổng hợp KPI toàn cảnh."""
    if df.empty:
        return None
    try:
        from FUNCTION.decision_support import kpi_summary
        kpi = kpi_summary(df)
        if kpi is None:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
        fig.suptitle('Dashboard KPI Tổng Hợp', fontsize=14, fontweight='bold')

        tong_dt = kpi.get('tong_doanh_thu', 0)
        dt_text = f"{tong_dt / 1e9:.2f} Tỷ VNĐ\n({tong_dt:,.0f} đ)" if tong_dt >= 1e9 else f"{tong_dt:,.0f} VNĐ"

        # KPI Cards
        kpi_items = [
            ('Tổng Doanh Thu', dt_text, COLORS['primary']),
            ('Số Giao Dịch', f"{kpi.get('so_giao_dich', 0):,}", COLORS['success']),
            ('Giá Trị TB / Đơn', f"{kpi.get('gia_tri_tb_gd', 0):,.0f} VNĐ", COLORS['warning']),
            ('Số Khách Hàng', f"{kpi.get('so_khach_hang', 0):,}", COLORS['info']),
            ('Số Mặt Hàng', f"{kpi.get('so_san_pham', 0)}", COLORS['purple']),
            ('Tăng Trưởng Doanh Thu', f"{kpi.get('tang_truong_dt', 0):+.1f}%", COLORS['teal']),
        ]

        for ax, (title, value, color) in zip(axes.flat, kpi_items):
            ax.axis('off')
            ax.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.84, transform=ax.transAxes,
                                        facecolor=color, alpha=0.1, edgecolor=color, linewidth=2))
            ax.text(0.5, 0.58, value, ha='center', va='center', fontsize=12,
                    fontweight='bold', color=color, transform=ax.transAxes)
            ax.text(0.5, 0.22, title, ha='center', va='center', fontsize=9.5,
                    fontweight='bold', color='#455a64', transform=ax.transAxes)

        fig.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Lỗi tạo biểu đồ KPI dashboard: {e}")
        return None
