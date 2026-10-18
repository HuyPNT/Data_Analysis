import sys
import os

# Fix utf-8 output encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 65)
print("KIỂM TRA CÁC MODULE VÀ CHỨC NĂNG NÂNG CẤP")
print("=" * 65)

# 1. Load Data
print("\n[1/7] Kiểm tra Data Loader...")
from data.data_loader import load_all_sales_data, load_sales_data
df = load_all_sales_data()
assert not df.empty, "Lỗi: Không tải được dữ liệu!"
print(f"  -> Tải thành công {len(df):,} dòng, {len(df.columns)} cột.")

# 2. Data Cleaning
print("\n[2/7] Kiểm tra Data Cleaning...")
from FUNCTION.data_cleaning import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    handle_missing_values,
    detect_duplicates,
    normalize_data,
    data_quality_report,
    full_clean_pipeline
)
report = data_quality_report(df)
print(f"  -> Báo cáo chất lượng: {report['total_rows']:,} dòng, {report['duplicates']} trùng lặp, {report['outliers_count']} ngoại lệ.")
outliers, low, high = detect_outliers_iqr(df)
print(f"  -> IQR Outliers: {len(outliers)} dòng (ngưỡng: {low:,.0f} - {high:,.0f}).")
df_cleaned, clean_report = full_clean_pipeline(df)
print(f"  -> Pipeline làm sạch hoàn tất: còn {len(df_cleaned):,} dòng hợp lệ.")

# 3. Association Rules
print("\n[3/7] Kiểm tra Association Rules...")
from FUNCTION.association_rules import (
    prepare_baskets,
    find_frequent_itemsets,
    generate_rules,
    get_top_product_pairs,
    get_recommendations
)
pairs = get_top_product_pairs(df, top_n=5)
print(f"  -> Top {len(pairs)} cặp sản phẩm mua cùng nhau:")
if not pairs.empty:
    for _, r in pairs.iterrows():
        print(f"     * {r.iloc[0]} + {r.iloc[1]}: {r.iloc[2]} lượt")

rules = generate_rules(df, min_support=0.005, min_confidence=0.1, min_lift=1.0)
if rules is not None and not rules.empty:
    print(f"  -> Tìm được {len(rules)} luật kết hợp.")
    top_rule = rules.iloc[0]
    print(f"     * Luật tiêu biểu: {top_rule['antecedents']} -> {top_rule['consequents']} (Lift: {top_rule['lift']:.2f})")
else:
    print("  -> Không có luật kết hợp ở ngưỡng hiện tại (bình thường nếu giỏ hàng rời rạc).")

recs = get_recommendations(df, df['Tên mặt hàng'].iloc[0], top_n=3)
print(f"  -> Gợi ý kèm cho '{df['Tên mặt hàng'].iloc[0]}': {len(recs)} sản phẩm.")

# 4. Customer Analysis
print("\n[4/7] Kiểm tra Customer Analysis...")
from FUNCTION.customer_analysis import (
    calculate_rfm,
    segment_customers,
    customer_purchase_patterns,
    customer_lifetime_value,
    customer_summary
)
rfm = calculate_rfm(df)
print(f"  -> Tính RFM cho {len(rfm)} khách hàng.")
rfm_seg, seg_summary = segment_customers(df)
if not rfm_seg.empty:
    print(f"  -> Phân khúc KH (K-Means/Fallback):")
    for seg, cnt in rfm_seg['Phân khúc'].value_counts().items():
        print(f"     * {seg}: {cnt} KH")

clv = customer_lifetime_value(df)
print(f"  -> CLV ước tính thành công cho {len(clv)} KH.")
summary_cust = customer_summary(df)
print(f"  -> Tỷ lệ quay lại ước tính: {summary_cust.get('retention_rate_estimate', 0):.1f}%.")

# 5. Location Analysis
print("\n[5/7] Kiểm tra Location Analysis...")
from FUNCTION.location_analysis import (
    revenue_by_region,
    popular_products_by_region,
    payment_by_region,
    regional_comparison
)
rev_region = revenue_by_region(df)
print("  -> Doanh thu theo vùng miền:")
for _, r in rev_region.iterrows():
    print(f"     * {r['Vùng miền']}: {r['Doanh thu']:,.0f} VNĐ ({r['Tỷ lệ (%)']}%)")
pay_reg = payment_by_region(df)
print(f"  -> Bảng thanh toán theo vùng: {pay_reg.shape[0]} vùng x {pay_reg.shape[1]-1} hình thức.")

# 6. Prediction
print("\n[6/7] Kiểm tra Prediction...")
from FUNCTION.prediction import (
    predict_revenue_linear,
    product_trend_analysis,
    seasonal_analysis,
    detect_anomalies,
    revenue_forecast_summary
)
preds, metrics = predict_revenue_linear(df, months_ahead=3)
print(f"  -> Hồi quy tuyến tính: R2 = {metrics.get('r2_score', 0):.3f}, Xu hướng: {metrics.get('trend')}")
trends = product_trend_analysis(df, top_n=5)
print(f"  -> Phân tích xu hướng: {len(trends)} sản phẩm biến động mạnh.")
anomalies = detect_anomalies(df)
print(f"  -> Phát hiện ngoại lệ ngày: {len(anomalies)} ngày biến động bất thường.")

# 7. Decision Support
print("\n[7/7] Kiểm tra Decision Support & Charts...")
from FUNCTION.decision_support import (
    cross_sell_recommendations,
    inventory_alerts,
    employee_performance,
    promotion_suggestions,
    kpi_summary,
    business_insights
)
kpis = kpi_summary(df)
print(f"  -> KPI Doanh thu: {kpis.get('tong_doanh_thu', 0):,.0f} VNĐ | Số GD: {kpis.get('so_giao_dich', 0):,}")
inv = inventory_alerts(df)
print(f"  -> Cảnh báo tồn kho: {len(inv)} sản phẩm đã được gắn nhãn.")
perf = employee_performance(df)
print(f"  -> Đánh giá nhân viên: {len(perf)} nhân sự.")
insights = business_insights(df)
print(f"  -> Tự động sinh {len(insights)} nhận định kinh doanh.")

# Test Charts generation (non-interactive backend)
import matplotlib
matplotlib.use('Agg')
from CHARTS import advanced_charts

fig_kpi = advanced_charts.plot_kpi_dashboard(df)
assert fig_kpi is not None, "Lỗi vẽ biểu đồ KPI Dashboard!"
fig_rules = advanced_charts.plot_association_rules(df)
fig_rfm = advanced_charts.plot_rfm_analysis(df)
fig_region = advanced_charts.plot_revenue_by_region(df)
fig_pred = advanced_charts.plot_predict_ml(df)
print("  -> Kiểm tra vẽ biểu đồ: Tất cả biểu đồ mẫu khởi tạo thành công.")

print("\n" + "=" * 65)
print("✅ TẤT CẢ CÁC MODULE ĐÃ HOẠT ĐỘNG HOÀN HẢO!")
print("=" * 65)
