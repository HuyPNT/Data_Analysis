"""
Module phân tích giỏ hàng (Market Basket Analysis) và khai phá luật kết hợp (Association Rules)
sử dụng thuật toán Apriori cho dữ liệu bán hàng chuỗi cửa hàng tiện lợi.
"""

from collections import Counter
from itertools import combinations
import logging
import pandas as pd

# Thiết lập logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def _extract_baskets_series(df):
    """
    Hàm nội bộ: Nhóm các giao dịch thành các giỏ hàng ảo theo ('Mã khách hàng', ngày bán).
    Lọc bỏ các khách hàng vãng lai (NaN, KH_UNKNOWN) và chỉ giữ lại giỏ hàng có từ 2 sản phẩm trở lên.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu giao dịch.

    Trả về:
        pd.Series: Series các danh sách sản phẩm theo từng giỏ hàng, hoặc Series rỗng.
    """
    if df is None or df.empty:
        return pd.Series(dtype=object)

    required_cols = ['Mã khách hàng', 'Tên mặt hàng']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Thiếu cột bắt buộc '{col}' để tạo giỏ hàng.")
            return pd.Series(dtype=object)

    # Xác định cột ngày giao dịch
    if 'Ngày' in df.columns:
        date_series = pd.to_datetime(df['Ngày'], errors='coerce').dt.date
    elif 'Ngày bán' in df.columns:
        date_series = pd.to_datetime(df['Ngày bán'], errors='coerce').dt.date
    elif 'Thời gian' in df.columns:
        date_series = pd.to_datetime(df['Thời gian'], errors='coerce').dt.date
    else:
        logger.warning("Không tìm thấy cột ngày ('Ngày', 'Ngày bán', hoặc 'Thời gian') để nhóm giỏ hàng.")
        return pd.Series(dtype=object)

    # Tạo DataFrame làm việc và lọc khách hàng không hợp lệ (NaN, KH_UNKNOWN, ...)
    df_work = df[['Mã khách hàng', 'Tên mặt hàng']].copy()
    df_work['_date'] = date_series

    # Loại bỏ các dòng thiếu thông tin cần thiết
    invalid_cust = {'nan', 'none', 'null', 'kh_unknown', 'n/a', ''}
    cust_str = df_work['Mã khách hàng'].astype(str).str.strip()
    valid_cust_mask = df_work['Mã khách hàng'].notna() & (~cust_str.str.lower().isin(invalid_cust))

    item_str = df_work['Tên mặt hàng'].astype(str).str.strip()
    valid_item_mask = df_work['Tên mặt hàng'].notna() & (item_str != '') & (~item_str.str.lower().isin({'nan', 'none', 'n/a'}))
    valid_date_mask = df_work['_date'].notna()

    df_valid = df_work[valid_cust_mask & valid_item_mask & valid_date_mask].copy()
    if df_valid.empty:
        logger.warning("Không còn dữ liệu hợp lệ sau khi lọc khách hàng và sản phẩm.")
        return pd.Series(dtype=object)

    df_valid['Tên mặt hàng'] = df_valid['Tên mặt hàng'].astype(str).str.strip()

    # Nhóm sản phẩm theo từng khách hàng trong cùng một ngày
    baskets = (
        df_valid.groupby(['Mã khách hàng', '_date'])['Tên mặt hàng']
        .apply(lambda s: sorted(list(set(s))))
    )

    # Lọc giỏ hàng có từ 2 sản phẩm khác nhau trở lên
    baskets = baskets[baskets.apply(len) >= 2]
    return baskets


def prepare_baskets(df):
    """
    Chuẩn bị ma trận giỏ hàng One-Hot Encoded (True/False) cho thuật toán Apriori.
    Mỗi giao dịch đơn lẻ được gom nhóm theo ('Mã khách hàng', ngày bán) để tạo thành giỏ hàng ảo.
    Chỉ giữ lại các giỏ hàng có ít nhất 2 sản phẩm và khách hàng xác định.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu bán hàng.

    Trả về:
        pd.DataFrame: Ma trận nhị phân One-Hot Encoded kiểu boolean, trong đó mỗi dòng
                      là một giỏ hàng và mỗi cột là một mặt hàng.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi prepare_baskets.")
        return pd.DataFrame()

    baskets_series = _extract_baskets_series(df)
    if baskets_series.empty:
        logger.info("Không có giỏ hàng nào hợp lệ (cần ít nhất 2 sản phẩm).")
        return pd.DataFrame()

    # One-hot encoding sử dụng pd.crosstab
    basket_df = baskets_series.reset_index(name='items')
    basket_df['basket_id'] = range(len(basket_df))
    exploded = basket_df.explode('items').reset_index(drop=True)

    basket_encoded_df = pd.crosstab(exploded['basket_id'], exploded['items']).astype(bool)
    basket_encoded_df.columns.name = None
    basket_encoded_df.index.name = None

    logger.info(
        f"Chuẩn bị giỏ hàng thành công: {basket_encoded_df.shape[0]} giỏ hàng, "
        f"{basket_encoded_df.shape[1]} mặt hàng."
    )
    return basket_encoded_df


def find_frequent_itemsets(df, min_support=0.01):
    """
    Tìm các tập mục phổ biến (frequent itemsets) bằng thuật toán Apriori từ thư viện mlxtend.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu bán hàng đầu vào.
        min_support (float): Ngưỡng độ hỗ trợ tối thiểu (mặc định 0.01 = 1%).

    Trả về:
        pd.DataFrame: DataFrame chứa các tập phổ biến với các cột ['itemsets', 'support'],
                      được sắp xếp giảm dần theo support.
    """
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi find_frequent_itemsets.")
        return pd.DataFrame(columns=['itemsets', 'support'])

    try:
        from mlxtend.frequent_patterns import apriori
    except ImportError:
        logger.error("Thư viện 'mlxtend' chưa được cài đặt. Vui lòng cài đặt qua 'pip install mlxtend'.")
        return pd.DataFrame(columns=['itemsets', 'support'])

    basket_encoded = prepare_baskets(df)
    if basket_encoded.empty:
        logger.warning("Ma trận giỏ hàng rỗng, không thể chạy Apriori.")
        return pd.DataFrame(columns=['itemsets', 'support'])

    try:
        frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
        if frequent_itemsets is None or frequent_itemsets.empty:
            logger.info(f"Không tìm thấy tập phổ biến nào với min_support={min_support}.")
            return pd.DataFrame(columns=['itemsets', 'support'])

        # Chuẩn hóa cột trả về theo thứ tự ['itemsets', 'support']
        frequent_itemsets = frequent_itemsets[['itemsets', 'support']].sort_values(
            by='support', ascending=False
        ).reset_index(drop=True)

        logger.info(f"Tìm thấy {len(frequent_itemsets)} tập phổ biến với min_support={min_support}.")
        return frequent_itemsets

    except Exception as e:
        logger.error(f"Lỗi khi thực thi thuật toán Apriori: {e}")
        return pd.DataFrame(columns=['itemsets', 'support'])


def generate_rules(df, min_support=0.01, min_confidence=0.3, min_lift=1.0):
    """
    Sinh các luật kết hợp (Association Rules) từ các tập phổ biến sử dụng mlxtend.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu bán hàng đầu vào.
        min_support (float): Độ hỗ trợ tối thiểu cho Apriori (mặc định 0.01).
        min_confidence (float): Độ tin cậy tối thiểu cho luật (mặc định 0.3).
        min_lift (float): Độ nâng (lift) tối thiểu cho luật (mặc định 1.0).

    Trả về:
        pd.DataFrame: DataFrame chứa danh sách luật kết hợp với các cột:
                      ['antecedents', 'consequents', 'support', 'confidence', 'lift'],
                      được định dạng chuỗi phân tách bởi dấu phẩy và sắp xếp theo lift giảm dần.
    """
    columns_out = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi generate_rules.")
        return pd.DataFrame(columns=columns_out)

    try:
        from mlxtend.frequent_patterns import association_rules
    except ImportError:
        logger.error("Thư viện 'mlxtend' chưa được cài đặt.")
        return pd.DataFrame(columns=columns_out)

    # Tìm tập phổ biến
    frequent_itemsets = find_frequent_itemsets(df, min_support=min_support)
    if frequent_itemsets.empty:
        logger.info("Không có tập phổ biến nào để sinh luật kết hợp.")
        return pd.DataFrame(columns=columns_out)

    try:
        # mlxtend yêu cầu DataFrame có cột 'support' và 'itemsets'
        rules = association_rules(
            frequent_itemsets,
            metric='confidence',
            min_threshold=min_confidence
        )

        if rules is None or rules.empty:
            logger.info("Không tìm thấy luật nào thỏa mãn min_confidence.")
            return pd.DataFrame(columns=columns_out)

        # Lọc theo min_lift
        if 'lift' in rules.columns:
            rules = rules[rules['lift'] >= min_lift]

        if rules.empty:
            logger.info(f"Không có luật nào thỏa mãn min_lift >= {min_lift}.")
            return pd.DataFrame(columns=columns_out)

        # Sắp xếp giảm dần theo lift
        rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)

        # Định dạng antecedents và consequents thành chuỗi thay vì frozenset
        def _format_itemset(itemset):
            if isinstance(itemset, (set, frozenset, list, tuple)):
                return ', '.join(sorted(str(item) for item in itemset))
            return str(itemset)

        rules['antecedents'] = rules['antecedents'].apply(_format_itemset)
        rules['consequents'] = rules['consequents'].apply(_format_itemset)

        result_df = rules[columns_out].copy()
        logger.info(f"Sinh thành công {len(result_df)} luật kết hợp.")
        return result_df

    except Exception as e:
        logger.error(f"Lỗi khi sinh luật kết hợp: {e}")
        return pd.DataFrame(columns=columns_out)


def get_top_product_pairs(df, top_n=10):
    """
    Tìm top N cặp sản phẩm thường xuyên được mua cùng nhau nhất trong cùng một ngày.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu bán hàng.
        top_n (int): Số lượng cặp sản phẩm cần lấy (mặc định 10).

    Trả về:
        pd.DataFrame: DataFrame có các cột:
                      ['Sản phẩm 1', 'Sản phẩm 2', 'Số lần mua cùng']
    """
    columns_out = ['Sản phẩm 1', 'Sản phẩm 2', 'Số lần mua cùng']
    if df is None or df.empty:
        logger.warning("DataFrame rỗng khi gọi get_top_product_pairs.")
        return pd.DataFrame(columns=columns_out)

    baskets_series = _extract_baskets_series(df)
    if baskets_series.empty:
        logger.info("Không có giỏ hàng nào có từ 2 sản phẩm trở lên.")
        return pd.DataFrame(columns=columns_out)

    # Đếm số lần xuất hiện của từng cặp sản phẩm
    pair_counts = Counter()
    for basket in baskets_series:
        unique_items = sorted(list(set(basket)))
        if len(unique_items) >= 2:
            for pair in combinations(unique_items, 2):
                pair_counts[pair] += 1

    if not pair_counts:
        return pd.DataFrame(columns=columns_out)

    top_pairs = pair_counts.most_common(top_n)
    rows = [
        {
            'Sản phẩm 1': pair[0][0],
            'Sản phẩm 2': pair[0][1],
            'Số lần mua cùng': int(pair[1])
        }
        for pair in top_pairs
    ]

    pairs_df = pd.DataFrame(rows)
    logger.info(f"Tìm thấy top {len(pairs_df)} cặp sản phẩm mua cùng nhiều nhất.")
    return pairs_df


def get_recommendations(df, product_name, top_n=5):
    """
    Gợi ý các sản phẩm liên quan khi khách hàng mua một sản phẩm cụ thể,
    dựa trên luật kết hợp (Association Rules) và tần suất đồng xuất hiện trong giỏ hàng.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu bán hàng.
        product_name (str): Tên sản phẩm khách hàng đang xem hoặc mua.
        top_n (int): Số lượng sản phẩm gợi ý tối đa (mặc định 5).

    Trả về:
        pd.DataFrame: DataFrame có các cột:
                      ['Sản phẩm gợi ý', 'Confidence', 'Lift']
    """
    columns_out = ['Sản phẩm gợi ý', 'Confidence', 'Lift']
    if df is None or df.empty or not product_name:
        logger.warning("Dữ liệu hoặc tên sản phẩm không hợp lệ khi gọi get_recommendations.")
        return pd.DataFrame(columns=columns_out)

    target_prod = str(product_name).strip()
    target_lower = target_prod.lower()

    # Thử lấy gợi ý từ sinh luật kết hợp (áp dụng ngưỡng hỗ trợ linh hoạt)
    try:
        rules = generate_rules(df, min_support=0.005, min_confidence=0.1, min_lift=1.0)
    except Exception as e:
        logger.warning(f"Không thể sinh luật qua generate_rules: {e}")
        rules = pd.DataFrame()

    recs = []
    if rules is not None and not rules.empty:
        for _, row in rules.iterrows():
            antecedent_items = [i.strip().lower() for i in str(row['antecedents']).split(',')]
            if target_lower in antecedent_items:
                consequent_items = [i.strip() for i in str(row['consequents']).split(',')]
                for item in consequent_items:
                    if item.lower() != target_lower:
                        recs.append({
                            'Sản phẩm gợi ý': item,
                            'Confidence': float(row['confidence']),
                            'Lift': float(row['lift'])
                        })

    # Nếu luật kết hợp không có kết quả hoặc mlxtend không có sẵn, tính toán trực tiếp từ giỏ hàng
    if not recs:
        baskets_series = _extract_baskets_series(df)
        if not baskets_series.empty:
            total_baskets = len(baskets_series)
            baskets_with_target = [
                b for b in baskets_series
                if any(item.lower() == target_lower for item in b)
            ]
            count_target = len(baskets_with_target)

            if count_target > 0:
                # Đếm tần suất các sản phẩm khác cùng xuất hiện
                co_occur = Counter()
                for b in baskets_with_target:
                    for item in b:
                        if item.lower() != target_lower:
                            co_occur[item] += 1

                # Đếm tần suất tổng quát của các sản phẩm đó
                overall_counts = Counter()
                for b in baskets_series:
                    for item in b:
                        if item in co_occur:
                            overall_counts[item] += 1

                for item, co_count in co_occur.items():
                    conf = co_count / count_target
                    p_item = overall_counts[item] / total_baskets
                    lift = conf / p_item if p_item > 0 else 1.0

                    recs.append({
                        'Sản phẩm gợi ý': item,
                        'Confidence': float(conf),
                        'Lift': float(lift)
                    })

    if not recs:
        logger.info(f"Không tìm thấy gợi ý nào cho sản phẩm '{product_name}'.")
        return pd.DataFrame(columns=columns_out)

    recs_df = pd.DataFrame(recs)

    # Loại bỏ trùng lặp nếu cùng gợi ý một sản phẩm (giữ luật có Lift cao nhất)
    recs_df = recs_df.sort_values(by=['Lift', 'Confidence'], ascending=[False, False])
    recs_df = recs_df.drop_duplicates(subset=['Sản phẩm gợi ý'], keep='first')
    recs_df = recs_df.head(top_n).reset_index(drop=True)

    logger.info(f"Đã tạo {len(recs_df)} gợi ý cho sản phẩm '{product_name}'.")
    return recs_df[columns_out]
