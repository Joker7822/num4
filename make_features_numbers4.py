import pandas as pd
import numpy as np
import ast
import re

# 入力 / 出力パス（同じフォルダに置く場合）
INPUT_PATH = "evaluation_result.csv"
OUTPUT_PATH = "features_from_evaluation_result.csv"

def parse_number_list(s):
    """文字列で保存された [2, 7, 9] のようなリストを Python の list[int] に変換する"""
    if pd.isna(s):
        return []
    if isinstance(s, (list, tuple)):
        return list(s)

    s = str(s).strip()
    if not s:
        return []

    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        return [int(v)]
    except (ValueError, SyntaxError):
        # 万が一 "2,7,9" みたいな形式でも頑張ってパース
        parts = [p for p in re.split(r"[^0-9]+", s) if p]
        return [int(p) for p in parts]

def expand_number_features(series, prefix):
    """
    予測番号や当選本数字の list[int] から、
    個別の番号や統計量を特徴量として展開した DataFrame を返す
    """
    nums = series.apply(parse_number_list)

    # 実データの最大長まで展開（上限は安全のため 6）
    max_len = nums.map(len).max()
    max_len = min(max_len, 6)

    cols = {}
    for i in range(max_len):
        cols[f"{prefix}_{i+1}"] = nums.apply(
            lambda x, i=i: x[i] if len(x) > i else np.nan
        )

    # 統計量
    cols[f"{prefix}_len"] = nums.apply(len)
    cols[f"{prefix}_sum"] = nums.apply(lambda x: sum(x) if x else np.nan)
    cols[f"{prefix}_min"] = nums.apply(lambda x: min(x) if x else np.nan)
    cols[f"{prefix}_max"] = nums.apply(lambda x: max(x) if x else np.nan)
    cols[f"{prefix}_range"] = cols[f"{prefix}_max"] - cols[f"{prefix}_min"]
    cols[f"{prefix}_mean"] = nums.apply(lambda x: float(np.mean(x)) if x else np.nan)
    cols[f"{prefix}_std"] = nums.apply(
        lambda x: float(np.std(x)) if len(x) > 1 else 0.0
    )

    return pd.DataFrame(cols), nums

def main(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    # CSV 読み込み
    df = pd.read_csv(input_path)

    # 日付特徴量
    df["抽せん日"] = pd.to_datetime(df["抽せん日"])
    df["year"] = df["抽せん日"].dt.year
    df["month"] = df["抽せん日"].dt.month
    df["day"] = df["抽せん日"].dt.day
    df["dayofweek"] = df["抽せん日"].dt.dayofweek  # 月曜=0

    # 予測番号 / 当選本数字 の展開
    pred_features, pred_nums = expand_number_features(df["予測番号"], "pred")
    true_features, true_nums = expand_number_features(df["当選本数字"], "true")

    # 一致数を自前計算した特徴量（検算＋特徴）
    def count_match(p, t):
        # Numbers4でも重複を正しく数える（multiset一致数）
        from collections import Counter
        cp, ct = Counter(p), Counter(t)
        return sum(min(cp[d], ct[d]) for d in (cp.keys() | ct.keys()))

    match_count = [count_match(p, t) for p, t in zip(pred_nums, true_nums)]
    match_df = pd.DataFrame({
        "match_count_calc": match_count,
        "hit_flag": np.array(match_count) > 0,
    })

    # 元の数値列も特徴として利用
    base_numeric = df[["一致数", "信頼度"]].copy()
    # 例: 「一致数」は目的変数として使う前提でラベル名っぽくリネーム
    base_numeric = base_numeric.rename(columns={"一致数": "一致数_label"})

    # カテゴリ列のワンホット化
    cat_df = pd.get_dummies(
        df[["出力元", "予測番号インデックス"]],
        prefix=["model", "pred_idx"]
    )

    # すべて連結して特徴量テーブル完成
    feature_df = pd.concat(
        [
            df[["抽せん日", "year", "month", "day", "dayofweek"]],
            base_numeric,
            pred_features,
            true_features,
            match_df,
            cat_df,
        ],
        axis=1,
    )

    # 保存
    feature_df.to_csv(output_path, index=False)
    print(f"特徴量を {output_path} に保存しました。")
    print("feature_df の先頭5行：")
    print(feature_df.head())

if __name__ == "__main__":
    main()
