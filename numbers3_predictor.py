# -*- coding: utf-8 -*-
"""
Numbers3 再チャレンジ用の軽量・純粋予測スクリプト。
- 過去の「的中データ注入」や Forced 系の依存を完全排除
- 直近ウィンドウ（例: 120 抽せん）に強く適応
- 再現性のある評価（ウォークフォワード）と、次回予測の保存

実行例:
    python numbers3_retry.py --csv numbers3.csv --window 120 --k 5 --half_life 60 \
        --eval_last 200 --mode box

出力:
- console: 評価サマリ（的中率/期待収益など）
- Numbers3_predictions.csv: 次回抽せん日の上位 k 予測と信頼度
- numbers3_retry_eval.csv: ウォークフォワード評価詳細

必要列（どちらか存在すればOK）:
- 抽せん日, 本数字
  または
- 抽せん日, 当選本数字
"""

from __future__ import annotations
import argparse
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# ===== ユーティリティ =====

def parse_numbers(x) -> List[int]:
    """文字列/リストから 3 桁の整数配列へ。余分は無視、欠損は空配列。
    例: "[3, 5, 8]" -> [3,5,8], "3 5 8" -> [3,5,8]
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [int(v) for v in x if str(v).isdigit()][:3]
    s = str(x).strip().replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    toks = [t for t in s.replace(",", " ").split() if t.isdigit()]
    out = [int(t) for t in toks][:3]
    return out if len(out) == 3 else []


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 列名ゆらぎ対応
    num_col = "本数字" if "本数字" in df.columns else ("当選本数字" if "当選本数字" in df.columns else None)
    if num_col is None:
        raise ValueError("numbers3.csv に '本数字' もしくは '当選本数字' 列が必要です")
    df[num_col] = df[num_col].apply(parse_numbers)
    df = df[df[num_col].apply(len) == 3].copy()
    if df.empty:
        raise ValueError("有効な当選データがありません（3桁が見つからない）")
    # 日付
    if "抽せん日" not in df.columns:
        raise ValueError("'抽せん日' 列が必要です")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"]).sort_values("抽せん日").reset_index(drop=True)
    df.rename(columns={num_col: "本数字"}, inplace=True)
    return df


# ===== スコアリング（直近期重み + ペア共起） =====

@dataclass
class ScorerConfig:
    half_life_days: float = 60.0   # 直近期重みの半減期
    pair_weight: float = 0.6       # ペア共起の寄与
    digit_weight: float = 1.0      # 単独出現頻度の寄与


def make_recency_weights(dates: pd.Series, ref_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """指数減衰の重み w = 0.5 ** (Δ日 / half_life)"""
    delta = (ref_date - dates).dt.days.clip(lower=0).astype(float)
    w = 0.5 ** (delta / max(1.0, half_life_days))
    return w.values


def build_stats(df_win: pd.DataFrame, end_date: pd.Timestamp, cfg: ScorerConfig):
    weights = make_recency_weights(df_win["抽せん日"], end_date, cfg.half_life_days)
    # 各桁の頻度
    digit_scores = np.zeros(10, dtype=float)
    # ペアの頻度（順不同）
    pair_scores: Dict[Tuple[int, int], float] = {}
    for w, nums in zip(weights, df_win["本数字" ].values):
        a, b, c = nums
        for d in (a, b, c):
            digit_scores[d] += w
        for (x, y) in itertools.combinations(sorted([a, b, c]), 2):
            pair_scores[(x, y)] = pair_scores.get((x, y), 0.0) + w
    # 正規化（スケール依存を軽減）
    if digit_scores.sum() > 0:
        digit_scores = digit_scores / (digit_scores.sum() + 1e-9)
    if pair_scores:
        s = sum(pair_scores.values()) + 1e-9
        pair_scores = {k: v / s for k, v in pair_scores.items()}
    return digit_scores, pair_scores


def combo_score(triplet: Tuple[int, int, int], digit_scores: np.ndarray, pair_scores: Dict[Tuple[int, int], float], cfg: ScorerConfig) -> float:
    a, b, c = sorted(triplet)
    ds = digit_scores[a] + digit_scores[b] + digit_scores[c]
    ps = pair_scores.get((a, b), 0.0) + pair_scores.get((a, c), 0.0) + pair_scores.get((b, c), 0.0)
    return cfg.digit_weight * ds + cfg.pair_weight * ps


def rank_candidates(digit_scores: np.ndarray, pair_scores: Dict[Tuple[int,int], float], k: int, ban_exact: List[int]|None, cfg: ScorerConfig):
    # 上位 m 桁から 3-組み合わせを列挙
    m = 7  # 十分な多様性を確保
    top_digits = np.argsort(-digit_scores)[:m]
    cand = []
    for comb in itertools.combinations(sorted(top_digits.tolist()), 3):
        if ban_exact and sorted(ban_exact) == list(comb):
            continue
        s = combo_score(comb, digit_scores, pair_scores, cfg)
        cand.append((list(comb), s))
    cand.sort(key=lambda x: -x[1])
    top = cand[:k]
    # スコアを 0-1 にスケールして「信頼度」っぽく表示
    if top:
        scores = np.array([s for _, s in top])
        lo, hi = scores.min(), scores.max()
        conf = [float((s - lo) / (hi - lo + 1e-9)) * 0.3 + 0.65 for s in scores]  # 0.65〜0.95
        return [(nums, round(c, 3)) for (nums, _), c in zip(top, conf)]
    return []


# ===== 的中判定 =====

def prize_kind(pred: List[int], actual: List[int], mode: str = "box") -> str:
    # mode="box": 並びは問わず（ボックス判定）/ mode="straight": 完全一致
    if mode == "straight":
        return "ストレート" if pred == actual else "はずれ"
    # ボックス or ミニ（後ろ2桁一致）も併記
    if sorted(pred) == sorted(actual):
        return "ボックス"
    if pred[1:] == actual[1:]:
        return "ミニ"
    return "はずれ"


# ===== ウォークフォワード評価 =====

def walk_forward_eval(df: pd.DataFrame, window: int, k: int, cfg: ScorerConfig, mode: str, eval_last: int = 200) -> pd.DataFrame:
    rows = []
    last_idx = len(df) - 1
    start_eval = max(0, last_idx - eval_last)
    for t in range(start_eval + 1, last_idx + 1):
        end_date = df.loc[t - 1, "抽せん日"]
        win_lo = max(0, t - 1 - window + 1)
        df_win = df.iloc[win_lo: t]
        digit_scores, pair_scores = build_stats(df_win, end_date, cfg)
        last_actual = df.iloc[t - 1]["本数字"]
        preds = rank_candidates(digit_scores, pair_scores, k=k, ban_exact=last_actual, cfg=cfg)
        actual = df.iloc[t]["本数字"]
        # 集計
        hit_any = False
        kind_any = "はずれ"
        for (pnums, conf) in preds:
            kind = prize_kind(pnums, actual, mode=mode)
            if kind != "はずれ":
                hit_any = True
                kind_any = kind
                break
        rows.append({
            "抽せん日": df.iloc[t]["抽せん日"],
            "実際": actual,
            "予測": [p for p, _ in preds],
            "最初のヒット": kind_any,
            "候補数": len(preds)
        })
    return pd.DataFrame(rows)


# ===== 期待収益計算 =====
PAYOUT = {"ストレート": 105000, "ボックス": 15000, "ミニ": 4000, "はずれ": 0}
COST_PER_TICKET = 200


def summarize_eval(df_eval: pd.DataFrame) -> pd.DataFrame:
    if df_eval.empty:
        return pd.DataFrame()
    counts = df_eval["最初のヒット"].value_counts().to_dict()
    total_trials = int(df_eval["候補数"].sum())
    # 1 抽せんあたり k 口購入として試算
    total_cost = int(df_eval["候補数"].sum() * COST_PER_TICKET)
    total_payout = 0
    for kind, n in counts.items():
        total_payout += PAYOUT.get(kind, 0) * n
    profit = total_payout - total_cost
    out = {
        "件数": len(df_eval),
        "ストレート": counts.get("ストレート", 0),
        "ボックス": counts.get("ボックス", 0),
        "ミニ": counts.get("ミニ", 0),
        "はずれ": counts.get("はずれ", 0),
        "総コスト(¥)": total_cost,
        "当選合計(¥)": total_payout,
        "損益(¥)": profit,
    }
    return pd.DataFrame([out])


# ===== 次回予測の生成と保存 =====

def next_predictions(df: pd.DataFrame, window: int, k: int, cfg: ScorerConfig, mode: str) -> Tuple[pd.Timestamp, List[Tuple[List[int], float]]]:
    last_date = df["抽せん日"].max()
    # 次営業日（Numbers3 は平日抽せん）簡易版：土日を飛ばす
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += pd.Timedelta(days=1)
    df_win = df.tail(window)
    digit_scores, pair_scores = build_stats(df_win, last_date, cfg)
    ban_exact = df.iloc[-1]["本数字"]
    preds = rank_candidates(digit_scores, pair_scores, k=k, ban_exact=ban_exact, cfg=cfg)
    return next_date, preds


def save_predictions_csv(date_ts: pd.Timestamp, preds: List[Tuple[List[int], float]], model_name: str = "RecencyPairV1", path: str = "Numbers3_predictions.csv"):
    row = {"抽せん日": date_ts.strftime("%Y-%m-%d")}
    for i, (nums, conf) in enumerate(preds[:5], 1):
        row[f"予測{i}"] = ", ".join(map(str, nums))
        row[f"信頼度{i}"] = round(float(conf), 3)
        row[f"出力元{i}"] = model_name
    try:
        if os.path.exists(path):
            ex = pd.read_csv(path, encoding="utf-8-sig")
            ex = ex[ex["抽せん日"] != row["抽せん日"]]
            ex = pd.concat([ex, pd.DataFrame([row])], ignore_index=True)
            ex.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8-sig")
    except Exception:
        # 環境差異で失敗しても致命ではない
        pass


# ===== main =====
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="numbers3.csv のパス")
    ap.add_argument("--window", type=int, default=120, help="直近ウィンドウ（抽せん数）")
    ap.add_argument("--k", type=int, default=5, help="一度に出す候補数")
    ap.add_argument("--half_life", type=float, default=60.0, help="直近期重みの半減期（日）")
    ap.add_argument("--mode", type=str, default="box", choices=["box", "straight"], help="評価モード")
    ap.add_argument("--eval_last", type=int, default=200, help="評価に使う末尾の抽せん数")
    args = ap.parse_args()

    df = load_data(args.csv)
    cfg = ScorerConfig(half_life_days=args.half_life)

    # 評価
    df_eval = walk_forward_eval(df, window=args.window, k=args.k, cfg=cfg, mode=args.mode, eval_last=args.eval_last)
    df_eval.to_csv("numbers3_retry_eval.csv", index=False)
    summary = summarize_eval(df_eval)
    if not summary.empty:
        print("\n=== 評価サマリ ===")
        print(summary.to_string(index=False))

    # 次回予測
    date_ts, preds = next_predictions(df, window=args.window, k=args.k, cfg=cfg, mode=args.mode)
    save_predictions_csv(date_ts, preds, model_name="RecencyPairV1")
    print("\n=== 次回予測 ===")
    print("抽せん日:", date_ts.strftime("%Y-%m-%d"))
    for i, (nums, conf) in enumerate(preds, 1):
        print(f"#{i}: {nums}  信頼度={conf:.3f}")


if __name__ == "__main__":
    main()
