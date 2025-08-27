# -*- coding: utf-8 -*-
"""
Numbers3 再チャレンジ：完全版（短期順応＋共起モデル）

目的:
  - 2025-06 以降の失速を踏まえ、過去実績注入や Forced 系依存を排除し、
    「直近の実勢」に強く追従する純粋予測パイプラインを提供する。
  - ウォークフォワード評価、収支シミュレーション、次回予測の保存までを
    1ファイルで完結させる。

設計のポイント:
  - データリークを避ける: 未来情報の混入や“当たりデータ注入”は禁止。
  - 短期順応: 直近ウィンドウの指数減衰重み（半減期指定）。
  - シンプルで頑健: 桁頻度・ペア共起（順不同）＋（任意で）順序付きペア（ミニ対策）。
  - 多様性: 候補間の距離（ハミング/ジャッカード）で最低多様性を確保可能。
  - 評価の再現性: ウォークフォワードで k 候補を毎回固定ルールで選定し、
    収支を賞金テーブル＆購入単価から算出。

使用例:
    python numbers3_retry_full.py \
        --csv numbers3.csv \
        --window 120 \
        --k 6 \
        --half_life 60 \
        --mode box \
        --eval_last 200 \
        --min_diversity 1 \
        --skip_last_exact 1 \
        --out_predictions Numbers3_predictions.csv \
        --out_eval numbers3_retry_eval.csv

必要列（どちらか）:
  - 抽せん日, 本数字
    もしくは
  - 抽せん日, 当選本数字

出力:
  - Console: 評価サマリ（各等級 / 総コスト / 当選合計 / 損益）
  - numbers3_retry_eval.csv: ウォークフォワード評価の詳細
  - Numbers3_predictions.csv: 次回抽せん日の上位 k 予測と信頼度（0.65–0.95）

注意:
  - 本スクリプトは確率ゲームに対して“統計的ヒューリスティック”を用いるもので、
    期待値の保証はありません。投資判断は自己責任で行ってください。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================
# ユーティリティ
# =============================

def parse_numbers(x) -> List[int]:
    """文字列/リストから3桁の整数配列へ正規化。余分は切り捨て、欠損は空配列。
    例: "[3, 5, 8]" -> [3,5,8] / "3 5 8" -> [3,5,8]
    """
    if x is None:
        return []
    if isinstance(x, list):
        nums = [int(v) for v in x if str(v).isdigit()]
        return nums[:3] if len(nums) >= 3 else []
    s = (
        str(x)
        .strip()
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .replace("/", " ")
    )
    toks = [t for t in s.replace(",", " ").split() if t.isdigit()]
    nums = [int(t) for t in toks]
    return nums[:3] if len(nums) >= 3 else []


def load_numbers3_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # 列名ゆらぎ
    num_col = (
        "本数字"
        if "本数字" in df.columns
        else ("当選本数字" if "当選本数字" in df.columns else None)
    )
    if num_col is None:
        raise ValueError("入力CSVに '本数字' もしくは '当選本数字' 列が必要です。")
    if "抽せん日" not in df.columns:
        raise ValueError("入力CSVに '抽せん日' 列が必要です。")

    df = df.copy()
    df[num_col] = df[num_col].apply(parse_numbers)
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")
    df = df.dropna(subset=["抽せん日"])  # 変換失敗行を除外
    df = df[df[num_col].apply(len) == 3].copy()
    df = df.sort_values("抽せん日").reset_index(drop=True)
    df.rename(columns={num_col: "本数字"}, inplace=True)
    if df.empty:
        raise ValueError("有効な3桁データが見つかりませんでした。")
    return df


# =============================
# スコアリング & 候補生成
# =============================

@dataclass
class ScorerConfig:
    half_life_days: float = 60.0  # 直近期重みの半減期（日）
    pair_weight: float = 0.7      # ペア共起の寄与（順不同）
    digit_weight: float = 1.0     # 単独出現頻度の寄与
    opair_weight: float = 0.2     # 順序付きペアの寄与（ミニ対策・弱め）
    equal_weight: bool = False  # True なら全期間を同等重みで集計



def make_recency_weights(dates: pd.Series, ref_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    delta_days = (ref_date - dates).dt.days.clip(lower=0).astype(float)
    half = max(1.0, half_life_days)
    return np.power(0.5, delta_days / half)


def build_stats(
    df_win: pd.DataFrame,
    end_date: pd.Timestamp,
    cfg: ScorerConfig,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    \"\"\"直近ウィンドウで桁頻度・順不同ペア・順序付きペアを指数減衰で集計。
    戻り値:
      digit_scores: shape(10,) 0-1 正規化
      pair_scores:  {(i,j): score}, i<j
      opair_scores: {(a,b): score}, 位置順（後ろ2桁対策など）
    \"\"\"
    w = np.ones(len(df_win), dtype=float) if cfg.equal_weight else make_recency_weights(df_win[\"抽せん日\"], end_date, cfg.half_life_days)
    digit_scores = np.zeros(10, dtype=float)
    pair_scores: Dict[Tuple[int, int], float] = {}
    opair_scores: Dict[Tuple[int, int], float] = {}

    for weight, nums in zip(w, df_win["本数字" ].values):
        a, b, c = nums
        # 桁
        digit_scores[a] += weight
        digit_scores[b] += weight
        digit_scores[c] += weight
        # 順不同ペア（a<b<c を強制）
        x, y, z = sorted([a, b, c])
        for p in ((x, y), (x, z), (y, z)):
            pair_scores[p] = pair_scores.get(p, 0.0) + weight
        # 順序付きペア（例: 後ろ2桁一致の参考）
        opair_scores[(b, c)] = opair_scores.get((b, c), 0.0) + weight

    # 正規化
    if digit_scores.sum() > 0:
        digit_scores = digit_scores / (digit_scores.sum() + 1e-9)
    if pair_scores:
        s = sum(pair_scores.values()) + 1e-9
        pair_scores = {k: v / s for k, v in pair_scores.items()}
    if opair_scores:
        s2 = sum(opair_scores.values()) + 1e-9
        opair_scores = {k: v / s2 for k, v in opair_scores.items()}

    return digit_scores, pair_scores, opair_scores


def combo_score(
    triplet: Tuple[int, int, int],
    digit_scores: np.ndarray,
    pair_scores: Dict[Tuple[int, int], float],
    opair_scores: Dict[Tuple[int, int], float],
    cfg: ScorerConfig,
) -> float:
    a, b, c = sorted(triplet)
    ds = digit_scores[a] + digit_scores[b] + digit_scores[c]
    ps = pair_scores.get((a, b), 0.0) + pair_scores.get((a, c), 0.0) + pair_scores.get((b, c), 0.0)
    # 後ろ2桁が (b,c) になりがちな傾向の弱い加点
    ops = opair_scores.get((b, c), 0.0)
    return cfg.digit_weight * ds + cfg.pair_weight * ps + cfg.opair_weight * ops


def hamming_distance(a: Iterable[int], b: Iterable[int]) -> int:
    aa, bb = list(a), list(b)
    # 長さが違うことは基本ないが、念のため揃える
    m = min(len(aa), len(bb))
    return sum(1 for i in range(m) if aa[i] != bb[i]) + abs(len(aa) - len(bb))


def jaccard_distance(a: Iterable[int], b: Iterable[int]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - (inter / union if union else 0.0)


def rank_candidates(
    digit_scores: np.ndarray,
    pair_scores: Dict[Tuple[int, int], float],
    opair_scores: Dict[Tuple[int, int], float],
    k: int,
    ban_exact: Optional[List[int]],
    cfg: ScorerConfig,
    min_diversity: int = 0,
    diversity_metric: str = "hamming",
    top_m_digits: int = 7,
) -> List[Tuple[List[int], float]]:
    """上位 m 桁から3桁の全組合せを列挙→スコア上位を多様性制約付きで選抜。
    返り値は [(nums, confidence)] 。
    """
    top_idx = np.argsort(-digit_scores)[: max(3, min(10, top_m_digits))]
    pool = sorted(top_idx.tolist())
    scored: List[Tuple[List[int], float]] = []
    for comb in itertools.combinations(pool, 3):
        if ban_exact and sorted(ban_exact) == list(comb):
            continue
        s = combo_score(comb, digit_scores, pair_scores, opair_scores, cfg)
        scored.append((list(comb), s))

    if not scored:
        return []

    scored.sort(key=lambda x: -x[1])

    # スコア→信頼度（0.65〜0.95）へ線形スケーリング
    scores = np.array([s for _, s in scored], dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    def to_conf(val: float) -> float:
        if hi - lo < 1e-12:
            return 0.8
        return 0.65 + 0.30 * (val - lo) / (hi - lo)

    selected: List[Tuple[List[int], float]] = []
    for nums, s in scored:
        # 多様性制約
        if min_diversity > 0 and selected:
            dist_ok = True
            for prev, _ in selected:
                if diversity_metric == "jaccard":
                    d = jaccard_distance(nums, prev)
                    if d < min_diversity / 3.0:  # 0〜1 の距離に粗対応
                        dist_ok = False
                        break
                else:
                    d = hamming_distance(nums, prev)
                    if d < min_diversity:
                        dist_ok = False
                        break
            if not dist_ok:
                continue
        selected.append((nums, round(to_conf(s), 3)))
        if len(selected) >= k:
            break

    return selected


# =============================
# 判定・評価・出力
# =============================

PAYOUT = {"ストレート": 105000, "ボックス": 15000, "ミニ": 4000, "はずれ": 0}
COST_PER_TICKET = 200


def prize_kind(pred: List[int], actual: List[int], mode: str = "box") -> str:
    """mode="straight": 並び完全一致のみをヒット
       mode="box": 並び無視で一致をボックス判定、後ろ2桁一致をミニとして扱う
    """
    if mode == "straight":
        return "ストレート" if pred == actual else "はずれ"
    if sorted(pred) == sorted(actual):
        return "ボックス"
    if pred[1:] == actual[1:]:
        return "ミニ"
    return "はずれ"


def walk_forward_eval(
    df: pd.DataFrame,
    window: int,
    k: int,
    cfg: ScorerConfig,
    mode: str,
    eval_last: int = 200,
    min_diversity: int = 0,
    skip_last_exact: bool = True,
    top_m_digits: int = 7,
) -> pd.DataFrame:
    rows = []
    last_idx = len(df) - 1
    start_eval = max(0, last_idx - eval_last)

    for t in range(start_eval + 1, last_idx + 1):
        prev_date = df.loc[t - 1, \"抽せん日\"]
        if cfg.equal_weight:
            df_win = df.iloc[:t]
        else:
            lo = max(0, t - 1 - window + 1)
            df_win = df.iloc[lo:t]
        digit_scores, pair_scores, opair_scores = build_stats(df_win, prev_date, cfg)
        ban = df.iloc[t - 1]["本数字"] if skip_last_exact else None
        preds = rank_candidates(
            digit_scores,
            pair_scores,
            opair_scores,
            k=k,
            ban_exact=ban,
            cfg=cfg,
            min_diversity=min_diversity,
            diversity_metric="hamming",
            top_m_digits=top_m_digits,
        )
        actual = df.iloc[t]["本数字"]

        first_hit = "はずれ"
        which = None
        for i, (pnums, _conf) in enumerate(preds):
            kind = prize_kind(pnums, actual, mode=mode)
            if kind != "はずれ":
                first_hit = kind
                which = i + 1
                break

        rows.append(
            {
                "抽せん日": df.iloc[t]["抽せん日"],
                "実際": actual,
                "予測": [p for p, _ in preds],
                "最初のヒット": first_hit,
                "ヒット順位": which,
                "候補数": len(preds),
            }
        )

    return pd.DataFrame(rows)


def summarize_eval(df_eval: pd.DataFrame, k: int) -> pd.DataFrame:
    if df_eval.empty:
        return pd.DataFrame()
    cnt = df_eval["最初のヒット"].value_counts().to_dict()
    n_days = len(df_eval)
    total_cost = int(n_days * k * COST_PER_TICKET)
    total_payout = sum(PAYOUT.get(kind, 0) * cnt.get(kind, 0) for kind in PAYOUT)
    profit = total_payout - total_cost
    hit_rate_any = (n_days - cnt.get("はずれ", 0)) / max(1, n_days)

    out = {
        "評価対象日数": n_days,
        "候補/日": k,
        "ストレート": cnt.get("ストレート", 0),
        "ボックス": cnt.get("ボックス", 0),
        "ミニ": cnt.get("ミニ", 0),
        "はずれ": cnt.get("はずれ", 0),
        "何かしら当たり率": round(hit_rate_any, 4),
        "総コスト(¥)": total_cost,
        "当選合計(¥)": total_payout,
        "損益(¥)": profit,
    }
    return pd.DataFrame([out])


# =============================
# 次回予測（平日のみ）
# =============================

def next_business_day(last_date: pd.Timestamp) -> pd.Timestamp:
    d = last_date + pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5=土,6=日
        d += pd.Timedelta(days=1)
    return d


def next_predictions(
    df: pd.DataFrame,
    window: int,
    k: int,
    cfg: ScorerConfig,
    min_diversity: int = 0,
    skip_last_exact: bool = True,
    top_m_digits: int = 7,
) -> Tuple[pd.Timestamp, List[Tuple[List[int], float]]]:
    last_date = df["抽せん日"].max()
    df_win = df if cfg.equal_weight else df.tail(window)
    digit_scores, pair_scores, opair_scores = build_stats(df_win, last_date, cfg)
    ban = df.iloc[-1]["本数字"] if skip_last_exact else None
    preds = rank_candidates(
        digit_scores,
        pair_scores,
        opair_scores,
        k=k,
        ban_exact=ban,
        cfg=cfg,
        min_diversity=min_diversity,
        diversity_metric="hamming",
        top_m_digits=top_m_digits,
    )
    return next_business_day(last_date), preds


def save_predictions_csv(
    date_ts: pd.Timestamp,
    preds: List[Tuple[List[int], float]],
    model_name: str,
    path: str,
    append: bool = True,
) -> None:
    row = {"抽せん日": date_ts.strftime("%Y-%m-%d")}
    for i, (nums, conf) in enumerate(preds[:5], 1):
        row[f"予測{i}"] = ", ".join(map(str, nums))
        row[f"信頼度{i}"] = float(conf)
        row[f"出力元{i}"] = model_name
    df_row = pd.DataFrame([row])

    if append and os.path.exists(path):
        try:
            ex = pd.read_csv(path, encoding="utf-8-sig")
            ex = ex[ex["抽せん日"] != row["抽せん日"]]
            ex = pd.concat([ex, df_row], ignore_index=True)
            ex.to_csv(path, index=False, encoding="utf-8-sig")
            return
        except Exception:
            pass
    df_row.to_csv(path, index=False, encoding="utf-8-sig")


# =============================
# CLI
# =============================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="numbers3.csv のパス")
    p.add_argument("--window", type=int, default=120, help="直近ウィンドウ（抽せん数）")
    p.add_argument("--k", type=int, default=5, help="一度に出す候補数")
    p.add_argument("--half_life", type=float, default=60.0, help="直近期重みの半減期（日）")
    p.add_argument("--mode", choices=["box", "straight"], default="box", help="評価モード")
    p.add_argument("--eval_last", type=int, default=200, help="評価に使う末尾の抽せん数")
    p.add_argument("--min_diversity", type=int, default=1, help="候補間の最小ハミング距離(0=無効)")
    p.add_argument("--skip_last_exact", type=int, default=1, help="直前当選の完全コピーを除外するか 1/0")
    p.add_argument("--top_m_digits", type=int, default=7, help="候補生成で使う上位桁の数(3-10)")
    p.add_argument("--out_predictions", default="Numbers3_predictions.csv")
    p.add_argument("--out_eval", default="numbers3_retry_eval.csv")
    p.add_argument("--json_summary", default=None, help="評価サマリを JSON でも保存するパス")
    p.add_argument("--model_name", default="RecencyPairFullV1")
    args = p.parse_args()

    df = load_numbers3_csv(args.csv)
    cfg = ScorerConfig(half_life_days=args.half_life, equal_weight=bool(args.equal_weight))

    # ===== 評価 =====
    df_eval = walk_forward_eval(
        df=df,
        window=args.window,
        k=args.k,
        cfg=cfg,
        mode=args.mode,
        eval_last=args.eval_last,
        min_diversity=args.min_diversity,
        skip_last_exact=bool(args.skip_last_exact),
        top_m_digits=max(3, min(10, args.top_m_digits)),
    )
    df_eval.to_csv(args.out_eval, index=False)

    summary = summarize_eval(df_eval, k=args.k)
    if not summary.empty:
        print("\n=== 評価サマリ ===")
        print(summary.to_string(index=False))
        if args.json_summary:
            with open(args.json_summary, "w", encoding="utf-8") as f:
                json.dump(summary.iloc[0].to_dict(), f, ensure_ascii=False, indent=2)

    # ===== 次回予測 =====
    date_ts, preds = next_predictions(
        df=df,
        window=args.window,
        k=args.k,
        cfg=cfg,
        min_diversity=args.min_diversity,
        skip_last_exact=bool(args.skip_last_exact),
        top_m_digits=max(3, min(10, args.top_m_digits)),
    )
    save_predictions_csv(
        date_ts=date_ts,
        preds=preds,
        model_name=args.model_name,
        path=args.out_predictions,
    )

    print("\n=== 次回予測 ===")
    print("抽せん日:", date_ts.strftime("%Y-%m-%d"))
    for i, (nums, conf) in enumerate(preds, 1):
        print(f"#{i}: {nums}  信頼度={conf:.3f}")


if __name__ == "__main__":
    main()
