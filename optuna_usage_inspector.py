#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optuna_usage_inspector.py

Numbers3_predictions.csv と numbers3.csv が
optuna_search_resumable.py の評価で「実際に使われる行」が何行かを可視化します。

Usage:
  python optuna_usage_inspector.py \
    --backfill Numbers3_predictions.csv \
    --csv numbers3.csv \
    --out optuna_eval_rows.csv \
    --print-dates

出力:
- 標準出力: 各ステップの有効行数と内積（実評価対象行数）
- --out     で指定した CSV に、実評価対象の抽せん日リストを書き出し（任意）
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Optional, List

import pandas as pd


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def _parse_numbers3_row_to_cands(row: pd.Series) -> List[List[int]]:
    cands: List[List[int]] = []
    for i in range(1, 6):
        k = f"予測{i}"
        if k not in row or pd.isna(row[k]):
            continue
        nums = [int(x) for x in re.findall(r"\d", str(row[k]))]
        nums = [n for n in nums if 0 <= n <= 9]
        if len(nums) >= 3:
            cands.append(nums[:3])
    return cands


def _extract_numbers3_winners(row: pd.Series) -> Optional[List[int]]:
    if "本数字" in row.index:
        nums = [int(x) for x in re.findall(r"\d", str(row["本数字"]))]
        nums = [n for n in nums if 0 <= n <= 9]
        if len(nums) >= 3:
            return nums[:3]
    nums = [int(x) for x in re.findall(r"\d", " ".join(map(str, row.values)))]
    nums = [n for n in nums if 0 <= n <= 9]
    return nums[:3] if len(nums) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", default="Numbers3_predictions.csv")
    ap.add_argument("--csv", default="numbers3.csv")
    ap.add_argument("--out", default="optuna_eval_rows.csv")
    ap.add_argument("--print-dates", action="store_true")
    args = ap.parse_args()

    bf = _read_csv_any(args.backfill)
    df = _read_csv_any(args.csv)

    if "抽せん日" not in bf.columns:
        raise SystemExit("backfill 側に '抽せん日' 列が必要です。")
    if "抽せん日" not in df.columns:
        raise SystemExit("csv 側に '抽せん日' 列が必要です。")

    bf = bf.copy()
    df = df.copy()
    bf["抽せん日"] = pd.to_datetime(bf["抽せん日"], errors="coerce")
    df["抽せん日"] = pd.to_datetime(df["抽せん日"], errors="coerce")

    bf = bf.dropna(subset=["抽せん日"]).reset_index(drop=True)
    df = df.dropna(subset=["抽せん日"]).reset_index(drop=True)

    # backfill: cand_list を作成
    if "予測1" not in bf.columns:
        raise SystemExit("Numbers3_predictions.csv を想定しています（列 '予測1..5' が必要）。")
    bf["cand_list"] = bf.apply(_parse_numbers3_row_to_cands, axis=1)
    bf["cand_ok"] = bf["cand_list"].map(lambda x: len(x) > 0)

    # truth: winners を抽出
    df["winners"] = df.apply(_extract_numbers3_winners, axis=1)
    df["win_ok"] = df["winners"].notna()

    # 集計
    print(f"backfill 総行数: {len(bf)}")
    print(f"backfill: 抽出OK行: {int(bf['cand_ok'].sum())}")
    print(f"truth   総行数: {len(df)}")
    print(f"truth  : 抽出OK行: {int(df['win_ok'].sum())}")

    merged = pd.merge(
        bf[bf["cand_ok"]][["抽せん日"]],
        df[df["win_ok"]][["抽せん日"]],
        on="抽せん日",
        how="inner"
    ).drop_duplicates().sort_values("抽せん日")

    print(f"Optuna の実評価対象行（内積）: {len(merged)}")

    # 出力
    if args.out:
        merged.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[saved] 評価対象日の一覧を {args.out} に保存しました。")

    if args.print_dates:
        # 画面にも日付を表示
        for d in merged["抽せん日"].dt.strftime("%Y-%m-%d").tolist():
            print(d)

if __name__ == "__main__":
    main()
