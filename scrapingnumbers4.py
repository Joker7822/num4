# -*- coding: utf-8 -*-
"""
Mizuho銀行 ナンバーズ4 抽せん結果スクレイパ（堅牢版）
- PC/SPどちらのテーブルにも対応（.js-lottery-temp-pc / -sp / フォールバック）
- 中身(textContent)が入るまで待機 + リロード再試行
- 既存CSVに追記（同一「抽せん日」をスキップ）＆日付昇順整列
- 既存CSVの列名を正規化（セット系 & 販売実績額）してから整形
"""

import csv
import os
import re
import time
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers4/index.html"
CSV_PATH = "numbers4.csv"

# 列名は常にこれで固定（ここが「正」となる）
FIELDNAMES = [
    "抽せん日",
    "本数字",
    "回別",
    "ストレート",
    "ボックス",
    "セット(ストレート)",
    "セット(ボックス)",
    "販売実績額",
]

# 旧バージョンで作られたCSVの列名を、新しい正式名に揃えるためのマッピング
COLUMN_ALIASES = {
    "セット（ストレート）": "セット(ストレート)",
    "セット（ボックス）": "セット(ボックス)",
    "ミニ": "販売実績額",  # 万が一 Numbers3 っぽい列名が混ざっていた場合の保険
}


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,2600")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=ja-JP,ja")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    return driver


def _find_first(scope_el, selectors):
    for css in selectors:
        try:
            el = scope_el.find_element(By.CSS_SELECTOR, css)
            if el:
                return el
        except Exception:
            continue
    return None


def _filled(el):
    try:
        return el is not None and el.get_attribute("textContent").strip() != ""
    except Exception:
        return False


def wait_for_tables(driver, timeout_total: int = 45, retries: int = 2):
    """
    PC/SPいずれかのテーブルが現れ、先頭テーブルの date/issue/number が埋まるまで待つ。
    必要に応じて再読み込みを数回試行。
    """
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            wait = WebDriverWait(driver, timeout_total)

            # ローディング文言の消失（あれば）
            try:
                wait.until(
                    EC.invisibility_of_element_located(
                        (By.CSS_SELECTOR, ".js-now-loading")
                    )
                )
            except Exception:
                pass

            # どれかのテーブルが出るまで待機
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".section__table-wrap .js-lottery-temp-pc, "
                                      ".section__table-wrap .js-lottery-temp-sp, "
                                      ".section__table-wrap table.section__table")
                )
            )

            # 中身が入るまでポーリング
            t_end = time.time() + timeout_total
            while time.time() < t_end:
                tables = driver.find_elements(
                    By.CSS_SELECTOR,
                    ".section__table-wrap .js-lottery-temp-pc, "
                    ".section__table-wrap .js-lottery-temp-sp, "
                    ".section__table-wrap table.section__table",
                )
                tables = [t for t in tables if t.is_displayed()] or tables
                if not tables:
                    time.sleep(0.5)
                    continue

                t0 = tables[0]
                date_el = _find_first(
                    t0, [".js-lottery-date-pc", ".js-lottery-date"]
                )
                issue_el = _find_first(
                    t0, [".js-lottery-issue-pc", ".js-lottery-issue"]
                )
                num_el = _find_first(
                    t0, [".js-lottery-number-pc", ".js-lottery-number"]
                )

                if (
                    date_el
                    and issue_el
                    and num_el
                    and all(_filled(x) for x in (date_el, issue_el, num_el))
                ):
                    return  # 準備OK
                time.sleep(0.5)

            # タイムアウト → 次の試行のためリロード
            if attempt <= retries:
                driver.refresh()
                driver.execute_script(
                    "window.scrollTo(0, document.body.scrollHeight/2);"
                )
                time.sleep(1.2)
                continue
            else:
                raise TimeoutError("テーブルが埋まる前にタイムアウトしました。")

        except Exception:
            if attempt <= retries:
                driver.refresh()
                time.sleep(1.0)
                continue
            raise


def parse_tables(driver):
    """
    ページ内にある Numbers4 テーブルから、
    1 行＝1 抽せん分の dict を作って返す。
    """
    tables = driver.find_elements(
        By.CSS_SELECTOR,
        ".section__table-wrap .js-lottery-temp-pc, "
        ".section__table-wrap .js-lottery-temp-sp, "
        ".section__table-wrap table.section__table",
    )
    tables = [t for t in tables if t.is_displayed()] or tables

    results = []
    for t in tables:
        date_el = _find_first(t, [".js-lottery-date-pc", ".js-lottery-date"])
        issue_el = _find_first(t, [".js-lottery-issue-pc", ".js-lottery-issue"])
        num_el = _find_first(t, [".js-lottery-number-pc", ".js-lottery-number"])
        if not (date_el and issue_el and num_el):
            continue

        date_text = date_el.get_attribute("textContent").strip()
        issue_text = issue_el.get_attribute("textContent").strip()
        num_text = num_el.get_attribute("textContent").strip()
        if not (date_text and issue_text and re.search(r"\d", num_text or "")):
            continue

        # 日付
        try:
            draw_date = datetime.strptime(date_text, "%Y年%m月%d日").strftime(
                "%Y-%m-%d"
            )
        except Exception:
            m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", date_text)
            if not m:
                continue
            y, mo, d = map(int, m.groups())
            draw_date = f"{y:04d}-{mo:02d}-{d:02d}"

        # 当せん数字（4桁）
        digits = [int(d) for d in re.findall(r"\d", num_text)]
        if len(digits) != 4:
            continue
        main_number = str(digits)

        # 賞金（ストレート / ボックス / セット(ストレート) / セット(ボックス) / 販売実績額）
        prize_rows = t.find_elements(
            By.CSS_SELECTOR, "tr.js-lottery-prize-pc, tr.js-lottery-prize"
        )
        prizes = []
        for pr in prize_rows:
            strongs = pr.find_elements(
                By.CSS_SELECTOR, "td strong.section__text--bold, td strong"
            )
            val = None
            if strongs:
                txt = strongs[-1].get_attribute("textContent").strip()
                if "円" in txt:
                    try:
                        val = int(
                            txt.replace(",", "")
                            .replace("円", "")
                            .strip()
                        )
                    except Exception:
                        val = None
            prizes.append(val)

        # 5個揃うように埋める
        while len(prizes) < 5:
            prizes.append(None)

        results.append(
            {
                "回別": issue_text,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": prizes[0],
                "ボックス": prizes[1],
                "セット(ストレート)": prizes[2],
                "セット(ボックス)": prizes[3],
                "販売実績額": prizes[4],
            }
        )
    return results


def load_existing(csv_path: str):
    """
    既存CSVを読み込み：
    - 列名を正規化（COLUMN_ALIASES で rename）
    - 足りない列を追加
    - FIELDNAMES の順に並べ替え
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # ファイルが無い場合は空DFを返す
        empty = pd.DataFrame(columns=FIELDNAMES)
        return empty, set(), FIELDNAMES

    # 列名の別名を正規化
    df = df.rename(columns=COLUMN_ALIASES)

    # 足りない列を追加
    for col in FIELDNAMES:
        if col not in df.columns:
            df[col] = pd.NA

    # 日付を str 化してキー集合を作る
    df["抽せん日"] = df["抽せん日"].astype(str)
    existing_dates = set(df["抽せん日"].tolist())

    # 列順を揃えておく
    df = df[FIELDNAMES]

    return df, existing_dates, FIELDNAMES


def append_and_sort(csv_path: str, fieldnames, new_rows):
    """
    new_rows を CSV に追記し、その後ファイル全体を読み直して
    日付 + 回別 でソートして保存し直す。
    """
    if not new_rows:
        return 0

    # 余計なキーを削り、足りないキーを None で埋める
    sanitized = []
    for row in new_rows:
        sanitized.append({k: row.get(k) for k in fieldnames})

    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(sanitized)

    # 一度全部読み直して列名を正規化＆ソート
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_ALIASES)

    for col in fieldnames:
        if col not in df.columns:
            df[col] = pd.NA

    try:
        df["抽せん日_dt"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        df.sort_values(["抽せん日_dt", "回別"], inplace=True)
        df.drop(columns=["抽せん日_dt"], inplace=True)
    except Exception:
        df.sort_values(["抽せん日", "回別"], inplace=True)

    df = df[fieldnames]
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return len(sanitized)


def main():
    driver = build_driver(headless=True)
    try:
        driver.get(URL)
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight/3);"
        )
        time.sleep(0.8)

        wait_for_tables(driver, timeout_total=45, retries=2)
        scraped = parse_tables(driver)
    finally:
        driver.quit()

    if not scraped:
        print(
            "[INFO] 表が取得できませんでした（ページ構造変更/一時的な読込失敗の可能性）。"
        )
        return

    _, existing_dates, fieldnames = load_existing(CSV_PATH)

    # 既存は「抽せん日」単位でユニークにしていたのでそれに合わせる
    new_rows = [row for row in scraped if row["抽せん日"] not in existing_dates]

    saved = append_and_sort(CSV_PATH, fieldnames, new_rows)
    if saved:
        print(
            f"[INFO] {saved}件を保存し、日付順に並び替えました。CSV: {CSV_PATH}"
        )
    else:
        print("[INFO] 既存CSVと同一日のため、新規保存はありません。")

    for row in new_rows:
        print(row)


if __name__ == "__main__":
    main()
