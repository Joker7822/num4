import csv
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os

# === Chrome設定 ===
options = Options()
options.add_argument("--headless=new")  # 新ヘッドレス
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1280,2000")
options.add_argument(
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 20)

url = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"
driver.get(url)

data = []

try:
    # ページの主要テーブルが描画されるまで待機（PC向けテーブル）
    table = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
    )
    # 行単位で処理（tbody > tr）
    rows = wait.until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "tbody tr")
        )
    )

    for i, row in enumerate(rows, start=1):
        try:
            # 行内で必要要素を探す（クラス名は変わりやすいのでフォールバックあり）
            def q(css):
                els = row.find_elements(By.CSS_SELECTOR, css)
                return els[0] if els else None

            el_date  = q(".js-lottery-date-pc, .section__date, .date, time")
            el_issue = q(".js-lottery-issue-pc, .issue, .section__issue")
            el_num   = q(".js-lottery-number-pc, .number, .section__number")

            # 見出し・空行スキップ
            if not el_date or not el_issue or not el_num:
                continue

            date_text = el_date.get_attribute("textContent").strip()
            issue_text = el_issue.get_attribute("textContent").strip()
            num_text = el_num.get_attribute("textContent").strip()

            if not date_text or not re.search(r"\d", date_text):
                # 空や見出しの場合はスキップ
                # print(f"[DEBUG] 行 {i}: 日付が空/非数値（スキップ）")
                continue

            # 日付パース
            # 例: 2025年08月18日 → 2025-08-18
            draw_date = datetime.strptime(date_text, "%Y年%m月%d日").strftime("%Y-%m-%d")

            # 回別
            draw_number = issue_text

            # 本数字（数字だけ抽出して配列化）
            digits = [int(d) for d in re.findall(r"\d", num_text)]
            main_number = str(digits)

            # 賞金はその行の中の strong を収集（上位から5件）
            prize_strongs = row.find_elements(By.CSS_SELECTOR, "strong")
            # 余計な strong を弾くため「円」を含むテキストに限定
            prize_vals = []
            for s in prize_strongs:
                t = s.get_attribute("textContent").strip()
                if "円" in t:
                    try:
                        prize_vals.append(int(t.replace(",", "").replace("円", "").strip()))
                    except:
                        prize_vals.append(None)

            # 期待順: ストレート, ボックス, セット(スト), セット(ボ), ミニ
            # 足りなければ None で埋める
            while len(prize_vals) < 5:
                prize_vals.append(None)

            data.append({
                "回別": draw_number,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": prize_vals[0],
                "ボックス": prize_vals[1],
                "セット(ストレート)": prize_vals[2],
                "セット(ボックス)": prize_vals[3],
                "ミニ": prize_vals[4],
            })

        except Exception as e:
            # 行単位で握りつぶして先へ
            # print(f"[WARN] 行 {i} でエラー: {e}")
            continue

finally:
    driver.quit()

# === 保存処理 ===
csv_path = "numbers3.csv"
try:
    existing = pd.read_csv(csv_path)
    existing_dates = existing["抽せん日"].astype(str).tolist()
    fieldnames = existing.columns.tolist()
except FileNotFoundError:
    existing = pd.DataFrame()
    existing_dates = []
    fieldnames = ["抽せん日", "本数字", "回別", "ストレート", "ボックス", "セット(ストレート)", "セット(ボックス)", "ミニ"]

# 新しいデータ（同一日付除外）
new_rows = [row for row in data if row["抽せん日"] not in existing_dates]

if new_rows:
    write_header = existing.empty or not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)

    # 並び替え（昇順）
    df = pd.read_csv(csv_path)
    df.sort_values("抽せん日", inplace=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[INFO] {len(new_rows)}件を保存し、日付順に並び替えました。")

# === 結果出力 ===
for row in new_rows:
    print(row)
