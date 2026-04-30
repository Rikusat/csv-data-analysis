# CSV 可視化ダッシュボード

どんなカラム構成の CSV でも自動解析・可視化できる汎用 Streamlit アプリです。

---

## 機能

| タブ | 内容 |
|------|------|
| 📋 概要 | KPI カード・クイックグラフ |
| 📈 時系列 | 折れ線グラフ（グループ切替・複数指標） |
| 📊 カテゴリ分析 | 棒グラフ・積み上げ棒・円グラフ |
| 🔗 相関分析 | 散布図・相関ヒートマップ |
| 🗂️ データ | テーブル表示・CSV/Excel エクスポート |

---

## セットアップ

```bash
# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run streamlit_app.py
```

ブラウザで `http://localhost:8501` が開きます。

---

## 使い方

1. サイドバーの **「CSVファイルをアップロード」** から CSV を選択する
2. アップロードしない場合は製造業サンプルデータ（デモ）で動作します
3. サイドバーのフィルターで絞り込み → 各タブでグラフを確認
4. **データタブ** からフィルター適用済みデータを CSV / Excel でダウンロード

---

## カラム自動判別ルール

`components/detect.py` の `detect_columns()` が以下の優先順位で判定します。

| 種別 | 判定条件 |
|------|----------|
| **日付** | datetime 型 / `date 日付 month 年 月 time 日時 week` などのキーワード / 文字列が日時として解析できる |
| **数値** | numeric 型 / 80% 以上の値が数値として解析できる |
| **カテゴリ** | `plant line process equipment 拠点 ライン 工程 設備 status 状態 type 区分` などのキーワード / 文字列かつユニーク数 20 以下 |
| **テキスト** | 上記以外 |

カラム判定結果はサイドバーの **「カラム判定結果」** で確認できます。  
意図と異なる場合は CSV のカラム名をキーワードに合わせると精度が上がります。

---

## ファイル構成

```
streamlit_app.py        # エントリーポイント
components/
  detect.py             # カラム自動判別ロジック
  charts.py             # Plotly グラフ生成関数
  kpi.py                # KPI カード表示関数
  filters.py            # フィルター UI 生成関数
requirements.txt
README.md
CLAUDE.md               # AI 修正時の設計ガイドライン
```

---

## Streamlit Cloud へのデプロイ

### 前提
- GitHub リポジトリに本プロジェクトをプッシュ済み
- `requirements.txt` がルートにある
- CSV データファイルはリポジトリに含めない（アップロード方式で対応）

### 手順

1. [share.streamlit.io](https://share.streamlit.io) にアクセスしてログイン
2. **「New app」** をクリック
3. 以下を入力
   - **Repository**: `<GitHub ユーザー名>/<リポジトリ名>`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. **「Deploy!」** をクリック → 数分で公開 URL が発行される

### 注意事項
- アップロードされた CSV はセッションメモリのみに保持され、サーバーには保存されません
- 外部 API へのデータ送信は一切行いません
- 無料プランでは同時接続数に制限があります

---

## デモデータ仕様

| カラム | 説明 |
|--------|------|
| DATE | 日付（過去 30 日間） |
| PLANT | 拠点（東工場 / 西工場） |
| LINE | ライン（LINE-A / B / C） |
| PLAN_QTY | 計画数量 |
| ACTUAL_QTY | 実績数量 |
| NG_QTY | 不良数 |
| 達成率(%) | ACTUAL_QTY / PLAN_QTY × 100 |
| 不良率(%) | NG_QTY / ACTUAL_QTY × 100 |
