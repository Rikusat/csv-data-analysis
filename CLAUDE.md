# CLAUDE.md — 設計ガイドライン

このファイルは AI（Claude）が本プロジェクトを修正する際に必ず参照する設計指針です。
機能追加・バグ修正を問わず、以下の原則を守ってください。

---

## アーキテクチャ原則

### 責務の分離（変更禁止）

| ファイル | 責務 | 変えてはいけないこと |
|----------|------|----------------------|
| `streamlit_app.py` | UI レイアウト・タブ制御・データフロー | タブ数（5）、サイドバー構成、`main()` のフロー順序 |
| `components/detect.py` | カラム自動判別のみ | `detect_columns()` の返り値の型（`dict[str, list[str]]`）とキー名 |
| `components/charts.py` | Plotly グラフ生成のみ | 各関数のシグネチャ（引数・返り値 `Optional[go.Figure]`） |
| `components/kpi.py` | KPI カード表示のみ | `render_kpi_cards()` のシグネチャ |
| `components/filters.py` | フィルター UI 生成のみ | `apply_filters()` の返り値（DataFrame） |

> **components 間の直接参照は禁止。** 必ず `streamlit_app.py` を経由してデータを渡すこと。

---

## データフロー（変更禁止）

```
CSVアップロード or デモデータ
        ↓
  detect_columns(df_raw)   ←── col_info を生成
        ↓
  render_*_filters()       ←── サイドバー UI
        ↓
  apply_filters(df_raw, …) ←── フィルター済み df を生成
        ↓
  各タブレンダラー(_tab_*)  ←── df と col_info のみ受け取る
        ↓
  render_*_chart / render_kpi_cards
```

フィルターは **必ず `apply_filters()` を通す**こと。タブ内で直接 df を絞り込まない。

---

## 「汎用性」は最優先

本アプリはどんなカラム構成の CSV でも動作することが設計上の最重要要件です。

- **特定のカラム名をハードコードしない**（例外: デモデータ生成関数のみ許可）
- カラムの存在チェックを必ず行い、ない場合は `st.info()` で案内する
- `pd.to_numeric(series, errors='coerce')` を使って型変換エラーを吸収する

---

## エラーハンドリング規約

- グラフ生成関数（`charts.py`）は必ず `try/except` で囲み、失敗時は `None` を返す
- `None` を受け取った呼び出し側は `st.error()` でユーザーに通知する
- ユーザー向けメッセージは日本語で、原因を具体的に書く
  - 良い例: `「{col}」は数値でないため集計できません。別の列を選択してください。`
  - 悪い例: `エラーが発生しました。`

---

## パフォーマンス規約

- データ読み込みは `@st.cache_data` でキャッシュする
- グラフ用データは `_downsample()` で最大 1,000 点に制限する（`charts.py` に実装済み）
- 10 万行超のデータは `main()` でサンプリングしてから各タブに渡す

---

## UI・デザイン規約

- アクセントカラー: `#2563EB`（変更禁止）
- セクション見出しは `<p class="section-title">` で統一
- KPI カードは `border-left: 4px solid #2563EB` のカード型
- グラフは全て Plotly（`st.line_chart` 等の組み込みチャートは使用禁止）
- フォント: Inter（Google Fonts、`streamlit_app.py` の CSS で読み込み済み）

---

## 追加・変更時のチェックリスト

新機能を追加する前に以下を確認してください。

- [ ] 既存の `detect_columns()` の返り値で対応できるか
- [ ] 特定のカラム名に依存していないか
- [ ] グラフ関数に `try/except` があり `None` を返せるか
- [ ] 大量データでも `_downsample()` が効いているか
- [ ] タブは既存の 5 タブに収まるか（新タブ追加は要相談）
- [ ] `apply_filters()` を経由したデータを使っているか

---

## 半導体工場向け追加機能

### charts.py に追加済みの関数
- `render_spc_chart(df, date_col, value_col, group_col)` — SPC 管理図（±3σ）
- `render_pareto_chart(df, category_col, value_col, agg_method, top_n)` — パレート図

### デモデータ（`_demo_data()`）
- 60日 × 4工程（リソグラフィ/エッチング/CVD/CMP）× 2装置
- カラム: DATE, PROCESS, EQUIPMENT_ID, WAFER_IN, WAFER_OUT, YIELD_RATE, DEFECT_DENSITY, THICKNESS_NM, UNIFORMITY_PCT, UPTIME_PCT, THROUGHPUT_WPH

### detect.py の注意事項
- `_is_date_column` で numeric dtype のカラムは date と誤判定しない制約を追加
  （例: UPTIME_PCT に 'time' が含まれても numeric のまま）
- 日付判定に年範囲チェック（1990–2100）を追加

---

## 禁止事項

- `st.experimental_rerun` の使用（`st.rerun` を使う）
- CSV データのファイル保存・外部 API 送信
- `components/` 間の直接 import
- グラフライブラリを Plotly 以外に変更すること
- `col_info` のキー名（`date` / `category` / `numeric` / `text`）の変更
