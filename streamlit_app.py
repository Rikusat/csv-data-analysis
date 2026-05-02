"""
半導体工場 生産データ分析ダッシュボード
Entry point — run with: streamlit run streamlit_app.py
"""

import csv
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    render_category_chart,
    render_correlation_heatmap,
    render_pareto_chart,
    render_scatter_chart,
    render_spc_chart,
    render_timeseries_chart,
)
from components.detect import detect_columns
from components.filters import apply_filters, render_category_filters, render_date_filter
from components.kpi import render_kpi_cards

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="半導体工場 生産データ分析",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet">
    <style>
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        .section-title {
            font-size: 16px; font-weight: 600; color: #111827;
            border-left: 4px solid #2563EB; padding-left: 12px;
            margin: 1.5rem 0 1rem 0;
        }
        .demo-banner {
            background: #eff6ff; border: 1px solid #bfdbfe;
            border-radius: 8px; padding: 10px 16px;
            color: #1d4ed8; font-size: 14px; margin-bottom: 1rem;
        }
        div[data-testid="stSidebarContent"] { background: #f8fafc; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Demo data（半導体工場） ────────────────────────────────────
@st.cache_data(ttl=86400)
def _demo_data() -> pd.DataFrame:
    """
    Generate 60-day semiconductor fab demo data.
    4 processes × 2 equipment IDs, with realistic process variation
    and a few intentional out-of-control events for SPC demonstration.
    """
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=60, freq='D')
    processes = ['リソグラフィ', 'エッチング', 'CVD', 'CMP']
    equipment_map = {
        'リソグラフィ': ['LITHO-01', 'LITHO-02'],
        'エッチング':   ['ETCH-01',  'ETCH-02'],
        'CVD':         ['CVD-01',   'CVD-02'],
        'CMP':         ['CMP-01',   'CMP-02'],
    }
    # Baseline thickness per process (nm)
    thickness_base = {
        'リソグラフィ': 120, 'エッチング': 80, 'CVD': 500, 'CMP': 300,
    }

    rows = []
    for d in dates:
        for proc in processes:
            for eq in equipment_map[proc]:
                wafer_in = int(np.random.randint(20, 26))
                yield_rate = np.clip(np.random.normal(96.0, 1.5), 88, 100)
                wafer_out = int(wafer_in * yield_rate / 100)
                defect_density = np.clip(np.random.normal(0.35, 0.12), 0.05, 2.0)
                thickness = np.random.normal(thickness_base[proc], thickness_base[proc] * 0.01)
                uniformity = np.clip(np.random.normal(98.5, 0.8), 95, 100)
                uptime = np.clip(np.random.normal(92.0, 4.0), 70, 100)
                throughput = np.clip(np.random.normal(22.0, 2.0), 15, 30)

                # Inject occasional out-of-control events (≈5% chance)
                if np.random.random() < 0.05:
                    defect_density *= np.random.uniform(3, 5)
                    yield_rate -= np.random.uniform(5, 12)
                    yield_rate = max(yield_rate, 70)

                rows.append({
                    'DATE':              d.strftime('%Y-%m-%d'),
                    'PROCESS':           proc,
                    'EQUIPMENT_ID':      eq,
                    'WAFER_IN':          wafer_in,
                    'WAFER_OUT':         wafer_out,
                    'YIELD_RATE':        round(yield_rate, 2),
                    'DEFECT_DENSITY':    round(defect_density, 3),
                    'THICKNESS_NM':      round(thickness, 1),
                    'UNIFORMITY_PCT':    round(uniformity, 2),
                    'UPTIME_PCT':        round(uptime, 1),
                    'THROUGHPUT_WPH':    round(throughput, 1),
                })

    return pd.DataFrame(rows)


# ── File loader helpers ───────────────────────────────────────

def _strip_numeric_fmt(s: pd.Series) -> pd.Series:
    """Strip currency prefixes, thousands commas, and percent suffixes."""
    return (
        s.str.strip()
         .str.replace(r'^[¥$€£]\s*', '', regex=True)
         .str.replace(r'[,，]', '', regex=True)
         .str.replace(r'%\s*$', '', regex=True)
    )


def _try_parse_numeric(series: pd.Series) -> pd.Series:
    """Return numeric series if ≥80 % of non-null values parse as numbers (after cleanup)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return series
    direct = pd.to_numeric(non_null, errors='coerce')
    if direct.notna().sum() / len(non_null) >= 0.8:
        return pd.to_numeric(series, errors='coerce')
    cleaned = _strip_numeric_fmt(non_null.astype(str))
    if pd.to_numeric(cleaned, errors='coerce').notna().sum() / len(non_null) >= 0.8:
        return pd.to_numeric(_strip_numeric_fmt(series.astype(str)), errors='coerce')
    return series


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and convert formatted numeric strings to float in place."""
    df = df.copy()
    # Strip whitespace and BOM, deduplicate names
    cols = [str(c).strip().lstrip('﻿') for c in df.columns]
    seen: dict = {}
    clean: list = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            clean.append(f'{c}_{seen[c]}')
        else:
            seen[c] = 1
            clean.append(c)
    df.columns = clean
    # include both legacy object dtype and pandas 3.x StringDtype
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = _try_parse_numeric(df[col])
    return df


@st.cache_data
def _load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel. Auto-detects encoding and delimiter. Returns clean DataFrame."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'csv'

    if ext in ('xlsx', 'xls'):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
            return _normalize_dataframe(df)
        except Exception as exc:
            raise ValueError(f"Excelファイルの読み込みに失敗しました: {exc}") from exc

    # CSV: try encodings, sniff delimiter
    for enc in ('utf-8', 'utf-8-sig', 'shift-jis', 'cp932'):
        try:
            raw = file_bytes[:8192].decode(enc, errors='ignore')
            try:
                dialect = csv.Sniffer().sniff(raw, delimiters=',\t;|')
                sep = dialect.delimiter
            except csv.Error:
                sep = ','
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=sep)
            # If only 1 column detected with ',', retry with other separators
            if df.shape[1] <= 1:
                for alt in ('\t', ';', '|'):
                    try:
                        df2 = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=alt)
                        if df2.shape[1] > df.shape[1]:
                            df = df2
                    except Exception:
                        pass
            return _normalize_dataframe(df)
        except Exception:
            continue

    raise ValueError(
        "ファイルの読み込みに失敗しました。"
        "UTF-8 / Shift-JIS / CP932 のいずれかで保存されたファイルを使用してください。"
    )


# ── Export helpers ────────────────────────────────────────────
@st.cache_data
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


@st.cache_data
def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='データ')
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────
def _render_sidebar():
    """Render sidebar and return (df_raw, col_info, is_demo)."""
    st.sidebar.title("🔬 半導体工場 分析")
    st.sidebar.markdown("---")

    uploaded = st.sidebar.file_uploader(
        "CSV / Excel をアップロード",
        type=['csv', 'xlsx'],
        help="CSV（UTF-8 / Shift-JIS / CP932 / タブ・セミコロン区切り）または Excel（.xlsx）対応",
    )

    is_demo = uploaded is None

    if is_demo:
        df_raw = _demo_data()
    else:
        with st.spinner("ファイルを読み込み中..."):
            try:
                df_raw = _load_file(uploaded.read(), uploaded.name)
            except ValueError as exc:
                st.sidebar.error(str(exc))
                df_raw = _demo_data()
                is_demo = True

    st.sidebar.markdown(
        f"**行数:** {len(df_raw):,}　**列数:** {df_raw.shape[1]}"
    )

    if not is_demo:
        with st.sidebar.expander("📄 データプレビュー（先頭5行）", expanded=False):
            st.dataframe(df_raw.head(5), use_container_width=True)

    st.sidebar.markdown("---")

    col_info = detect_columns(df_raw)

    with st.sidebar.expander("🔍 カラム判定結果", expanded=False):
        label_map = {
            'date':     '📅 日付',
            'category': '🏷️ カテゴリ',
            'numeric':  '🔢 数値',
            'text':     '📝 テキスト',
        }
        for role, cols in col_info.items():
            if cols:
                st.markdown(f"**{label_map[role]}**")
                for c in cols:
                    st.markdown(f"- `{c}`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### フィルター")
    if st.sidebar.button("🔄 フィルターをリセット", key="reset_filters"):
        for col in col_info['category']:
            k = f"filter_cat_{col}"
            if k in st.session_state:
                st.session_state[k] = 'すべて'
        if "filter_date_range" in st.session_state:
            del st.session_state["filter_date_range"]
        st.rerun()

    return df_raw, col_info, is_demo


# ── Tab renderers ─────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame, col_info: dict) -> None:
    st.markdown('<p class="section-title">主要 KPI</p>', unsafe_allow_html=True)

    agg_method = st.selectbox(
        "集計方法", ['平均', '合計', '最大', '最小'], key='kpi_agg'
    )
    date_col = col_info['date'][0] if col_info['date'] else None
    render_kpi_cards(df, col_info['numeric'], date_col=date_col, agg_method=agg_method)

    if col_info['date'] and col_info['numeric']:
        st.markdown(
            '<p class="section-title">歩留まり・欠陥トレンド（クイックビュー）</p>',
            unsafe_allow_html=True,
        )
        quick_metrics = col_info['numeric'][:2]
        fig = render_timeseries_chart(
            df, col_info['date'][0], col_info['numeric'],
            selected_metrics=quick_metrics,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if col_info['category'] and col_info['numeric']:
        st.markdown(
            '<p class="section-title">工程・装置別集計（クイックビュー）</p>',
            unsafe_allow_html=True,
        )
        fig = render_category_chart(df, col_info['category'][0], col_info['numeric'][0])
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def _tab_timeseries(df: pd.DataFrame, col_info: dict) -> None:
    if not col_info['date']:
        st.info("日付列が検出されませんでした。CSVに日付列を含めてください。")
        return
    if not col_info['numeric']:
        st.info("数値列が検出されませんでした。")
        return

    date_col = col_info['date'][0]

    # ── トレンドグラフ ────────────────────────────────────────
    st.markdown('<p class="section-title">トレンドグラフ</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_metrics = st.multiselect(
            "指標を選択（複数可）",
            col_info['numeric'],
            default=col_info['numeric'][:2],
            key='ts_metrics',
        )
    with c2:
        group_options = ['なし'] + col_info['category']
        sel = st.selectbox("グループ列", group_options, key='ts_group')
        group_col = None if sel == 'なし' else sel
    with c3:
        ts_agg = st.selectbox("集計方法", ['平均', '合計', '最大', '最小'], key='ts_agg')
        ts_agg_en = {'平均': 'mean', '合計': 'sum', '最大': 'max', '最小': 'min'}[ts_agg]

    if not selected_metrics:
        st.warning("指標を1つ以上選択してください。")
    else:
        fig = render_timeseries_chart(df, date_col, col_info['numeric'], group_col, selected_metrics, ts_agg_en)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("グラフを生成できませんでした。データを確認してください。")

    # ── SPC 管理図 ────────────────────────────────────────────
    st.markdown('<p class="section-title">SPC 管理図（±3σ 制御限界）</p>', unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        spc_metric = st.selectbox(
            "監視指標", col_info['numeric'], key='spc_metric'
        )
    with sc2:
        spc_group_opts = ['なし'] + col_info['category']
        spc_grp_sel = st.selectbox("グループ別", spc_group_opts, key='spc_group')
        spc_group = None if spc_grp_sel == 'なし' else spc_grp_sel

    fig_spc = render_spc_chart(df, date_col, spc_metric, spc_group)
    if fig_spc:
        st.plotly_chart(fig_spc, use_container_width=True)
        st.caption("🔴 赤×印: 中心値 ± 3σ を超えた管理外点。工程異常の可能性があります。CL=中心線, UCL/LCL=上下管理限界。グループ指定時は各グループで独立した制御限界を適用しています。")
    else:
        st.error(
            f"「{spc_metric}」の SPC 管理図を生成できませんでした。"
            "数値列を選択してください。"
        )


def _tab_category(df: pd.DataFrame, col_info: dict) -> None:
    if not col_info['category']:
        st.info("カテゴリ列が検出されませんでした。")
        return
    if not col_info['numeric']:
        st.info("数値列が検出されませんでした。")
        return

    # ── 工程・装置別集計グラフ ────────────────────────────────
    st.markdown(
        '<p class="section-title">工程・装置別集計グラフ</p>', unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cat_col = st.selectbox("カテゴリ列", col_info['category'], key='cat_col')
    with c2:
        num_col = st.selectbox("数値列", col_info['numeric'], key='cat_num')
    with c3:
        chart_type = st.selectbox(
            "グラフ種別", ['棒グラフ', '積み上げ棒グラフ', '円グラフ'], key='cat_type'
        )
    with c4:
        agg_method = st.selectbox(
            "集計方法", ['平均', '合計', '最大', '最小'], key='cat_agg'
        )

    top_n = st.slider("上位 N 件", min_value=3, max_value=30, value=10, key='cat_topn')

    fig = render_category_chart(df, cat_col, num_col, chart_type, agg_method, top_n)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(
            f"「{num_col}」は数値でないため集計できません。別の列を選択してください。"
        )

    # ── パレート図 ────────────────────────────────────────────
    st.markdown('<p class="section-title">パレート図</p>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pareto_cat = st.selectbox(
            "分類列（欠陥種別・装置など）", col_info['category'], key='pareto_cat'
        )
    with pc2:
        pareto_num = st.selectbox(
            "集計列", col_info['numeric'], key='pareto_num'
        )
    with pc3:
        pareto_agg = st.selectbox(
            "集計方法", ['合計', '平均', '最大', '最小'], key='pareto_agg'
        )

    pareto_n = st.slider("上位 N 件", min_value=3, max_value=20, value=10, key='pareto_topn')

    fig_pareto = render_pareto_chart(df, pareto_cat, pareto_num, pareto_agg, pareto_n)
    if fig_pareto:
        st.plotly_chart(fig_pareto, use_container_width=True)
    else:
        st.error("パレート図を生成できませんでした。カテゴリ列と数値列を確認してください。")


def _tab_correlation(df: pd.DataFrame, col_info: dict) -> None:
    if len(col_info['numeric']) < 2:
        st.info("相関分析には数値列が 2 列以上必要です。")
        return

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("X 軸", col_info['numeric'], key='sc_x')
    with c2:
        y_opts = [c for c in col_info['numeric'] if c != x_col]
        y_col = st.selectbox("Y 軸", y_opts, key='sc_y')

    c3, c4 = st.columns(2)
    with c3:
        color_opts = ['なし'] + col_info['category']
        color_sel = st.selectbox("色分け列", color_opts, key='sc_color')
        color_col = None if color_sel == 'なし' else color_sel
    with c4:
        size_opts = ['なし'] + [c for c in col_info['numeric'] if c not in [x_col, y_col]]
        size_sel = st.selectbox("バブルサイズ列", size_opts, key='sc_size')
        size_col = None if size_sel == 'なし' else size_sel

    st.markdown('<p class="section-title">散布図</p>', unsafe_allow_html=True)
    fig = render_scatter_chart(df, x_col, y_col, color_col, size_col)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<p class="section-title">プロセスパラメータ 相関ヒートマップ</p>',
        unsafe_allow_html=True,
    )
    fig_hm = render_correlation_heatmap(df, col_info['numeric'])
    if fig_hm:
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("相関ヒートマップを生成できませんでした。")


def _tab_data(df: pd.DataFrame) -> None:
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_cols = st.multiselect(
            "表示する列",
            df.columns.tolist(),
            default=df.columns.tolist(),
            key='tbl_cols',
        )
    with c2:
        safe_len = max(1, len(df))
        row_limit = st.number_input(
            "表示行数",
            min_value=1,
            max_value=safe_len,
            value=min(100, safe_len),
            step=10,
            key='tbl_rows',
        )

    st.markdown('<p class="section-title">データテーブル</p>', unsafe_allow_html=True)

    if selected_cols:
        st.dataframe(
            df[selected_cols].head(int(row_limit)),
            use_container_width=True,
            height=400,
        )
    else:
        st.warning("表示する列を選択してください。")

    st.markdown("---")
    st.markdown("### エクスポート")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ CSV ダウンロード",
            data=_to_csv_bytes(df),
            file_name="filtered_data.csv",
            mime="text/csv",
        )
    with ec2:
        st.download_button(
            "⬇️ Excel ダウンロード",
            data=_to_excel_bytes(df),
            file_name="filtered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    df_raw, col_info, is_demo = _render_sidebar()

    date_col = col_info['date'][0] if col_info['date'] else None
    date_range = render_date_filter(df_raw, date_col) if date_col else None
    cat_selections = render_category_filters(df_raw, col_info['category'])

    df = apply_filters(df_raw, cat_selections, date_range, date_col)
    if len(df) < len(df_raw):
        st.sidebar.caption(f"絞り込み後: {len(df):,} 行 / {len(df_raw):,} 行")

    if is_demo:
        st.markdown(
            '<div class="demo-banner">'
            '🔵 半導体工場サンプルデータを表示中です。'
            'サイドバーから実データ CSV をアップロードすると切り替わります。'
            '</div>',
            unsafe_allow_html=True,
        )

    if len(df) > 100_000:
        st.warning(
            f"データが大きいため（{len(df):,} 行）、10 万行にサンプリングして表示します。"
        )
        df = df.sample(100_000, random_state=42).reset_index(drop=True)

    tabs = st.tabs([
        "📋 ダッシュボード",
        "📈 トレンド・SPC",
        "🏭 工程・装置分析",
        "🔗 相関・多変量",
        "🗂️ データ",
    ])

    with tabs[0]:
        _tab_overview(df, col_info)
    with tabs[1]:
        _tab_timeseries(df, col_info)
    with tabs[2]:
        _tab_category(df, col_info)
    with tabs[3]:
        _tab_correlation(df, col_info)
    with tabs[4]:
        _tab_data(df)


if __name__ == '__main__':
    main()
