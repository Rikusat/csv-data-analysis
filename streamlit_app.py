"""
半導体工場 生産データ分析ダッシュボード
Entry point — run with: streamlit run streamlit_app.py
"""

import csv
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    render_category_chart,
    render_correlation_heatmap,
    render_heatmap_chart,
    render_pareto_chart,
    render_period_bar_chart,
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


_FREQ_OPTIONS = {'日次': 'D', '週次': 'W', '月次': 'ME', '年次': 'YE'}


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
    cols = [str(c).strip().lstrip('\ufeff') for c in df.columns]
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


@st.cache_data
def _detect_columns(df: pd.DataFrame) -> dict:
    return detect_columns(df)


# ── Export helpers ────────────────────────────────────────────
@st.cache_data
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')


@st.cache_data
def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='データ')
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────
def _render_sidebar():
    """Render sidebar and return (df_raw, col_info, is_demo, freq)."""
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

    col_info = _detect_columns(df_raw)

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
    st.sidebar.markdown("### 集計粒度")
    freq_sel = st.sidebar.selectbox(
        "時間粒度",
        list(_FREQ_OPTIONS.keys()),
        key='global_freq',
        help="トレンドグラフ・期間集計バーチャート・KPI前期比に適用されます",
    )
    freq = _FREQ_OPTIONS[freq_sel]

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

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ 自動更新")
    auto_refresh = st.sidebar.checkbox("自動更新を有効にする", key='auto_refresh')
    refresh_interval = 30
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "更新間隔（秒）", min_value=5, max_value=300, value=30, step=5,
            key='refresh_interval',
        )
        st.sidebar.caption(f"⏱️ {refresh_interval} 秒ごとにデータを再読み込みします。")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕐 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df_raw, col_info, is_demo, freq, auto_refresh, refresh_interval


# ── Overlay helpers ───────────────────────────────────────────

def _parse_float(s: str):
    """Return float from string, or None for empty / non-numeric input."""
    try:
        return float(s.strip()) if s and s.strip() else None
    except ValueError:
        return None


def _apply_overlay_lines(fig, target, usl, lsl) -> None:
    """Add target / spec-limit horizontal lines to a Plotly figure."""
    if target is not None:
        fig.add_hline(
            y=target, line_color='#10b981', line_dash='solid', line_width=1.5,
            annotation_text=f'目標: {target:g}',
            annotation_position='top left',
            annotation_font=dict(color='#10b981', size=11),
        )
    if usl is not None:
        fig.add_hline(
            y=usl, line_color='#ef4444', line_dash='dot', line_width=1.5,
            annotation_text=f'USL: {usl:g}',
            annotation_position='top right',
            annotation_font=dict(color='#ef4444', size=11),
        )
    if lsl is not None:
        fig.add_hline(
            y=lsl, line_color='#ef4444', line_dash='dot', line_width=1.5,
            annotation_text=f'LSL: {lsl:g}',
            annotation_position='bottom right',
            annotation_font=dict(color='#ef4444', size=11),
        )


# ── Period comparison helpers ────────────────────────────────

_COMP_OFFSET = {
    'D':  lambda n: pd.Timedelta(days=n),
    'W':  lambda n: pd.Timedelta(weeks=n),
    'ME': lambda n: pd.DateOffset(months=n),
    'YE': lambda n: pd.DateOffset(years=n),
}
_COMP_UNIT    = {'D': '日', 'W': '週', 'ME': 'ヶ月', 'YE': '年'}
_COMP_DEFAULT = {'D': 30,   'W': 8,    'ME': 6,      'YE': 2}


def _build_comparison_df(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    n_periods: int,
) -> pd.DataFrame:
    """
    Return a DataFrame with '_period' label ('今期'/'前期') for overlay comparison.
    Previous-period dates are shifted forward by one period so both align on the same X axis.
    """
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        max_date = dates.max()
        if pd.isna(max_date):
            return pd.DataFrame()

        offset = _COMP_OFFSET.get(freq, lambda n: pd.Timedelta(days=n))(n_periods)

        current_start = max_date - offset
        prev_start    = current_start - offset

        curr_df = df[dates > current_start].copy()
        curr_df['_period'] = '今期'

        prev_df = df[(dates > prev_start) & (dates <= current_start)].copy()
        prev_df['_period'] = '前期'
        prev_df[date_col] = pd.to_datetime(prev_df[date_col], errors='coerce') + offset

        return pd.concat([curr_df, prev_df], ignore_index=True)
    except Exception:
        return pd.DataFrame()


# ── SPC outlier helper ───────────────────────────────────────

def _calc_spc_outliers(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col,
):
    """
    Return (outliers_df, total_points) using the same ±3σ logic as render_spc_chart.
    outliers_df is empty when no violations exist.
    """
    try:
        df_p = df.copy()
        df_p[date_col] = pd.to_datetime(df_p[date_col], errors='coerce')
        df_p[value_col] = pd.to_numeric(df_p[value_col], errors='coerce')
        df_p = df_p.dropna(subset=[date_col, value_col])

        if group_col and group_col in df_p.columns:
            agg = df_p.groupby([date_col, group_col], as_index=False)[value_col].mean()
        else:
            agg = df_p.groupby(date_col, as_index=False)[value_col].mean()

        if agg.empty:
            return pd.DataFrame(), 0

        total = len(agg)
        rows = []

        def _process_group(sub, grp_label=None):
            mu = float(sub[value_col].mean())
            sigma = float(sub[value_col].std())
            if not np.isfinite(sigma) or sigma == 0:
                sigma = abs(mu) * 0.001 if mu != 0 else 0.001
            ucl, lcl = mu + 3 * sigma, mu - 3 * sigma
            for _, row in sub.iterrows():
                v = float(row[value_col])
                if v > ucl or v < lcl:
                    r = {
                        '日付': row[date_col].date(),
                        '実測値': round(v, 4),
                        '偏差 (実測 − CL)': round(v - mu, 4),
                        'CL': round(mu, 4),
                        'UCL': round(ucl, 4),
                        'LCL': round(lcl, 4),
                        '判定': 'UCL 超過' if v > ucl else 'LCL 超過',
                    }
                    if grp_label is not None:
                        r[group_col] = grp_label
                    rows.append(r)

        if group_col and group_col in agg.columns:
            for grp in sorted(agg[group_col].dropna().unique(), key=str):
                _process_group(agg[agg[group_col] == grp], grp)
        else:
            _process_group(agg)

        if not rows:
            return pd.DataFrame(), total

        out = pd.DataFrame(rows).sort_values('日付').reset_index(drop=True)
        # Reorder: 日付 → group (if any) → 実測値 → 偏差 → CL/UCL/LCL → 判定
        base_cols = ['日付']
        if group_col and group_col in out.columns:
            base_cols.append(group_col)
        base_cols += ['実測値', '偏差 (実測 − CL)', 'CL', 'UCL', 'LCL', '判定']
        out = out[[c for c in base_cols if c in out.columns]]
        return out, total
    except Exception:
        return pd.DataFrame(), 0


# ── Process capability helpers ───────────────────────────────

def _calc_process_capability(series: pd.Series, usl, lsl):
    """Compute Cp, Cpk, σ level, μ, σ, n. Returns dict or None on failure."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < 2:
        return None
    mu = float(s.mean())
    sigma = float(s.std(ddof=1))
    if sigma == 0:
        return None

    cp  = (usl - lsl) / (6 * sigma) if usl is not None and lsl is not None else None
    cpu = (usl - mu)  / (3 * sigma) if usl is not None else None
    cpl = (mu  - lsl) / (3 * sigma) if lsl is not None else None

    if cpu is not None and cpl is not None:
        cpk = min(cpu, cpl)
    elif cpu is not None:
        cpk = cpu
    elif cpl is not None:
        cpk = cpl
    else:
        cpk = None

    return {
        'mu': mu, 'sigma': sigma, 'n': int(len(s)),
        'cp': cp, 'cpk': cpk,
        'sigma_level': cpk * 3 if cpk is not None else None,
    }


_CPK_THRESHOLDS = [
    (1.67, '#10b981', '超優良'),
    (1.33, '#2563EB', '優良'),
    (1.00, '#f59e0b', '合格'),
]


def _cpk_color_label(cpk):
    if cpk is None:
        return '#6b7280', '—'
    for thr, color, label in _CPK_THRESHOLDS:
        if cpk >= thr:
            return color, label
    return '#ef4444', '要改善'


def _cap_card(label: str, value: str, sub: str, border_color: str) -> str:
    css = (
        f"padding:16px;background:#fff;border-radius:12px;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid {border_color};"
        f"margin-bottom:8px;"
    )
    return (
        f'<div style="{css}">'
        f'<p style="color:#6b7280;font-size:12px;margin:0;font-weight:500;">{label}</p>'
        f'<p style="color:#111827;font-size:22px;font-weight:700;margin:6px 0 0 0;">{value}</p>'
        f'<p style="color:#9ca3af;font-size:11px;margin:2px 0 0 0;">{sub}</p>'
        f'</div>'
    )


# ── Achievement card helpers ─────────────────────────────────

def _fmt_value(value: float) -> str:
    try:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value:,.1f}"
        if abs(value - round(value)) < 1e-9:
            return f"{round(value):,}"
        return f"{value:.2f}"
    except Exception:
        return '—'


def _render_achievement_card(label: str, actual: float, target: float, agg_method: str) -> str:
    if target == 0:
        return ''
    rate = actual / target * 100
    if rate >= 100:
        color, badge = '#10b981', '✅ 達成'
    elif rate >= 80:
        color, badge = '#f59e0b', '⚠️ 接近中'
    else:
        color, badge = '#ef4444', '❌ 未達成'
    css = (
        f"padding:16px;background:#fff;border-radius:12px;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid {color};"
        f"margin-bottom:8px;"
    )
    return (
        f'<div style="{css}">'
        f'<p style="color:#6b7280;font-size:12px;margin:0;font-weight:500;">🎯 {label}</p>'
        f'<p style="color:#111827;font-size:22px;font-weight:700;margin:6px 0 0 0;">{rate:.1f}%</p>'
        f'<p style="color:{color};font-size:12px;margin:2px 0 0 0;">{badge}</p>'
        f'<p style="color:#9ca3af;font-size:11px;margin:4px 0 0 0;">'
        f'実績: {_fmt_value(actual)} / 目標: {_fmt_value(target)}&nbsp;({agg_method})</p>'
        f'</div>'
    )


# ── Tab renderers ─────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame, col_info: dict, freq: str = 'D') -> None:
    st.markdown('<p class="section-title">主要 KPI</p>', unsafe_allow_html=True)

    agg_method = st.selectbox(
        "集計方法", ['平均', '合計', '最大', '最小'], key='kpi_agg'
    )
    date_col = col_info['date'][0] if col_info['date'] else None
    render_kpi_cards(df, col_info['numeric'], date_col=date_col, agg_method=agg_method, freq=freq)

    # ── 目標達成率 ────────────────────────────────────────────
    st.markdown('<p class="section-title">目標達成率</p>', unsafe_allow_html=True)

    with st.expander("🎯 目標値の設定（指標ごとに入力）", expanded=False):
        tgt_metrics = col_info['numeric'][:8]
        tgt_vals: dict = {}
        for row_start in range(0, len(tgt_metrics), 4):
            row_items = tgt_metrics[row_start:row_start + 4]
            tcols = st.columns(len(row_items))
            for j, m in enumerate(row_items):
                with tcols[j]:
                    s = st.text_input(m, key=f'tgt_{m}', placeholder="例: 95.0")
                    tgt_vals[m] = _parse_float(s)

    active_targets = [(m, t) for m, t in tgt_vals.items() if t is not None]

    if not active_targets:
        st.info("🎯 上の「目標値の設定」で目標値を入力すると達成率カードが表示されます。")
    else:
        _agg_map = {'平均': 'mean', '合計': 'sum', '最大': 'max', '最小': 'min'}
        agg_fn = _agg_map.get(agg_method, 'mean')

        for row_start in range(0, len(active_targets), 4):
            row_items = active_targets[row_start:row_start + 4]
            row_cols = st.columns(len(row_items))
            for i, (metric, target) in enumerate(row_items):
                with row_cols[i]:
                    try:
                        series = pd.to_numeric(df[metric], errors='coerce').dropna()
                        actual = float(getattr(series, agg_fn)())
                        html = _render_achievement_card(metric, actual, target, agg_method)
                        if html:
                            st.markdown(html, unsafe_allow_html=True)
                    except Exception:
                        st.error(f"「{metric}」の達成率を計算できませんでした。")

        st.caption(
            "💡 低い方が良い指標（欠陥密度・不良率など）は目標値を小さく設定し、"
            "達成率の解釈を逆にしてください（100% 超 = 目標値以下に抑制できた状態）。"
        )

    if col_info['date'] and col_info['numeric']:
        st.markdown(
            '<p class="section-title">歩留まり・欠陥トレンド（クイックビュー）</p>',
            unsafe_allow_html=True,
        )
        quick_metrics = col_info['numeric'][:2]
        fig = render_timeseries_chart(
            df, col_info['date'][0], col_info['numeric'],
            selected_metrics=quick_metrics, freq=freq,
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


def _tab_timeseries(df: pd.DataFrame, col_info: dict, freq: str = 'D') -> None:
    if not col_info['date']:
        st.info("日付列が検出されませんでした。CSVに日付列を含めてください。")
        return
    if not col_info['numeric']:
        st.info("数値列が検出されませんでした。")
        return

    date_col = col_info['date'][0]

    # ── 目標線・規格線の設定 ──────────────────────────────────
    with st.expander("📏 目標線・規格線の設定（トレンドグラフ・SPC に適用）", expanded=False):
        ol1, ol2, ol3 = st.columns(3)
        with ol1:
            target_str = st.text_input("目標値", key='ol_target', placeholder="例: 95.0（空欄で非表示）")
        with ol2:
            usl_str = st.text_input("規格上限 (USL)", key='ol_usl', placeholder="例: 100.0")
        with ol3:
            lsl_str = st.text_input("規格下限 (LSL)", key='ol_lsl', placeholder="例: 90.0")
    target_val = _parse_float(target_str)
    usl_val    = _parse_float(usl_str)
    lsl_val    = _parse_float(lsl_str)

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

    with st.expander("〜 移動平均の設定", expanded=False):
        ma_c1, ma_c2 = st.columns(2)
        with ma_c1:
            ma_enabled = st.checkbox("移動平均を表示", key='ma_enabled')
        with ma_c2:
            ma_window = st.slider(
                "ウィンドウサイズ（データ点数）",
                min_value=2, max_value=60, value=7, step=1,
                key='ma_window', disabled=not ma_enabled,
            )
    ma_windows_arg = [ma_window] if ma_enabled else None

    if not selected_metrics:
        st.warning("指標を1つ以上選択してください。")
    else:
        fig = render_timeseries_chart(df, date_col, col_info['numeric'], group_col, selected_metrics, ts_agg_en, freq=freq, ma_windows=ma_windows_arg)
        if fig:
            if any(v is not None for v in (target_val, usl_val, lsl_val)):
                if len(selected_metrics) > 1:
                    st.caption("📏 目標線・規格線は指標を1つ選択しているときに正確に適用されます。複数指標では Y 軸スケールが混在します。")
                _apply_overlay_lines(fig, target_val, usl_val, lsl_val)
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
        _apply_overlay_lines(fig_spc, target_val, usl_val, lsl_val)
        st.plotly_chart(fig_spc, use_container_width=True)
        st.caption("🔴 赤×印: 中心値 ± 3σ を超えた管理外点。工程異常の可能性があります。CL=中心線, UCL/LCL=上下管理限界。グループ指定時は各グループで独立した制御限界を適用しています。")
    else:
        st.error(
            f"「{spc_metric}」の SPC 管理図を生成できませんでした。"
            "数値列を選択してください。"
        )

    # ── SPC 異常点サマリー ────────────────────────────────────
    st.markdown('<p class="section-title">SPC 異常点サマリー</p>', unsafe_allow_html=True)

    outliers_df, total_pts = _calc_spc_outliers(df, date_col, spc_metric, spc_group)

    if outliers_df.empty:
        st.success(f"✅ 「{spc_metric}」に管理外点はありません。工程は ±3σ 以内で安定しています。")
    else:
        pct = len(outliers_df) / total_pts * 100 if total_pts > 0 else 0
        st.warning(
            f"⚠️ 管理外点: **{len(outliers_df)}** 件 / 全 {total_pts} 点（{pct:.1f}%）"
        )
        st.dataframe(
            outliers_df,
            use_container_width=True,
            height=min(400, (len(outliers_df) + 1) * 35 + 40),
        )
        st.download_button(
            "⬇️ 異常点 CSV ダウンロード",
            data=outliers_df.to_csv(index=False).encode('utf-8-sig'),
            file_name=f"spc_outliers_{spc_metric}.csv",
            mime="text/csv",
            key='dl_outliers',
        )

    # ── 工程能力指数 ──────────────────────────────────────────
    st.markdown('<p class="section-title">工程能力指数（Cp / Cpk）</p>', unsafe_allow_html=True)
    st.caption(f"対象指標: **{spc_metric}** ／ 規格上下限は上の「目標線・規格線の設定」と共有")

    if usl_val is None and lsl_val is None:
        st.info("📏 上の「目標線・規格線の設定」で USL または LSL を入力すると Cp・Cpk を算出します。")
    else:
        cap = _calc_process_capability(df[spc_metric], usl_val, lsl_val)
        if cap is None:
            st.warning(f"「{spc_metric}」の工程能力を計算できません。有効なデータが 2 件以上必要です。")
        else:
            cpk_color, cpk_judgment = _cpk_color_label(cap['cpk'])

            cards = []
            if cap['cp'] is not None:
                cards.append(('Cp', f"{cap['cp']:.3f}", '工程能力（両側）', '#2563EB'))
            if cap['cpk'] is not None:
                cards.append(('Cpk', f"{cap['cpk']:.3f}", cpk_judgment, cpk_color))
            if cap['sigma_level'] is not None:
                cards.append(('σ 水準', f"{cap['sigma_level']:.2f} σ", '中心からの余裕', '#2563EB'))
            cards.append(('μ（平均）', f"{cap['mu']:.4g}", f"n = {cap['n']:,}", '#6b7280'))
            cards.append(('σ（標準偏差）', f"{cap['sigma']:.4g}", '母集団推定 (ddof=1)', '#6b7280'))

            cols = st.columns(len(cards))
            for i, (label, value, sub, color) in enumerate(cards):
                with cols[i]:
                    st.markdown(_cap_card(label, value, sub, color), unsafe_allow_html=True)

            with st.expander("📘 Cpk 判定基準", expanded=False):
                st.markdown(
                    "| Cpk | 判定 | 意味 |\n"
                    "|-----|------|------|\n"
                    "| ≥ 1.67 | 🟢 超優良 | 6σ 以上の余裕。工程は非常に安定 |\n"
                    "| ≥ 1.33 | 🔵 優良 | 一般的な量産合格基準 |\n"
                    "| ≥ 1.00 | 🟡 合格 | 最低限の基準。改善余地あり |\n"
                    "| < 1.00 | 🔴 要改善 | 規格外品の発生リスクあり |"
                )
                if cap['cp'] is not None and cap['cpk'] is not None:
                    skew = cap['cp'] - cap['cpk']
                    if skew > 0.1:
                        st.caption(f"⚠️ Cp ({cap['cp']:.3f}) と Cpk ({cap['cpk']:.3f}) の差 ({skew:.3f}) が大きく、工程平均が規格中心から偏っています。")

    # ── 期間集計バーチャート ──────────────────────────────────
    st.markdown('<p class="section-title">期間集計バーチャート</p>', unsafe_allow_html=True)

    pb1, pb2, pb3 = st.columns(3)
    with pb1:
        pb_metric = st.selectbox("集計指標", col_info['numeric'], key='pb_metric')
    with pb2:
        pb_group_opts = ['なし'] + col_info['category']
        pb_grp_sel = st.selectbox("グループ列", pb_group_opts, key='pb_group')
        pb_group = None if pb_grp_sel == 'なし' else pb_grp_sel
    with pb3:
        pb_agg = st.selectbox("集計方法", ['合計', '平均', '最大', '最小'], key='pb_agg')

    fig_pb = render_period_bar_chart(df, date_col, pb_metric, freq, pb_agg, pb_group)
    if fig_pb:
        st.plotly_chart(fig_pb, use_container_width=True)
    else:
        st.error(
            f"「{pb_metric}」の期間集計バーチャートを生成できませんでした。"
            "日付列と数値列を確認してください。"
        )

    # ── 前期比較モード ────────────────────────────────────────
    st.markdown('<p class="section-title">前期比較モード</p>', unsafe_allow_html=True)

    cmp1, cmp2, cmp3 = st.columns(3)
    with cmp1:
        cmp_metric = st.selectbox("比較指標", col_info['numeric'], key='cmp_metric')
    with cmp2:
        cmp_n = st.number_input(
            f"比較期間（{_COMP_UNIT.get(freq, '期')}単位）",
            min_value=1, max_value=365,
            value=_COMP_DEFAULT.get(freq, 30),
            step=1, key='cmp_n',
            help="今期・前期それぞれの期間長。サイドバーの集計粒度と同じ単位です。",
        )
    with cmp3:
        cmp_agg = st.selectbox("集計方法", ['平均', '合計', '最大', '最小'], key='cmp_agg')
        cmp_agg_en = {'平均': 'mean', '合計': 'sum', '最大': 'max', '最小': 'min'}[cmp_agg]

    cmp_df = _build_comparison_df(df, date_col, freq, int(cmp_n))
    if cmp_df.empty:
        st.info("前期比較を表示できませんでした。日付列と十分なデータ量を確認してください。")
    else:
        fig_cmp = render_timeseries_chart(
            cmp_df, date_col, col_info['numeric'],
            group_col='_period',
            selected_metrics=[cmp_metric],
            agg_method=cmp_agg_en,
            freq=freq,
        )
        if fig_cmp:
            unit = _COMP_UNIT.get(freq, '期')
            st.plotly_chart(fig_cmp, use_container_width=True)
            st.caption(
                f"今期（直近 {int(cmp_n)} {unit}）と前期（その前の {int(cmp_n)} {unit}）を同一軸で比較。"
                "前期の日付は今期の日付軸に揃えてシフトしています。"
            )
        else:
            st.error("前期比較グラフを生成できませんでした。データを確認してください。")


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

    # ── シフト別集計 ──────────────────────────────────────────
    st.markdown('<p class="section-title">シフト別集計</p>', unsafe_allow_html=True)

    if not col_info['category'] or not col_info['numeric']:
        st.info("シフト別集計にはカテゴリ列と数値列の両方が必要です。")
    else:
        # Auto-detect shift column by keyword
        auto_shift = next(
            (c for c in col_info['category'] if any(kw in c.lower() for kw in ('shift', 'シフト'))),
            None,
        )
        shift_default_idx = col_info['category'].index(auto_shift) if auto_shift else 0

        if auto_shift:
            st.caption(f"🔍 シフト列を自動検出: `{auto_shift}`　（変更する場合は下のセレクトボックスで選択）")
        else:
            st.caption("シフト列が自動検出されませんでした。手動でシフト相当の列を選択してください。")

        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            shift_col = st.selectbox(
                "シフト列", col_info['category'],
                index=shift_default_idx, key='shift_col',
            )
        with sh2:
            shift_metric = st.selectbox("集計指標", col_info['numeric'], key='shift_metric')
        with sh3:
            shift_agg = st.selectbox(
                "集計方法", ['平均', '合計', '最大', '最小'], key='shift_agg'
            )

        fig_shift = render_category_chart(df, shift_col, shift_metric, '棒グラフ', shift_agg, top_n=20)
        if fig_shift:
            st.plotly_chart(fig_shift, use_container_width=True)
        else:
            st.error(f"「{shift_metric}」のシフト別グラフを生成できませんでした。数値列を確認してください。")

        # Summary table: shift × all numeric cols (cap at 8)
        st.markdown("**シフト別サマリー（全指標）**")
        try:
            agg_func_map = {'平均': 'mean', '合計': 'sum', '最大': 'max', '最小': 'min'}
            agg_func = agg_func_map[shift_agg]
            num_cols_tbl = col_info['numeric'][:8]
            tmp = df[[shift_col] + num_cols_tbl].copy()
            for c in num_cols_tbl:
                tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
            summary_df = (
                tmp.groupby(shift_col)[num_cols_tbl]
                .agg(agg_func)
                .round(3)
                .reset_index()
            )
            st.dataframe(summary_df, use_container_width=True)
            if len(col_info['numeric']) > 8:
                st.caption(
                    f"※ 表示は先頭 8 列のみ。残り {len(col_info['numeric']) - 8} 列は省略しています。"
                )
        except Exception:
            st.error("シフト別サマリーの生成に失敗しました。")

    # ── 稼働率ヒートマップ ────────────────────────────────────
    st.markdown('<p class="section-title">稼働率ヒートマップ</p>', unsafe_allow_html=True)

    if not col_info['date']:
        st.info("ヒートマップには日付列が必要です。CSVに日付列を含めてください。")
    else:
        hm1, hm2, hm3 = st.columns(3)
        with hm1:
            hm_cat = st.selectbox("カテゴリ列（Y 軸）", col_info['category'], key='hm_cat')
        with hm2:
            hm_val = st.selectbox("数値列（色）", col_info['numeric'], key='hm_val')
        with hm3:
            hm_agg = st.selectbox("集計方法", ['平均', '合計', '最大', '最小'], key='hm_agg')

        hm4, hm5 = st.columns(2)
        with hm4:
            _hm_freq_opts = {'日次': 'D', '週次': 'W', '月次': 'ME', '年次': 'YE'}
            hm_freq_sel = st.selectbox(
                "集計粒度", list(_hm_freq_opts.keys()), index=2, key='hm_freq'
            )
            hm_freq = _hm_freq_opts[hm_freq_sel]
        with hm5:
            hm_reverse = st.checkbox(
                "スケールを反転（低い値が良い指標）",
                key='hm_reverse',
                help="欠陥密度・不良率など低い方が望ましい指標はチェックしてください。",
            )

        fig_hm = render_heatmap_chart(
            df, col_info['date'][0], hm_cat, hm_val, hm_agg, hm_freq, hm_reverse,
        )
        if fig_hm:
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("色が緑に近いほど良好、赤に近いほど注意が必要です（スケール反転時は逆）。直近 60 期間を表示。")
        else:
            st.error("ヒートマップを生成できませんでした。日付列・カテゴリ列・数値列を確認してください。")

    # ── ランキングテーブル ────────────────────────────────────
    st.markdown('<p class="section-title">ランキングテーブル</p>', unsafe_allow_html=True)

    rk1, rk2, rk3, rk4 = st.columns(4)
    with rk1:
        rk_cat = st.selectbox("カテゴリ列", col_info['category'], key='rk_cat')
    with rk2:
        rk_val = st.selectbox("集計列", col_info['numeric'], key='rk_val')
    with rk3:
        rk_agg = st.selectbox("集計方法", ['合計', '平均', '最大', '最小'], key='rk_agg')
    with rk4:
        rk_order = st.selectbox("並び順", ['降順（高い順）', '昇順（低い順）'], key='rk_order')

    rk_n = st.slider("表示件数", min_value=3, max_value=50, value=10, key='rk_n')

    try:
        _rk_agg_map = {'合計': 'sum', '平均': 'mean', '最大': 'max', '最小': 'min'}
        agg_fn = _rk_agg_map[rk_agg]
        ascending = rk_order == '昇順（低い順）'

        tmp = df[[rk_cat, rk_val]].copy()
        tmp[rk_val] = pd.to_numeric(tmp[rk_val], errors='coerce')

        all_agg = tmp.groupby(rk_cat)[rk_val].agg(agg_fn).dropna()
        if all_agg.empty:
            st.info("集計結果が空です。列を確認してください。")
        else:
            total   = float(all_agg.sum())
            mean_v  = float(all_agg.mean())

            ranked = (
                all_agg
                .sort_values(ascending=ascending)
                .head(rk_n)
                .reset_index()
            )
            ranked.columns = [rk_cat, rk_agg]
            ranked.insert(0, '順位', range(1, len(ranked) + 1))
            ranked[f'全体比 (%)'] = (ranked[rk_agg] / total * 100).round(1) if total != 0 else float('nan')
            ranked['平均比']       = (ranked[rk_agg] / mean_v).round(3)       if mean_v != 0 else float('nan')
            ranked['差分 (vs 平均)'] = (ranked[rk_agg] - mean_v).round(3)
            ranked[rk_agg]         = ranked[rk_agg].round(3)

            st.dataframe(
                ranked, use_container_width=True,
                height=min(500, (len(ranked) + 1) * 35 + 40),
            )
            st.download_button(
                "⬇️ ランキング CSV ダウンロード",
                data=ranked.to_csv(index=False).encode('utf-8-sig'),
                file_name=f"ranking_{rk_cat}_{rk_val}.csv",
                mime="text/csv",
                key='dl_ranking',
            )
    except Exception:
        st.error("ランキングテーブルを生成できませんでした。カテゴリ列と数値列を確認してください。")


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
    df_raw, col_info, is_demo, freq, auto_refresh, refresh_interval = _render_sidebar()

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
        _tab_overview(df, col_info, freq)
    with tabs[1]:
        _tab_timeseries(df, col_info, freq)
    with tabs[2]:
        _tab_category(df, col_info)
    with tabs[3]:
        _tab_correlation(df, col_info)
    with tabs[4]:
        _tab_data(df)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()
