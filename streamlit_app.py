"""
半導体工場 生産データ分析ダッシュボード
Entry point — run with: streamlit run streamlit_app.py
"""

import csv
import html as _html
import io
import re
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
    # Preserve leading-zero strings — they are codes/IDs (e.g. "001", "00123"), not numbers
    if non_null.astype(str).str.strip().str.match(r'^0\d').any():
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

    # Convert YYYYMMDD integer columns to datetime (e.g. 20240115 \u2192 2024-01-15)
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        try:
            smin, smax = int(s.min()), int(s.max())
            str_s = s.astype(str)
            if 19000101 <= smin and smax <= 21001231 and str_s.str.len().max() == 8:
                converted = pd.to_datetime(str_s, format='%Y%m%d', errors='coerce')
                if converted.notna().mean() >= 0.8:
                    as_str = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
                    df[col] = pd.to_datetime(as_str, format='%Y%m%d', errors='coerce')
        except Exception:
            pass

    # include both legacy object dtype and pandas 3.x StringDtype
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = _try_parse_numeric(df[col])
    return df


@st.cache_data
def _get_excel_sheets(file_bytes: bytes) -> list:
    """Return sheet names from an Excel file."""
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
        return xl.sheet_names
    except Exception:
        return []


@st.cache_data
def _load_file(file_bytes: bytes, filename: str, sheet_name=None) -> pd.DataFrame:
    """Load CSV or Excel. Auto-detects encoding and delimiter. Returns clean DataFrame."""
    if not file_bytes or not file_bytes.strip():
        raise ValueError("ファイルが空です。データが含まれているファイルを選択してください。")

    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'csv'

    if ext in ('xlsx', 'xls'):
        try:
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                engine='openpyxl',
                sheet_name=sheet_name if sheet_name is not None else 0,
            )
            df = _normalize_dataframe(df)
            if df.empty:
                sheet_label = f"（シート: {sheet_name}）" if sheet_name else ""
                raise ValueError(
                    f"データ行がありません{sheet_label}。"
                    "ヘッダー行のみのシートはサポートされていません。"
                )
            return df
        except ValueError:
            raise
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
            df = _normalize_dataframe(df)
            if df.empty:
                raise ValueError(
                    "データ行がありません。ヘッダー行のみのCSVはサポートされていません。"
                )
            return df
        except ValueError:
            raise
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

    uploaded_list = st.sidebar.file_uploader(
        "CSV / Excel をアップロード（複数可）",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="CSV（UTF-8 / Shift-JIS / CP932 / タブ・セミコロン区切り）または Excel（.xlsx）。複数ファイルを選択すると縦結合されます。上限: 1ファイルあたり 200 MB。",
    )

    is_demo = len(uploaded_list) == 0

    if is_demo:
        df_raw = _demo_data()
    else:
        _total_bytes = sum(f.size for f in uploaded_list)
        if _total_bytes > 50 * 1024 * 1024:
            st.sidebar.warning(
                f"合計ファイルサイズが大きいです（{_total_bytes / 1024 / 1024:.0f} MB）。"
                "読み込みに時間がかかる場合があります。"
            )

        # Pass 1: read bytes and show sheet selectors for Excel files
        file_queue = []
        for f in uploaded_list:
            raw_bytes = f.read()
            sheet_name = None
            if f.name.lower().rsplit('.', 1)[-1] in ('xlsx', 'xls'):
                sheets = _get_excel_sheets(raw_bytes)
                if len(sheets) > 1:
                    sheet_name = st.sidebar.selectbox(
                        f"📊 {f.name} — シート選択",
                        sheets,
                        key=f'sheet_{f.name}',
                    )
                elif sheets:
                    sheet_name = sheets[0]
            file_queue.append((raw_bytes, f.name, sheet_name))

        # Pass 2: load DataFrames
        with st.spinner("ファイルを読み込み中..."):
            loaded_dfs: list[pd.DataFrame] = []
            for raw_bytes, fname, sheet_name in file_queue:
                try:
                    df_f = _load_file(raw_bytes, fname, sheet_name=sheet_name).copy()
                    if len(file_queue) > 1:
                        df_f['_source'] = fname
                    loaded_dfs.append(df_f)
                    size_mb = len(raw_bytes) / 1024 / 1024
                    sheet_label = f" [{sheet_name}]" if sheet_name else ""
                    st.sidebar.caption(f"📄 {fname}{sheet_label}  ({size_mb:.1f} MB, {len(df_f):,} 行)")
                except ValueError as exc:
                    st.sidebar.error(f"{fname}: {exc}")

            if not loaded_dfs:
                df_raw = _demo_data()
                is_demo = True
            elif len(loaded_dfs) == 1:
                df_raw = loaded_dfs[0]
            else:
                col_sets = [set(d.columns) for d in loaded_dfs]
                common_n = len(col_sets[0].intersection(*col_sets[1:]))
                all_n = len(col_sets[0].union(*col_sets[1:]))
                if common_n < all_n:
                    st.sidebar.info(
                        f"共通列: {common_n} / 全 {all_n} 列。"
                        "共通でない列は NaN で補完されます。"
                    )
                df_raw = pd.concat(loaded_dfs, ignore_index=True)
                st.sidebar.success(
                    f"✅ {len(loaded_dfs)} ファイルを結合: {len(df_raw):,} 行"
                )

    st.sidebar.markdown(
        f"**行数:** {len(df_raw):,}　**列数:** {df_raw.shape[1]}"
    )

    if is_demo:
        st.sidebar.download_button(
            "⬇️ デモデータを CSV でダウンロード",
            data=df_raw.to_csv(index=False).encode("utf-8-sig"),
            file_name="demo_semiconductor.csv",
            mime="text/csv",
            help="このサンプルデータを CSV として保存できます。自分のデータを用意する際のフォーマット参考にどうぞ。",
        )

    if not is_demo:
        with st.sidebar.expander("📄 データプレビュー（先頭5行）", expanded=False):
            st.dataframe(df_raw.head(5), use_container_width=True)

    st.sidebar.markdown("---")

    col_info = _detect_columns(df_raw)

    # ── Column role overrides ─────────────────────────────────
    _ROLE_JP  = {'date': '日付', 'category': 'カテゴリ', 'numeric': '数値', 'text': 'テキスト'}
    _ROLE_EN  = {v: k for k, v in _ROLE_JP.items()}
    _ROLE_OPTS = list(_ROLE_JP.values())

    with st.sidebar.expander("🔧 列ロール上書き", expanded=False):
        st.caption("自動判定が誤っている列を修正できます。")
        ovr_cols = st.multiselect(
            "変更する列",
            df_raw.columns.tolist(),
            key='role_ovr_cols',
        )
        for col in ovr_cols:
            current_role = next(
                (r for r, cs in col_info.items() if col in cs), 'text'
            )
            new_label = st.selectbox(
                f"`{col}`",
                _ROLE_OPTS,
                index=_ROLE_OPTS.index(_ROLE_JP[current_role]),
                key=f'role_ovr_{col}',
            )
            new_role = _ROLE_EN[new_label]
            if new_role != current_role:
                if col in col_info[current_role]:
                    col_info[current_role].remove(col)
                if col not in col_info[new_role]:
                    col_info[new_role].append(col)

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

    # ── Condition filters ─────────────────────────────────────
    cond_rules: list = []
    _SKIP = '（スキップ）'
    _NUM_OPS  = ['=', '≠', '>', '>=', '<', '<=']
    _STR_OPS  = ['=', '≠', '含む', '含まない']

    with st.sidebar.expander("🔎 条件付き行フィルター", expanded=False):
        st.caption("列・演算子・値を指定して行を絞り込みます（AND 結合）。")
        n_rules = int(st.number_input(
            "ルール数", min_value=1, max_value=5, value=1, step=1, key='cond_n_rules'
        ))
        col_opts = [_SKIP] + df_raw.columns.tolist()
        for i in range(n_rules):
            rc1, rc2, rc3 = st.columns([3, 2, 3])
            with rc1:
                col_sel = st.selectbox('列', col_opts, key=f'cond_col_{i}', label_visibility='collapsed')
            if col_sel == _SKIP:
                continue
            is_num = col_sel in col_info.get('numeric', [])
            ops = _NUM_OPS if is_num else _STR_OPS
            with rc2:
                op_sel = st.selectbox('演算子', ops, key=f'cond_op_{i}', label_visibility='collapsed')
            with rc3:
                val_sel = st.text_input('値', key=f'cond_val_{i}', label_visibility='collapsed')
            if val_sel.strip():
                cond_rules.append((col_sel, op_sel, val_sel.strip()))

        if cond_rules:
            st.caption(f"✅ {len(cond_rules)} 件のルールが有効")

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

    return df_raw, col_info, is_demo, freq, auto_refresh, refresh_interval, cond_rules


# ── Condition filter helper ───────────────────────────────────

def _apply_condition_filters(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """Apply a list of (col, op, val_str) condition rules to df."""
    for col, op, val_str in rules:
        if col not in df.columns:
            continue
        try:
            series_num = pd.to_numeric(df[col], errors='coerce')
            val_num = pd.to_numeric(val_str, errors='coerce')
            if op in ('>', '>=', '<', '<=') or pd.notna(val_num):
                val = float(val_num) if pd.notna(val_num) else None
                if op == '=' and val is not None:
                    df = df[series_num == val]
                elif op == '≠' and val is not None:
                    df = df[series_num != val]
                elif op == '>' and val is not None:
                    df = df[series_num > val]
                elif op == '>=' and val is not None:
                    df = df[series_num >= val]
                elif op == '<' and val is not None:
                    df = df[series_num < val]
                elif op == '<=' and val is not None:
                    df = df[series_num <= val]
                elif op == '含む':
                    df = df[df[col].astype(str).str.contains(val_str, na=False, regex=False)]
                elif op == '含まない':
                    df = df[~df[col].astype(str).str.contains(val_str, na=False, regex=False)]
            else:
                col_str = df[col].astype(str)
                if op == '=':
                    df = df[col_str == val_str]
                elif op == '≠':
                    df = df[col_str != val_str]
                elif op == '含む':
                    df = df[col_str.str.contains(val_str, na=False, regex=False)]
                elif op == '含まない':
                    df = df[~col_str.str.contains(val_str, na=False, regex=False)]
        except Exception:
            pass
    return df


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
_COMP_UNIT    = {'D': '日',        'W': '週',     'ME': 'ヶ月',   'YE': '年'}
_COMP_DEFAULT = {'D': 30,          'W': 8,         'ME': 6,        'YE': 2}
_PERIOD_FMT   = {'D': '%Y/%m/%d',  'W': '%Y-W%W',  'ME': '%Y/%m',  'YE': '%Y'}


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


def _safe_filename(s: str) -> str:
    """Strip characters that are invalid in filenames on Windows/Linux."""
    return re.sub(r'[\\/:*?"<>|\s]+', '_', s).strip('_') or 'col'


def _cap_card(label: str, value: str, sub: str, border_color: str) -> str:
    css = (
        f"padding:16px;background:#fff;border-radius:12px;"
        f"box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid {border_color};"
        f"margin-bottom:8px;"
    )
    return (
        f'<div style="{css}">'
        f'<p style="color:#6b7280;font-size:12px;margin:0;font-weight:500;">{_html.escape(label)}</p>'
        f'<p style="color:#111827;font-size:22px;font-weight:700;margin:6px 0 0 0;">{_html.escape(value)}</p>'
        f'<p style="color:#9ca3af;font-size:11px;margin:2px 0 0 0;">{_html.escape(sub)}</p>'
        f'</div>'
    )


# ── Chart render helper ───────────────────────────────────────

def _render_chart(fig, filename: str, key: str) -> None:
    """Render a Plotly chart and add a PNG download button below it."""
    st.plotly_chart(fig, use_container_width=True)
    try:
        png_bytes = fig.to_image(format='png', width=1400, height=560, scale=2)
        st.download_button(
            "📷 PNG ダウンロード",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            key=key,
        )
    except Exception:
        pass


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
        f'<p style="color:#6b7280;font-size:12px;margin:0;font-weight:500;">🎯 {_html.escape(label)}</p>'
        f'<p style="color:#111827;font-size:22px;font-weight:700;margin:6px 0 0 0;">{rate:.1f}%</p>'
        f'<p style="color:{color};font-size:12px;margin:2px 0 0 0;">{badge}</p>'
        f'<p style="color:#9ca3af;font-size:11px;margin:4px 0 0 0;">'
        f'実績: {_fmt_value(actual)} / 目標: {_fmt_value(target)}&nbsp;({agg_method})</p>'
        f'</div>'
    )


# ── Tab renderers ─────────────────────────────────────────────

def _tab_overview(df: pd.DataFrame, col_info: dict, freq: str = 'D') -> None:
    if df.empty:
        st.info("フィルター条件に一致するデータがありません。サイドバーの条件を緩めてください。")
        return
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
            _render_chart(fig, 'overview_trend.png', 'dl_ov_trend')

    if col_info['category'] and col_info['numeric']:
        st.markdown(
            '<p class="section-title">工程・装置別集計（クイックビュー）</p>',
            unsafe_allow_html=True,
        )
        fig = render_category_chart(df, col_info['category'][0], col_info['numeric'][0])
        if fig:
            _render_chart(fig, 'overview_category.png', 'dl_ov_cat')


def _tab_timeseries(df: pd.DataFrame, col_info: dict, freq: str = 'D') -> None:
    if df.empty:
        st.info("フィルター条件に一致するデータがありません。サイドバーの条件を緩めてください。")
        return
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
            _render_chart(fig, 'timeseries.png', 'dl_ts')
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
        _render_chart(fig_spc, 'spc_chart.png', 'dl_spc')
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
            file_name=f"spc_outliers_{_safe_filename(spc_metric)}.csv",
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
        _render_chart(fig_pb, 'period_bar.png', 'dl_pb')
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
            _render_chart(fig_cmp, 'comparison.png', 'dl_cmp')
            st.caption(
                f"今期（直近 {int(cmp_n)} {unit}）と前期（その前の {int(cmp_n)} {unit}）を同一軸で比較。"
                "前期の日付は今期の日付軸に揃えてシフトしています。"
            )
        else:
            st.error("前期比較グラフを生成できませんでした。データを確認してください。")

        # ── 前期比トレンド表 ─────────────────────────────────
        st.markdown("**前期比トレンド表**")
        try:
            fmt = _PERIOD_FMT.get(freq, '%Y/%m/%d')
            tmp = cmp_df[[date_col, '_period', cmp_metric]].copy()
            tmp[date_col]   = pd.to_datetime(tmp[date_col], errors='coerce')
            tmp[cmp_metric] = pd.to_numeric(tmp[cmp_metric], errors='coerce')

            if freq:
                tmp['_label'] = (
                    tmp[date_col].dt.to_period(freq).dt.start_time.dt.strftime(fmt)
                )
            else:
                tmp['_label'] = tmp[date_col].dt.strftime(fmt)

            agg_tbl = (
                tmp.groupby(['_label', '_period'])[cmp_metric]
                .agg(cmp_agg_en)
                .reset_index()
            )
            pivot_tbl = (
                agg_tbl.pivot(index='_label', columns='_period', values=cmp_metric)
                .reset_index()
            )
            pivot_tbl.columns.name = None

            if '今期' in pivot_tbl.columns and '前期' in pivot_tbl.columns:
                for c in ('今期', '前期'):
                    pivot_tbl[c] = pivot_tbl[c].round(3)
                pivot_tbl['増減'] = (pivot_tbl['今期'] - pivot_tbl['前期']).round(3)
                pivot_tbl['増減率 (%)'] = (
                    (pivot_tbl['今期'] - pivot_tbl['前期'])
                    / pivot_tbl['前期'].abs() * 100
                ).round(1)
                pivot_tbl = (
                    pivot_tbl.rename(columns={'_label': '期間'})
                    [['期間', '今期', '前期', '増減', '増減率 (%)']]
                    .sort_values('期間').reset_index(drop=True)
                )
                st.dataframe(pivot_tbl, use_container_width=True)
                st.download_button(
                    "⬇️ 前期比トレンド表 CSV",
                    data=pivot_tbl.to_csv(index=False).encode('utf-8-sig'),
                    file_name=f"trend_comparison_{_safe_filename(cmp_metric)}.csv",
                    mime="text/csv",
                    key='dl_trend_cmp',
                )
            else:
                st.info("今期・前期のデータが揃っていません。比較期間を広げてください。")
        except Exception:
            st.error("前期比トレンド表の生成に失敗しました。")


def _tab_category(df: pd.DataFrame, col_info: dict) -> None:
    if df.empty:
        st.info("フィルター条件に一致するデータがありません。サイドバーの条件を緩めてください。")
        return
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
        _render_chart(fig, 'category.png', 'dl_cat')
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
        _render_chart(fig_pareto, 'pareto.png', 'dl_pareto')
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
            _render_chart(fig_shift, 'shift.png', 'dl_shift')
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
            _render_chart(fig_hm, 'heatmap.png', 'dl_hm')
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
                file_name=f"ranking_{_safe_filename(rk_cat)}_{_safe_filename(rk_val)}.csv",
                mime="text/csv",
                key='dl_ranking',
            )
    except Exception:
        st.error("ランキングテーブルを生成できませんでした。カテゴリ列と数値列を確認してください。")


def _tab_correlation(df: pd.DataFrame, col_info: dict) -> None:
    if df.empty:
        st.info("フィルター条件に一致するデータがありません。サイドバーの条件を緩めてください。")
        return
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
        _render_chart(fig, 'scatter.png', 'dl_scatter')

    st.markdown(
        '<p class="section-title">プロセスパラメータ 相関ヒートマップ</p>',
        unsafe_allow_html=True,
    )
    fig_hm = render_correlation_heatmap(df, col_info['numeric'])
    if fig_hm:
        _render_chart(fig_hm, 'corr_heatmap.png', 'dl_corr')
    else:
        st.info("相関ヒートマップを生成できませんでした。")


def _build_quality_summary(df: pd.DataFrame, col_info: dict) -> pd.DataFrame:
    role_map = {col: role for role, cols in col_info.items() for col in cols}
    rows = []
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_pct = round(n_missing / n_total * 100, 1) if n_total > 0 else 0.0
        n_unique = int(series.nunique())
        role = role_map.get(col, 'text')
        notes = []
        if role == 'numeric':
            converted = pd.to_numeric(series, errors='coerce')
            bad = int((series.notna() & converted.isna()).sum())
            if bad > 0:
                notes.append(f"数値変換不可: {bad} 件")
        if role == 'date':
            converted = pd.to_datetime(series, errors='coerce')
            bad = int((series.notna() & converted.isna()).sum())
            if bad > 0:
                notes.append(f"日付変換不可: {bad} 件")
        if missing_pct >= 30:
            notes.append("欠損率高")
        rows.append({
            '列名': col,
            'ロール': role,
            '欠損数': n_missing,
            '欠損率(%)': missing_pct,
            'ユニーク数': n_unique,
            '備考': ' / '.join(notes) if notes else '―',
        })
    return pd.DataFrame(rows)


def _tab_data(df: pd.DataFrame, col_info: dict, freq: str = 'D') -> None:
    if df.empty:
        st.info("フィルター条件に一致するデータがありません。サイドバーの条件を緩めてください。")
        return

    # ── データ品質サマリー ────────────────────────────────────
    st.markdown('<p class="section-title">データ品質サマリー</p>', unsafe_allow_html=True)
    n_dup = int(df.duplicated().sum())
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("行数", f"{len(df):,}")
    q2.metric("列数", f"{df.shape[1]:,}")
    q3.metric("重複行", f"{n_dup:,}", delta=f"-{n_dup}" if n_dup else None,
              delta_color="inverse" if n_dup else "off")
    total_cells = len(df) * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    q4.metric("欠損セル率", f"{total_missing / total_cells * 100:.1f}%" if total_cells else "0%")

    quality_df = _build_quality_summary(df, col_info)
    has_issues = quality_df['備考'].ne('―').any()
    with st.expander(
        f"列ごとの品質詳細 {'⚠️ 要確認あり' if has_issues else '✅ 問題なし'}",
        expanded=has_issues,
    ):
        def _highlight_issues(row):
            color = '#fff3cd' if row['備考'] != '―' else ''
            return [f'background-color: {color}'] * len(row)
        st.dataframe(
            quality_df.style.apply(_highlight_issues, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
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

    # ── HTML Report ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📄 印刷レポート")

    numeric_cols = col_info.get('numeric', [])
    if not numeric_cols:
        st.info("数値列がないためレポートを生成できません。")
    else:
        rpt_metrics = st.multiselect(
            "レポートに含める指標",
            numeric_cols,
            default=numeric_cols[:5],
            key='rpt_metrics',
        )
        if st.button("📄 レポートを生成", key='gen_report'):
            html = _build_html_report(df, col_info, freq, rpt_metrics or None)
            st.session_state['_report_html'] = html

        if st.session_state.get('_report_html'):
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            st.download_button(
                "⬇️ HTML レポートをダウンロード",
                data=st.session_state['_report_html'].encode('utf-8'),
                file_name=f"report_{ts}.html",
                mime="text/html",
                key='dl_report',
            )
            st.caption(
                "ダウンロードした HTML をブラウザで開き、"
                "**Ctrl+P（Cmd+P）→ PDF として保存** でPDF出力できます。"
            )


# ── HTML Report builder ───────────────────────────────────────

def _build_html_report(df, col_info, freq='D', selected_metrics=None):
    """Return a self-contained HTML string suitable for browser printing / PDF save."""

    numeric_cols = col_info.get('numeric', [])
    date_col = col_info['date'][0] if col_info['date'] else None
    category_cols = col_info.get('category', [])

    metrics = selected_metrics or numeric_cols[:5]

    # ── KPI section ──────────────────────────────────────────
    kpi_html = ""
    for col in metrics:
        try:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            val = float(series.mean()) if not series.empty else float('nan')
            kpi_html += (
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{_html.escape(col)}</div>'
                f'<div class="kpi-value">{_fmt_value(val)}</div>'
                f'<div class="kpi-sub">平均</div>'
                f'</div>'
            )
        except Exception:
            pass

    # ── Chart section ─────────────────────────────────────────
    # Plotly CDN は <head> で一括ロード → 全チャートで include_plotlyjs=False
    charts_html = ""
    _chart_layout = dict(
        paper_bgcolor='white', plot_bgcolor='white',
        font_color='#111827', margin=dict(t=40, b=40, l=40, r=20),
    )
    if date_col and metrics:
        value_col = metrics[0]
        group_col = category_cols[0] if category_cols else None
        # numeric_cols にリストを渡し selected_metrics で絞り込む（正しいシグネチャ）
        fig = render_timeseries_chart(
            df, date_col, numeric_cols,
            group_col=group_col,
            selected_metrics=[value_col],
            freq=freq,
        )
        if fig is not None:
            fig.update_layout(**_chart_layout)
            charts_html += (
                '<h2 class="section-title">トレンドチャート</h2>'
                + fig.to_html(include_plotlyjs=False, full_html=False)
            )

    if category_cols and metrics:
        cat_col = category_cols[0]
        value_col = metrics[0]
        # agg_method をキーワード引数で指定（4番目の位置は chart_type のため）
        fig2 = render_category_chart(df, cat_col, value_col, agg_method='平均')
        if fig2 is not None:
            fig2.update_layout(**_chart_layout)
            charts_html += (
                '<h2 class="section-title">カテゴリ別集計</h2>'
                + fig2.to_html(include_plotlyjs=False, full_html=False)
            )

    # ── Data summary table ────────────────────────────────────
    summary_html = ""
    if metrics:
        try:
            summary = df[metrics].describe().round(3)
            summary_html = (
                '<h2 class="section-title">統計サマリー</h2>'
                + summary.to_html(classes='summary-table', border=0)
            )
        except Exception:
            pass

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>分析レポート — {generated_at}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', sans-serif; color: #111827; background: #fff; padding: 32px; }}
  h1 {{ font-size: 22px; font-weight: 700; color: #111827; margin-bottom: 4px; }}
  .meta {{ font-size: 12px; color: #6b7280; margin-bottom: 24px; }}
  .section-title {{
    font-size: 15px; font-weight: 600; color: #111827;
    border-left: 4px solid #2563EB; padding-left: 10px;
    margin: 28px 0 14px 0;
  }}
  .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }}
  .kpi-card {{
    flex: 1; min-width: 120px; max-width: 200px;
    padding: 14px 16px; background: #fff;
    border-radius: 10px; border-left: 4px solid #2563EB;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
  }}
  .kpi-label {{ font-size: 11px; color: #6b7280; font-weight: 500; }}
  .kpi-value {{ font-size: 20px; font-weight: 700; color: #111827; margin: 4px 0 2px; }}
  .kpi-sub {{ font-size: 10px; color: #9ca3af; }}
  .summary-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .summary-table th, .summary-table td {{
    border: 1px solid #e5e7eb; padding: 6px 10px; text-align: right;
  }}
  .summary-table th {{ background: #f9fafb; font-weight: 600; text-align: left; }}
  footer {{ font-size: 11px; color: #9ca3af; margin-top: 40px; border-top: 1px solid #e5e7eb; padding-top: 12px; }}
  @media print {{
    body {{ padding: 16px; }}
    .kpi-card {{ box-shadow: none; border: 1px solid #e5e7eb; }}
    footer {{ position: fixed; bottom: 0; width: 100%; }}
  }}
</style>
</head>
<body>
  <h1>生産データ分析レポート</h1>
  <p class="meta">生成日時: {generated_at} &nbsp;|&nbsp; 対象行数: {len(df):,} 行</p>

  <h2 class="section-title">KPI サマリー</h2>
  <div class="kpi-row">{kpi_html}</div>

  {charts_html}
  {summary_html}

  <footer>このレポートは CSV データ分析ダッシュボードにより自動生成されました。</footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    df_raw, col_info, is_demo, freq, auto_refresh, refresh_interval, cond_rules = _render_sidebar()

    date_col = col_info['date'][0] if col_info['date'] else None
    date_range = render_date_filter(df_raw, date_col) if date_col else None
    cat_selections = render_category_filters(df_raw, col_info['category'])

    df = apply_filters(df_raw, cat_selections, date_range, date_col)
    if cond_rules:
        df = _apply_condition_filters(df, cond_rules)
    if len(df) < len(df_raw):
        st.sidebar.caption(f"絞り込み後: {len(df):,} 行 / {len(df_raw):,} 行")

    if is_demo:
        col_banner, col_dl = st.columns([5, 1])
        with col_banner:
            st.markdown(
                '<div class="demo-banner">'
                '🔵 半導体工場サンプルデータを表示中です。'
                'サイドバーから実データ CSV をアップロードすると切り替わります。'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_dl:
            st.download_button(
                "⬇️ CSV",
                data=df_raw.to_csv(index=False).encode("utf-8-sig"),
                file_name="demo_semiconductor.csv",
                mime="text/csv",
                use_container_width=True,
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
        _tab_data(df, col_info, freq)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == '__main__':
    main()
