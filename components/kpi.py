"""KPI card rendering for the Streamlit dashboard."""

import pandas as pd
import streamlit as st
from typing import List, Optional


_CARD_CSS = (
    "padding:16px;background:#fff;border-radius:12px;"
    "box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:4px solid #2563EB;"
    "margin-bottom:8px;"
)


def render_kpi_cards(
    df: pd.DataFrame,
    numeric_cols: List[str],
    date_col: Optional[str] = None,
    agg_method: str = '合計',
    max_cards: int = 5,
) -> None:
    """
    Render a row of KPI metric cards.

    Each card shows the aggregated value for one numeric column and,
    when a date column is present, a day-over-day delta.

    Args:
        df: Filtered DataFrame.
        numeric_cols: Numeric columns to display as KPI cards.
        date_col: Optional date column for delta calculation.
        agg_method: One of '合計', '平均', '最大', '最小'.
        max_cards: Cap the number of cards rendered.
    """
    if df.empty:
        st.info("フィルター結果が 0 件です。絞り込み条件を変更してください。")
        return

    cols_to_show = numeric_cols[:max_cards]
    if not cols_to_show:
        st.info("数値列が見つかりませんでした。")
        return

    columns = st.columns(len(cols_to_show))

    for i, col in enumerate(cols_to_show):
        with columns[i]:
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                value = _aggregate(series, agg_method)
                delta = _day_over_day_delta(df, col, date_col, agg_method)
                _render_card(col, value, delta, agg_method)
            except Exception:
                st.markdown(
                    f'<div style="{_CARD_CSS}">'
                    f'<p style="color:#6b7280;font-size:12px;margin:0;">{col}</p>'
                    f'<p style="color:#ef4444;font-size:14px;margin:4px 0 0 0;">計算エラー</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Private helpers ──────────────────────────────────────────


_LOWER_IS_BETTER_KEYWORDS = [
    'defect', '欠陥', 'ng', 'fail', '不良', 'density', '密度',
    'error', 'エラー', 'reject', '不合格', 'loss', 'ロス',
    'particle', 'パーティクル',
]


def _is_lower_better(label: str) -> bool:
    low = label.lower()
    return any(kw in low for kw in _LOWER_IS_BETTER_KEYWORDS)


def _render_card(
    label: str,
    value: float,
    delta: Optional[float],
    agg_method: str,
) -> None:
    delta_html = ""
    if delta is not None:
        lower_better = _is_lower_better(label)
        positive_is_good = not lower_better
        color = "#10b981" if (delta >= 0) == positive_is_good else "#ef4444"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = (
            f'<p style="color:{color};font-size:12px;margin:4px 0 0 0;">'
            f'{arrow} {_fmt(abs(delta))} (前日比)</p>'
        )

    st.markdown(
        f'<div style="{_CARD_CSS}">'
        f'<p style="color:#6b7280;font-size:12px;margin:0;font-weight:500;">{label}</p>'
        f'<p style="color:#111827;font-size:22px;font-weight:700;margin:6px 0 0 0;">{_fmt(value)}</p>'
        f'<p style="color:#9ca3af;font-size:11px;margin:2px 0 0 0;">{agg_method}</p>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _aggregate(series: pd.Series, method: str) -> float:
    mapping = {'合計': 'sum', '平均': 'mean', '最大': 'max', '最小': 'min'}
    return float(getattr(series, mapping.get(method, 'sum'))())


def _fmt(value: float) -> str:
    try:
        if not isinstance(value, (int, float)) or not (value == value) or abs(value) == float('inf'):
            return '—'
    except Exception:
        return '—'
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value:,.1f}"
    if abs(value - round(value)) < 1e-9:
        return f"{round(value):,}"
    return f"{value:.2f}"


def _day_over_day_delta(
    df: pd.DataFrame,
    col: str,
    date_col: Optional[str],
    agg_method: str,
) -> Optional[float]:
    if not date_col or date_col not in df.columns:
        return None
    try:
        tmp = df.copy()
        tmp['_d'] = pd.to_datetime(tmp[date_col], errors='coerce')
        tmp = tmp.dropna(subset=['_d'])
        if tmp.empty:
            return None

        # Use the two most recent distinct dates — works for any granularity
        sorted_dates = sorted(tmp['_d'].dt.normalize().unique())
        if len(sorted_dates) < 2:
            return None
        latest = sorted_dates[-1]
        prev   = sorted_dates[-2]

        today_s = pd.to_numeric(tmp.loc[tmp['_d'].dt.normalize() == latest, col], errors='coerce').dropna()
        prev_s  = pd.to_numeric(tmp.loc[tmp['_d'].dt.normalize() == prev,   col], errors='coerce').dropna()

        if today_s.empty or prev_s.empty:
            return None

        return _aggregate(today_s, agg_method) - _aggregate(prev_s, agg_method)
    except Exception:
        return None
