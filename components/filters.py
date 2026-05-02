"""Filter UI generation for the Streamlit sidebar."""

import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple


def render_category_filters(
    df: pd.DataFrame,
    category_cols: List[str],
    max_filters: int = 3,
) -> Dict[str, Optional[str]]:
    """
    Render category selectbox filters in the Streamlit sidebar.

    Args:
        df: Input DataFrame.
        category_cols: List of category column names.
        max_filters: Maximum number of filters to render.

    Returns:
        Dict mapping column name to selected value, or None for 'すべて'.
    """
    selections: Dict[str, Optional[str]] = {}

    for col in category_cols[:max_filters]:
        try:
            unique_vals = sorted(df[col].dropna().unique().tolist(), key=str)
            try:
                unique_vals = sorted(unique_vals, key=float)
            except (ValueError, TypeError):
                pass
            options = ['すべて'] + [str(v) for v in unique_vals]
            selected = st.sidebar.selectbox(
                label=f"🏷️ {col}",
                options=options,
                key=f"filter_cat_{col}",
            )
            selections[col] = None if selected == 'すべて' else selected
        except Exception:
            selections[col] = None

    hidden = len(category_cols) - max_filters
    if hidden > 0:
        hidden_names = ', '.join(f'`{c}`' for c in category_cols[max_filters:])
        st.sidebar.caption(f"他 {hidden} 件のカテゴリ列はフィルター対象外です: {hidden_names}")

    return selections


def render_date_filter(
    df: pd.DataFrame,
    date_col: str,
) -> Optional[Tuple]:
    """
    Render a date-range picker in the Streamlit sidebar.

    Args:
        df: Input DataFrame.
        date_col: Name of the date column.

    Returns:
        Tuple (start_date, end_date) of date objects, or None if unavailable.
    """
    try:
        dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
        if dates.empty:
            return None

        min_date = dates.min().date()
        max_date = dates.max().date()
        if min_date == max_date:
            return None

        st.sidebar.markdown("📅 **日付範囲**")
        date_range = st.sidebar.date_input(
            label="期間を選択",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="filter_date_range",
        )

        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            return date_range[0], date_range[1]
        return None
    except Exception:
        return None


def apply_filters(
    df: pd.DataFrame,
    category_selections: Dict[str, Optional[str]],
    date_range: Optional[Tuple],
    date_col: Optional[str],
) -> pd.DataFrame:
    """
    Apply category and date-range filters to a DataFrame.

    Args:
        df: Input DataFrame.
        category_selections: Dict of column -> selected value (None = all).
        date_range: Tuple (start_date, end_date) or None.
        date_col: Name of the date column, or None.

    Returns:
        Filtered DataFrame copy.
    """
    filtered = df.copy()

    for col, val in category_selections.items():
        if val is not None and col in filtered.columns:
            try:
                filtered = filtered[filtered[col].astype(str) == val]
            except Exception:
                pass

    if date_range and date_col and date_col in filtered.columns:
        try:
            date_series = pd.to_datetime(filtered[date_col], errors='coerce')
            # Strip timezone so comparison with naive Timestamps from Streamlit's date picker works
            if hasattr(date_series.dtype, 'tz') and date_series.dtype.tz is not None:
                date_series = date_series.dt.tz_localize(None)
            start = pd.Timestamp(date_range[0])
            # Add 1 day so the full end day (including any timestamp up to 23:59:59) is included
            end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
            filtered = filtered[(date_series >= start) & (date_series < end)]
        except Exception:
            pass

    return filtered.reset_index(drop=True)
