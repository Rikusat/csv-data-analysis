"""Plotly chart generation functions for the dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional


_ACCENT = '#2563EB'
_COLORS = px.colors.qualitative.Set2
_MAX_POINTS = 1000


def render_timeseries_chart(
    df: pd.DataFrame,
    date_col: str,
    numeric_cols: List[str],
    group_col: Optional[str] = None,
    selected_metrics: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    """
    Generate a time-series line chart.

    Args:
        df: Filtered DataFrame.
        date_col: Date column name.
        numeric_cols: All available numeric columns.
        group_col: Optional column to split lines by colour.
        selected_metrics: Subset of numeric_cols to plot; defaults to first 3.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        df_p = df.copy()
        df_p[date_col] = pd.to_datetime(df_p[date_col], errors='coerce')
        df_p = df_p.dropna(subset=[date_col])

        metrics = [m for m in (selected_metrics or numeric_cols[:3]) if m in df_p.columns]
        if not metrics:
            return None

        if group_col and group_col in df_p.columns:
            agg = {m: 'sum' for m in metrics}
            df_g = df_p.groupby([date_col, group_col], as_index=False).agg(agg)
            df_g = _downsample(df_g)
            fig = px.line(
                df_g, x=date_col, y=metrics[0], color=group_col,
                title=f"時系列推移: {metrics[0]}",
                color_discrete_sequence=_COLORS,
                template='plotly_white',
            )
            for extra in metrics[1:]:
                for grp in df_g[group_col].unique():
                    sub = df_g[df_g[group_col] == grp]
                    fig.add_scatter(
                        x=sub[date_col], y=sub[extra],
                        name=f"{grp} – {extra}",
                        mode='lines',
                        line=dict(dash='dot'),
                    )
        else:
            agg = {m: 'sum' for m in metrics}
            df_g = df_p.groupby(date_col, as_index=False).agg(agg)
            df_g = _downsample(df_g)
            fig = go.Figure()
            for i, metric in enumerate(metrics):
                fig.add_scatter(
                    x=df_g[date_col], y=df_g[metric],
                    name=metric, mode='lines+markers',
                    line=dict(color=_COLORS[i % len(_COLORS)], width=2),
                    marker=dict(size=4),
                )
            fig.update_layout(title="時系列推移", template='plotly_white')

        fig.update_layout(
            xaxis_title=date_col,
            legend_title="凡例",
            height=420,
            hovermode='x unified',
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig
    except Exception:
        return None


def render_category_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    chart_type: str = '棒グラフ',
    agg_method: str = '合計',
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Generate a category aggregation chart.

    Args:
        df: Filtered DataFrame.
        category_col: Column to group by.
        value_col: Numeric column to aggregate.
        chart_type: '棒グラフ' | '積み上げ棒グラフ' | '円グラフ'
        agg_method: '合計' | '平均' | '最大' | '最小'
        top_n: Show only the top-N categories.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        _agg_map = {'合計': 'sum', '平均': 'mean', '最大': 'max', '最小': 'min'}
        agg_func = _agg_map.get(agg_method, 'sum')

        df_p = df.copy()
        df_p[value_col] = pd.to_numeric(df_p[value_col], errors='coerce')
        grouped = (
            df_p.groupby(category_col)[value_col]
            .agg(agg_func)
            .reset_index()
            .sort_values(value_col, ascending=False)
            .head(top_n)
        )

        title = f"{category_col} 別 {value_col}（{agg_method}）"

        if chart_type == '円グラフ':
            fig = px.pie(
                grouped, names=category_col, values=value_col,
                title=title, color_discrete_sequence=_COLORS,
                template='plotly_white',
            )
        elif chart_type == '積み上げ棒グラフ':
            fig = px.bar(
                grouped, x=category_col, y=value_col,
                title=title, color=category_col,
                color_discrete_sequence=_COLORS,
                template='plotly_white',
            )
            fig.update_layout(barmode='stack')
        else:
            fig = px.bar(
                grouped, x=category_col, y=value_col,
                title=title, color=category_col,
                color_discrete_sequence=_COLORS,
                template='plotly_white',
                text_auto=True,
            )

        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=(chart_type != '棒グラフ'),
        )
        return fig
    except Exception:
        return None


def render_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Generate a scatter (or bubble) plot.

    Args:
        df: Filtered DataFrame.
        x_col: Column for the X axis.
        y_col: Column for the Y axis.
        color_col: Optional column for colour coding.
        size_col: Optional numeric column for bubble size.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        df_p = df.copy()
        df_p[x_col] = pd.to_numeric(df_p[x_col], errors='coerce')
        df_p[y_col] = pd.to_numeric(df_p[y_col], errors='coerce')
        df_p = df_p.dropna(subset=[x_col, y_col])
        df_p = _downsample(df_p)

        kwargs: dict = dict(
            x=x_col, y=y_col,
            title=f"散布図: {x_col} vs {y_col}",
            template='plotly_white',
            color_discrete_sequence=_COLORS,
            opacity=0.7,
        )
        if color_col and color_col in df_p.columns:
            kwargs['color'] = color_col
        if size_col and size_col in df_p.columns:
            df_p[size_col] = pd.to_numeric(df_p[size_col], errors='coerce')
            min_s = df_p[size_col].min()
            if pd.notna(min_s) and min_s < 0:
                df_p[size_col] = df_p[size_col] - min_s + 1
            kwargs['size'] = size_col

        fig = px.scatter(df_p, **kwargs)
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    except Exception:
        return None


def render_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Optional[go.Figure]:
    """
    Generate a correlation heatmap for numeric columns.

    Args:
        df: Filtered DataFrame.
        numeric_cols: Columns to include in the correlation matrix.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        cols = [c for c in numeric_cols if c in df.columns]
        if len(cols) < 2:
            return None

        df_num = df[cols].apply(pd.to_numeric, errors='coerce')
        corr = df_num.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=11),
            hoverongaps=False,
        ))
        fig.update_layout(
            title="数値列の相関ヒートマップ",
            template='plotly_white',
            height=max(350, len(cols) * 55),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(tickangle=-30),
        )
        return fig
    except Exception:
        return None


# ── Private helpers ──────────────────────────────────────────


def _downsample(df: pd.DataFrame, max_points: int = _MAX_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)
