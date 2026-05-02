"""Plotly chart generation functions for the dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional


_ACCENT = '#2563EB'
_COLORS = px.colors.qualitative.Set2
_MAX_POINTS = 1000
_LINE_DASHES = ['solid', 'dot', 'dash', 'dashdot']


def render_timeseries_chart(
    df: pd.DataFrame,
    date_col: str,
    numeric_cols: List[str],
    group_col: Optional[str] = None,
    selected_metrics: Optional[List[str]] = None,
    agg_method: str = 'mean',
) -> Optional[go.Figure]:
    """
    Generate a time-series line chart.

    Args:
        df: Filtered DataFrame.
        date_col: Date column name.
        numeric_cols: All available numeric columns.
        group_col: Optional column to split lines by colour.
        selected_metrics: Subset of numeric_cols to plot; defaults to first 3.
        agg_method: pandas aggregation string ('mean', 'sum', 'max', 'min').

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

        title_metrics = ', '.join(metrics[:3]) + (f' 他{len(metrics)-3}件' if len(metrics) > 3 else '')

        if group_col and group_col in df_p.columns:
            agg_dict = {m: agg_method for m in metrics}
            df_g = df_p.groupby([date_col, group_col], as_index=False).agg(agg_dict)
            df_g = _downsample(df_g)

            groups = sorted(df_g[group_col].dropna().unique(), key=str)
            group_colors = {grp: _COLORS[i % len(_COLORS)] for i, grp in enumerate(groups)}

            fig = go.Figure()
            for grp in groups:
                sub = df_g[df_g[group_col] == grp]
                color = group_colors[grp]
                for j, metric in enumerate(metrics):
                    label = f"{grp}  {metric}" if len(metrics) > 1 else str(grp)
                    fig.add_scatter(
                        x=sub[date_col], y=sub[metric],
                        name=label, mode='lines+markers',
                        line=dict(color=color, width=2, dash=_LINE_DASHES[j % len(_LINE_DASHES)]),
                        marker=dict(size=4),
                        legendgroup=str(grp),
                    )
            fig.update_layout(title=f"時系列推移: {title_metrics}", template='plotly_white')

        else:
            agg_dict = {m: agg_method for m in metrics}
            df_g = df_p.groupby(date_col, as_index=False).agg(agg_dict)
            df_g = _downsample(df_g)

            fig = go.Figure()
            for i, metric in enumerate(metrics):
                fig.add_scatter(
                    x=df_g[date_col], y=df_g[metric],
                    name=metric, mode='lines+markers',
                    line=dict(color=_COLORS[i % len(_COLORS)], width=2),
                    marker=dict(size=4),
                )
            fig.update_layout(title=f"時系列推移: {title_metrics}", template='plotly_white')

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
            # Drop NaN in size column to prevent Plotly error
            df_p = df_p.dropna(subset=[size_col])
            min_s = df_p[size_col].min()
            if pd.notna(min_s) and min_s <= 0:
                df_p[size_col] = df_p[size_col] - min_s + 1
            kwargs['size'] = size_col

        df_p = _downsample(df_p)
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
        # Drop columns that are entirely NaN or constant (zero variance) after conversion
        df_num = df_num.dropna(axis=1, how='all')
        df_num = df_num.loc[:, df_num.nunique() > 1]
        cols = df_num.columns.tolist()
        if len(cols) < 2:
            return None

        corr = df_num.corr()

        # Build display text: show rounded value or '—' for NaN cells
        corr_rounded = np.round(corr.values, 2)
        text_matrix = np.where(
            np.isnan(corr_rounded),
            '—',
            corr_rounded.astype(str),
        )

        # Replace NaN in z with 0 so Plotly colours the cell neutrally instead of blank
        z_values = np.where(np.isnan(corr.values), 0.0, corr.values)

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=text_matrix,
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


def render_spc_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Generate an SPC (Statistical Process Control) chart.

    Plots daily-averaged values with a center line (CL), upper and lower
    control limits (UCL/LCL = mean ± 3σ).  Out-of-control points are
    highlighted in red.

    Args:
        df: Filtered DataFrame.
        date_col: Date column name.
        value_col: Numeric column to monitor.
        group_col: Optional column to draw one line per group.

    Returns:
        Plotly Figure or None on error.
    """
    try:
        df_p = df.copy()
        df_p[date_col] = pd.to_datetime(df_p[date_col], errors='coerce')
        df_p[value_col] = pd.to_numeric(df_p[value_col], errors='coerce')
        df_p = df_p.dropna(subset=[date_col, value_col])

        if group_col and group_col in df_p.columns:
            agg_full = df_p.groupby([date_col, group_col], as_index=False)[value_col].mean()
        else:
            agg_full = df_p.groupby(date_col, as_index=False)[value_col].mean()

        if agg_full.empty or agg_full[value_col].isna().all():
            return None

        groups = (
            sorted(agg_full[group_col].dropna().unique(), key=str)
            if (group_col and group_col in agg_full.columns)
            else [None]
        )
        # Only draw horizontal reference lines when ≤3 groups (otherwise too cluttered)
        draw_ref_lines = len(groups) <= 3

        fig = go.Figure()

        for i, grp in enumerate(groups):
            sub = (
                agg_full[agg_full[group_col] == grp].copy()
                if grp is not None else agg_full.copy()
            )
            color = _COLORS[i % len(_COLORS)]
            label = str(grp) if grp is not None else value_col

            # Per-group control limits — computed on full data before downsampling
            grp_mean = float(sub[value_col].mean())
            grp_std = float(sub[value_col].std())
            if not np.isfinite(grp_std) or grp_std == 0:
                grp_std = abs(grp_mean) * 0.001 if grp_mean != 0 else 0.001
            grp_ucl = grp_mean + 3 * grp_std
            grp_lcl = grp_mean - 3 * grp_std

            sub['_oc'] = (sub[value_col] > grp_ucl) | (sub[value_col] < grp_lcl)
            sub = _downsample(sub)

            normal = sub[~sub['_oc']]
            oc = sub[sub['_oc']]

            fig.add_scatter(
                x=normal[date_col], y=normal[value_col],
                mode='lines+markers', name=label,
                line=dict(color=color, width=2), marker=dict(size=5),
            )
            if not oc.empty:
                fig.add_scatter(
                    x=oc[date_col], y=oc[value_col],
                    mode='markers', name=f'{label}（管理外）',
                    marker=dict(color='#ef4444', size=10, symbol='x'),
                )

            if draw_ref_lines:
                ref_color = color if grp is not None else '#10b981'
                suffix = f' [{grp}]' if grp is not None else ''
                fig.add_hline(
                    y=grp_mean, line_color=ref_color, line_dash='solid', line_width=1,
                    annotation_text=f'CL={grp_mean:.3f}{suffix}',
                    annotation_position='top right',
                )
                fig.add_hline(
                    y=grp_ucl, line_color=ref_color, line_dash='dash', line_width=1,
                    annotation_text=f'UCL={grp_ucl:.3f}{suffix}',
                    annotation_position='top right',
                )
                if grp_lcl > 0:
                    fig.add_hline(
                        y=grp_lcl, line_color=ref_color, line_dash='dash', line_width=1,
                        annotation_text=f'LCL={grp_lcl:.3f}{suffix}',
                        annotation_position='bottom right',
                    )

        title_suffix = '（グループ別制御限界）' if len(groups) > 1 else ''
        fig.update_layout(
            title=f'SPC 管理図: {value_col}{title_suffix}',
            template='plotly_white',
            hovermode='x unified',
            height=420,
            margin=dict(l=20, r=140, t=50, b=20),
        )
        return fig
    except Exception:
        return None


def render_pareto_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    agg_method: str = '合計',
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Generate a Pareto chart (sorted bar + cumulative percentage line).

    Args:
        df: Filtered DataFrame.
        category_col: Column to group by (e.g. defect type).
        value_col: Numeric column to aggregate.
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

        # Compute total from ALL categories (not just top_n) for correct cumulative %
        grouped_all = (
            df_p.groupby(category_col)[value_col]
            .agg(agg_func)
            .reset_index()
        )
        total = float(pd.to_numeric(grouped_all[value_col], errors='coerce').sum())
        if total == 0 or not np.isfinite(total):
            return None

        grouped = (
            grouped_all
            .sort_values(value_col, ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        grouped['_cum_pct'] = (grouped[value_col].cumsum() / total * 100).round(1)

        fig = go.Figure()
        fig.add_bar(
            x=grouped[category_col], y=grouped[value_col],
            name=f'{value_col}（{agg_method}）',
            marker_color=_ACCENT,
            yaxis='y',
        )
        fig.add_scatter(
            x=grouped[category_col], y=grouped['_cum_pct'],
            name='累積割合(%)', mode='lines+markers',
            line=dict(color='#f59e0b', width=2),
            marker=dict(size=7),
            yaxis='y2',
        )
        # 80% reference line using add_shape (more reliable than add_hline on y2)
        fig.add_shape(
            type='line',
            x0=0, x1=1, xref='paper',
            y0=80, y1=80, yref='y2',
            line=dict(color='#9ca3af', dash='dot', width=1.5),
        )
        fig.add_annotation(
            x=1.01, y=80, xref='paper', yref='y2',
            text='80%', showarrow=False,
            font=dict(color='#9ca3af', size=11),
            xanchor='left',
        )
        fig.update_layout(
            title=f'パレート図: {category_col} × {value_col}',
            template='plotly_white',
            yaxis=dict(title=f'{value_col}（{agg_method}）'),
            yaxis2=dict(
                title='累積割合(%)', overlaying='y', side='right',
                range=[0, 110], ticksuffix='%',
            ),
            height=420,
            margin=dict(l=20, r=80, t=50, b=20),
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
