"""Unit tests for utility functions in streamlit_app.py"""
import io
import sys
import types
import pandas as pd
import pytest

# Stub streamlit before importing streamlit_app so tests run without a browser
st_stub = types.ModuleType('streamlit')
st_stub.cache_data = lambda f=None, **kw: (f if f else lambda g: g)
st_stub.session_state = {}
for _attr in ('sidebar', 'spinner', 'error', 'info', 'warning', 'success',
              'caption', 'markdown', 'dataframe', 'plotly_chart',
              'download_button', 'columns', 'tabs', 'expander',
              'selectbox', 'multiselect', 'text_input', 'number_input',
              'slider', 'checkbox', 'button', 'file_uploader', 'metric',
              'set_page_config', 'rerun'):
    setattr(st_stub, _attr, lambda *a, **kw: None)
sys.modules.setdefault('streamlit', st_stub)

# Stub plotly submodules so charts.py module-level code doesn't crash
_px = types.ModuleType('plotly.express')
_colors_qual = types.SimpleNamespace(Set2=['#1f77b4'])
_px.colors = types.SimpleNamespace(qualitative=_colors_qual)
_px.line = _px.bar = _px.scatter = _px.imshow = lambda *a, **kw: None
sys.modules.setdefault('plotly', types.ModuleType('plotly'))
sys.modules.setdefault('plotly.express', _px)
_go = types.ModuleType('plotly.graph_objects')
_go.Figure = object
_go.Scatter = _go.Bar = _go.Heatmap = lambda *a, **kw: None
sys.modules.setdefault('plotly.graph_objects', _go)

import importlib
app = importlib.import_module('streamlit_app')


# ── _safe_filename ────────────────────────────────────────────

def test_safe_filename_normal():
    assert app._safe_filename('YIELD_RATE') == 'YIELD_RATE'


def test_safe_filename_spaces():
    assert app._safe_filename('my col') == 'my_col'


def test_safe_filename_slashes():
    assert app._safe_filename('a/b\\c') == 'a_b_c'


def test_safe_filename_special():
    assert app._safe_filename('col:*?"<>|') == 'col'


def test_safe_filename_empty_fallback():
    assert app._safe_filename('') == 'col'


# ── _normalize_dataframe ──────────────────────────────────────

def test_normalize_strips_whitespace_columns():
    df = pd.DataFrame({' col_a ': [1, 2], 'col_b': [3, 4]})
    result = app._normalize_dataframe(df)
    assert 'col_a' in result.columns


def test_normalize_deduplicates_columns():
    df = pd.DataFrame([[1, 2, 3]], columns=['a', 'a', 'b'])
    result = app._normalize_dataframe(df)
    assert list(result.columns) == ['a', 'a_2', 'b']


def test_normalize_yyyymmdd_int_to_datetime():
    df = pd.DataFrame({'date_col': [20240101, 20240201, 20240301]})
    result = app._normalize_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(result['date_col'])


def test_normalize_numeric_string_conversion():
    df = pd.DataFrame({'val': ['1,000', '2,500', '3,100']})
    result = app._normalize_dataframe(df)
    assert pd.api.types.is_numeric_dtype(result['val'])


# ── _build_quality_summary ────────────────────────────────────

def test_quality_summary_columns():
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    col_info = {'date': [], 'category': ['b'], 'numeric': ['a'], 'text': []}
    result = app._build_quality_summary(df, col_info)
    assert set(result.columns) >= {'列名', 'ロール', '欠損数', '欠損率(%)', 'ユニーク数', '備考'}


def test_quality_summary_missing_rate():
    df = pd.DataFrame({'x': [1.0, None, None, None]})
    col_info = {'date': [], 'category': [], 'numeric': ['x'], 'text': []}
    result = app._build_quality_summary(df, col_info)
    row = result[result['列名'] == 'x'].iloc[0]
    assert row['欠損数'] == 3
    assert row['欠損率(%)'] == 75.0


def test_quality_summary_flags_high_missing():
    df = pd.DataFrame({'x': [1.0] + [None] * 9})
    col_info = {'date': [], 'category': [], 'numeric': ['x'], 'text': []}
    result = app._build_quality_summary(df, col_info)
    row = result[result['列名'] == 'x'].iloc[0]
    assert '欠損率高' in row['備考']


# ── _apply_condition_filters ──────────────────────────────────

def test_condition_filter_eq():
    df = pd.DataFrame({'val': [1, 2, 3, 4]})
    result = app._apply_condition_filters(df, [('val', '=', '2')])
    assert list(result['val']) == [2]


def test_condition_filter_gt():
    df = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
    result = app._apply_condition_filters(df, [('val', '>', '1.5')])
    assert list(result['val']) == [2.0, 3.0]


def test_condition_filter_contains():
    df = pd.DataFrame({'name': ['alpha', 'beta', 'gamma']})
    result = app._apply_condition_filters(df, [('name', '含む', 'al')])
    assert list(result['name']) == ['alpha']


def test_condition_filter_missing_col_ignored():
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = app._apply_condition_filters(df, [('nonexistent', '=', '1')])
    assert len(result) == 3
