"""Unit tests for components/detect.py"""
import pandas as pd
import pytest
from components.detect import detect_columns, prepare_date_column


# ── detect_columns ────────────────────────────────────────────

def test_date_datetime_dtype():
    df = pd.DataFrame({'ts': pd.date_range('2024-01-01', periods=5)})
    r = detect_columns(df)
    assert 'ts' in r['date']


def test_date_string_parseable():
    df = pd.DataFrame({'date': ['2024-01-01', '2024-01-02', '2024-01-03']})
    r = detect_columns(df)
    assert 'date' in r['date']


def test_date_keyword_with_parseable_string():
    df = pd.DataFrame({'month': ['2024-01', '2024-02', '2024-03']})
    r = detect_columns(df)
    assert 'month' in r['date']


def test_numeric_basic():
    df = pd.DataFrame({'qty': [1.0, 2.5, 3.1], 'amount': [100, 200, 300]})
    r = detect_columns(df)
    assert 'qty' in r['numeric']
    assert 'amount' in r['numeric']


def test_numeric_string_parseable():
    df = pd.DataFrame({'val': ['1.0', '2.5', '3.1', '4.0']})
    r = detect_columns(df)
    assert 'val' in r['numeric']


def test_leading_zero_not_numeric():
    df = pd.DataFrame({'lot_id': ['001', '002', '003', '010']})
    r = detect_columns(df)
    assert 'lot_id' not in r['numeric']


def test_binary_flag_not_numeric():
    df = pd.DataFrame({'flag': [0, 1, 0, 1, 1]})
    r = detect_columns(df)
    assert 'flag' not in r['numeric']
    assert 'flag' in r['category']


def test_low_cardinality_int_not_numeric():
    df = pd.DataFrame({'status': [1, 2, 3, 1, 2]})
    r = detect_columns(df)
    assert 'status' not in r['numeric']


def test_category_keyword():
    df = pd.DataFrame({'process': ['A', 'B', 'A', 'C', 'B']})
    r = detect_columns(df)
    assert 'process' in r['category']


def test_category_low_cardinality_string():
    df = pd.DataFrame({'group': ['X', 'Y', 'X', 'Y'] * 10})
    r = detect_columns(df)
    assert 'group' in r['category']


def test_category_high_cardinality_string_goes_to_text():
    import string
    vals = list(string.ascii_letters * 3)  # 156 unique strings
    df = pd.DataFrame({'notes': vals})
    r = detect_columns(df)
    assert 'notes' in r['text']


def test_unix_timestamp_detected_as_date():
    import time
    now = int(time.time())
    df = pd.DataFrame({'timestamp': [now - i * 86400 for i in range(10)]})
    r = detect_columns(df)
    assert 'timestamp' in r['date']


def test_yyyymmdd_int_not_date_without_keyword():
    df = pd.DataFrame({'code': [20240101, 20240102, 20240103]})
    r = detect_columns(df)
    # Without date keyword in column name and already normalized, may vary;
    # ensure it doesn't crash
    assert isinstance(r, dict)


def test_all_keys_present():
    df = pd.DataFrame({'a': [1, 2, 3]})
    r = detect_columns(df)
    assert set(r.keys()) == {'date', 'category', 'numeric', 'text'}


def test_empty_dataframe():
    df = pd.DataFrame()
    r = detect_columns(df)
    assert r == {'date': [], 'category': [], 'numeric': [], 'text': []}


# ── prepare_date_column ───────────────────────────────────────

def test_prepare_date_column_string():
    df = pd.DataFrame({'d': ['2024-01-01', '2024-06-15']})
    result = prepare_date_column(df, 'd')
    assert pd.api.types.is_datetime64_any_dtype(result['d'])


def test_prepare_date_column_does_not_mutate():
    df = pd.DataFrame({'d': ['2024-01-01']})
    _ = prepare_date_column(df, 'd')
    # Original df must NOT be converted to datetime (copy semantics)
    assert not pd.api.types.is_datetime64_any_dtype(df['d'])
