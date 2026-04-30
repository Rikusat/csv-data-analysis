"""Column auto-detection logic for CSV DataFrames."""

import pandas as pd
from typing import Dict, List


DATE_KEYWORDS = [
    'date', '日付', 'month', '年', '月', 'time', 'datetime',
    '日時', '期間', 'week', '週', 'day', '日',
]
CATEGORY_KEYWORDS = [
    'plant', 'line', 'process', 'equipment', '拠点', 'ライン', '工程', '設備',
    'status', '状態', 'type', '種別', '区分', 'category', 'カテゴリ',
    'name', '名前', '名称', 'code', 'コード',
]
NUMERIC_KEYWORDS = [
    'qty', 'amount', 'count', 'num', 'total', 'sum', 'avg', 'rate', 'ratio',
    '数', '量', '率', '金額', '合計', '平均', '最大', '最小', 'price', '価格',
    'cost', 'コスト',
]


def detect_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect column roles automatically.

    Rules (priority order):
      date     – datetime dtype, date keywords, or parseable date strings
      numeric  – numeric dtype or >=80% parseable as number
      category – category keywords OR string with <=20 unique values
      text     – everything else

    Args:
        df: Input DataFrame.

    Returns:
        Dict with keys 'date', 'category', 'numeric', 'text',
        each containing a list of matching column names.
    """
    result: Dict[str, List[str]] = {
        'date': [],
        'category': [],
        'numeric': [],
        'text': [],
    }

    for col in df.columns:
        col_lower = col.lower()
        series = df[col]
        try:
            if _is_date_column(series, col_lower):
                result['date'].append(col)
            elif _is_numeric_column(series, col_lower):
                result['numeric'].append(col)
            elif _is_category_column(series, col_lower):
                result['category'].append(col)
            else:
                result['text'].append(col)
        except Exception:
            result['text'].append(col)

    return result


def prepare_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convert a column to datetime dtype in-place on a copy.

    Args:
        df: Input DataFrame.
        date_col: Name of the column to convert.

    Returns:
        New DataFrame with the column converted to datetime.
    """
    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        pass
    return df


# ── Private helpers ──────────────────────────────────────────


def _is_date_column(series: pd.Series, col_lower: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if any(kw in col_lower for kw in DATE_KEYWORDS):
        try:
            pd.to_datetime(series.dropna().head(20), errors='raise')
            return True
        except Exception:
            pass

    if series.dtype == 'object':
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except Exception:
            pass

    return False


def _is_numeric_column(series: pd.Series, col_lower: str) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True

    if series.dtype == 'object':
        try:
            converted = pd.to_numeric(series.dropna(), errors='coerce')
            total = series.dropna().shape[0]
            if total > 0 and (converted.notna().sum() / total) >= 0.8:
                return True
        except Exception:
            pass

    return False


def _is_category_column(series: pd.Series, col_lower: str) -> bool:
    if any(kw in col_lower for kw in CATEGORY_KEYWORDS):
        return True

    if series.dtype == 'object':
        n_unique = series.nunique()
        n_total = series.shape[0]
        if n_unique <= 20:
            return True
        if n_total > 0 and (n_unique / n_total) <= 0.05:
            return True

    return False
