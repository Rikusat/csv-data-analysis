"""Column auto-detection logic for CSV DataFrames."""

import pandas as pd
from typing import Dict, List


DATE_KEYWORDS = [
    'date', '日付', 'month', '年', '月', 'time', 'datetime',
    '日時', '期間', 'week', '週', 'day', '日',
    'timestamp', '作成日', '更新日', '登録日', '発生日', '年月',
]
CATEGORY_KEYWORDS = [
    # 汎用
    'plant', 'line', 'process', 'equipment', '拠点', 'ライン', '工程', '設備',
    'status', '状態', 'type', '種別', '区分', 'category', 'カテゴリ',
    'name', '名前', '名称', 'code', 'コード',
    # 半導体
    'lot', 'ロット', 'wafer', 'ウェーハ', 'recipe', 'レシピ',
    'chamber', 'チャンバー', 'layer', 'レイヤー', 'product', '品種',
    'step', 'ステップ', 'shift', 'シフト', 'operator', 'オペレータ',
    'defect_type', '欠陥種別', 'fail', '不良', 'grade', 'グレード',
]
NUMERIC_KEYWORDS = [
    # 汎用
    'qty', 'amount', 'count', 'num', 'total', 'sum', 'avg', 'rate', 'ratio',
    '数', '量', '率', '金額', '合計', '平均', '最大', '最小', 'price', '価格',
    'cost', 'コスト',
    # 半導体
    'yield', '歩留', 'defect', '欠陥', 'thickness', '膜厚', 'uniformity', '均一性',
    'uptime', '稼働率', 'throughput', 'スループット', 'cycle', 'サイクル',
    'roughness', '粗さ', 'particle', 'パーティクル', 'overlay', 'オーバーレイ',
    'cd', 'critical', 'dimension', 'focus', 'dose', 'power', 'pressure',
    'temperature', '温度', 'flow', 'フロー', 'current', '電流', 'voltage', '電圧',
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

    # Numeric dtype: only allow Unix timestamps when column name contains a date keyword
    if pd.api.types.is_numeric_dtype(series):
        if (pd.api.types.is_integer_dtype(series)
                and any(kw in col_lower for kw in DATE_KEYWORDS)):
            sample = series.dropna().head(100)
            if len(sample) > 0:
                try:
                    smin, smax = int(sample.min()), int(sample.max())
                    # Unix timestamp range: ~2001-01-01 to ~2100-01-01 in seconds
                    if 978307200 <= smin and smax <= 4102444800:
                        converted = pd.to_datetime(sample, unit='s', errors='coerce')
                        valid = converted.dropna()
                        if (len(valid) / len(sample) >= 0.8
                                and valid.dt.year.between(2001, 2100).all()):
                            return True
                except Exception:
                    pass
        return False

    if any(kw in col_lower for kw in DATE_KEYWORDS):
        try:
            sample = series.dropna().head(100)
            if len(sample) > 0:
                converted = pd.to_datetime(sample, errors='coerce')
                valid = converted.dropna()
                if len(valid) / len(sample) >= 0.8 and valid.dt.year.between(1900, 2100).all():
                    return True
        except Exception:
            pass

    if pd.api.types.is_string_dtype(series):
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        try:
            converted = pd.to_datetime(sample, errors='coerce')
            valid = converted.dropna()
            if len(valid) / len(sample) >= 0.8 and valid.dt.year.between(1900, 2100).all():
                return True
        except Exception:
            pass

    return False


def _is_numeric_column(series: pd.Series, col_lower: str) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        # Binary flag columns (only 0 and 1) → hand off to category classification
        unique_vals = set(series.dropna().unique())
        if unique_vals <= {0, 1}:
            return False
        # Very low cardinality integers without a numeric keyword → likely a status/code
        if (pd.api.types.is_integer_dtype(series)
                and series.nunique() <= 5
                and not any(kw in col_lower for kw in NUMERIC_KEYWORDS)):
            return False
        return True

    if pd.api.types.is_string_dtype(series):
        # Preserve leading-zero strings (codes, lot IDs, part numbers, zip codes)
        non_null = series.dropna()
        if non_null.astype(str).str.strip().str.match(r'^0\d').any():
            return False
        try:
            converted = pd.to_numeric(non_null, errors='coerce')
            total = non_null.shape[0]
            if total > 0 and (converted.notna().sum() / total) >= 0.8:
                return True
        except Exception:
            pass

    return False


def _is_category_column(series: pd.Series, col_lower: str) -> bool:
    if series.dropna().empty:
        return False

    n_unique = series.nunique()

    # Category keywords apply to any dtype; cap at 200 unique to avoid flooding UI filters
    if any(kw in col_lower for kw in CATEGORY_KEYWORDS):
        if n_unique <= 200:
            return True

    # Low-cardinality numeric (int or float) → binary flag, status code, or small enum
    if pd.api.types.is_numeric_dtype(series) and n_unique <= 10:
        return True

    if pd.api.types.is_string_dtype(series):
        n_total = series.shape[0]
        if n_unique <= 20:
            return True
        if n_total > 0 and (n_unique / n_total) <= 0.05:
            return True

    return False
