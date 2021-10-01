from datetime import datetime

import pandas as pd
from pytest import fixture

from RF_classifier.features import get_curve_length, slope_r2
from RF_classifier.features import slope_moving_windows


@fixture
def df():
    data = {'date': [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4),
                     datetime(2020, 1, 5), datetime(2020, 1, 6), datetime(2020, 1, 7)],
            'data': [1, 2, 3, 4, 3, 2, 1],
            }
    df = pd.DataFrame(data)
    return df


def test_curve_length(df):
    length = get_curve_length(df, 'date', 'data', '2020-1-1', '2020-1-7')
    assert length == (1 + 1) ** 0.5 * (len(df['date']) - 1)

    df1 = df.copy()
    df1['data'] = df['data'] * 2
    length1 = get_curve_length(df1, 'date', 'data', '2020-1-1', '2020-1-7')

    assert length1 > length


def test_slope_moving_windows(df):
    s = slope_moving_windows(df, 5, 'date', 'data')
    assert s[0] == slope_r2([0, 1, 2, 3, 4], [1, 2, 3, 4, 3])[0]
    assert s[1] == slope_r2([1, 2, 3, 4, 5], [2, 3, 4, 3, 2])[0]
    assert s[2] == slope_r2([2, 3, 4, 5, 6], [3, 4, 3, 2, 1])[0]

    s = slope_moving_windows(df, 7, 'date', 'data')
    assert s[0] == slope_r2([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 3, 2, 1])[0]
