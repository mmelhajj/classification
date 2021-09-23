from datetime import datetime

import pandas as pd
from pytest import fixture

from RF_classifier.features import peaks_pos, get_curve_length


@fixture
def df():
    data = {'date': [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4),
                     datetime(2020, 1, 5), datetime(2020, 1, 6), datetime(2020, 1, 7)],
            'data': [1, 2, 3, 4, 3, 2, 1],
            }
    df = pd.DataFrame(data)
    return df


def test_if_peaks_are_correctely_dectected(df):
    x = peaks_pos(df, 'date', 'data')
    assert x[0] == datetime(2020, 1, 4)
    assert len(x) == 1


def test_curve_length(df):
    length = get_curve_length(df, 'date', 'data', '2020-1-1', '2020-1-7')
    assert length == (1 + 1) ** 0.5 * (len(df['date']) - 1)

    df1 = df.copy()
    df1['data'] = df['data'] * 2
    length1 = get_curve_length(df1, 'date', 'data', '2020-1-1', '2020-1-7')

    assert length1 > length
