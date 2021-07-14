
from openwater.split import split_time_series
import numpy as np


def test_create_split_windows():
    grp = {
        'DummyModel':{
            'inputs':np.zeros((1,1,10000))
        }
    }

    BREAKS = [
        [100,1000,5000],
        [0,100,1000,5000],
        [100,1000,5000,10000],
        [0,100,1000,5000,10000]
    ]

    for breaks in BREAKS:
        windows = split_time_series(grp,10,breaks)
        assert len(windows)==4
        assert windows[0] == (0,100)
        assert windows[1] == (100,1000)
        assert windows[2] == (1000,5000)
        assert windows[3] == (5000,10000)

    BREAKS = [
        [],
        [0],
        [10000],
        [0,10000]
    ]

    for breaks in BREAKS:
        windows = split_time_series(grp,11,breaks)
        assert len(windows)==1
        assert windows[0] == (0,10000)

    windows = split_time_series(grp,11,None)
    assert len(windows)==11
    assert windows[0] == (0,909)
    assert windows[1] == (909,1818)
    assert windows[10] == (9090,10000)

