import pytest
import pandas as pd
import numpy as np
from data import make_windows, TimeSeriesDataset

def test_make_windows():
    # Create a fake group_df
    df = pd.DataFrame({
        "latitude": np.arange(10),
        "longitude": np.arange(10),
        "altitude": np.arange(10),
        "radiometric_intensity": np.arange(10),
        "sensor_idx": np.arange(10),
        "reentry_phase": np.random.randint(0, 2, 10)
    })
    windows = make_windows(df, seq_len=5)
    assert len(windows) == 2
    for w in windows:
        assert w["x_numeric"].shape == (5, 4)
        assert w["x_sensor"].shape == (5,)
        assert w["y"].shape == (5,)

def test_timeseries_dataset():
    # Use the windows from above
    df = pd.DataFrame({
        "latitude": np.arange(10),
        "longitude": np.arange(10),
        "altitude": np.arange(10),
        "radiometric_intensity": np.arange(10),
        "sensor_idx": np.arange(10),
        "reentry_phase": np.random.randint(0, 2, 10)
    })
    windows = make_windows(df, seq_len=5)
    dataset = TimeSeriesDataset(windows, seq_len=5)
    x_numeric, x_sensor, y = dataset[0]
    assert x_numeric.shape == (5, 4)
    assert x_sensor.shape == (5,)
    assert y.shape == (5,)
