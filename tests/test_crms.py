import pandas as pd
from crms import psar, indicators, weekly_regime

def fixture(n=260):
    idx=pd.date_range('2025-01-01',periods=n,freq='D',tz='UTC')
    c=pd.Series(range(100,100+n),index=idx,dtype=float)
    return pd.DataFrame({'open':c-.2,'high':c+1,'low':c-1,'close':c,'volume':1000.0},index=idx)

def test_psar_shape_and_finite_tail():
    d=fixture(); p=psar(d)
    assert len(p)==len(d)
    assert p.psar.tail(100).notna().all()

def test_indicator_columns():
    z=indicators(fixture())
    for c in ['psar','macd_hist','rsi','adx','plus_di','minus_di','atr','ema200','rvol20']:
        assert c in z.columns

def test_weekly_regime_is_lagged():
    d=fixture(); w=weekly_regime(d)
    assert list(w.columns)==['weekly_psar','weekly_bull']
    # Early rows must not magically have a completed weekly PSAR history.
    assert w.weekly_psar.iloc[:7].isna().all()
