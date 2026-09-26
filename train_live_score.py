"""Fit the frozen Phase31 ExtraTrees percentile model on historical trades.

Run explicitly on the Railway service with CRMS_SCORE_PATH on a mounted volume.
The Phase30 dataset builder uses historical Binance 1h candles and frozen
Phase29 exit labels. This is a one-time, resource-intensive operation.
"""
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import crms

# Restrict the refit window to recent completed cycles so hourly history fits
# the production service's memory budget; older research remains unchanged.
original_fetch = crms.fetch
training_start = os.getenv("CRMS_TRAIN_START", "2023-01-01")
crms.fetch = lambda symbol, start=training_start: original_fetch(symbol, start)

source = Path(__file__).with_name("sar_phase30_entry_filters.py").read_text()
prefix = source.split("# univariate filters:")[0]
exec(compile(prefix, "phase30_dataset", "exec"))
if len(D) < 430:
    raise RuntimeError("insufficient historical events for Phase31 refit")
cutoff = os.getenv("CRMS_SCORE_CUTOFF")
train = D if not cutoff else D[D.ts < cutoff]
if len(train) < 430:
    raise RuntimeError("insufficient events before cutoff")
y = np.log1p(np.clip(train.ret.to_numpy(float), -.95, None))
model = ExtraTreesRegressor(n_estimators=800, max_depth=8,
                            min_samples_leaf=15, n_jobs=1, random_state=315)
model.fit(train[fn], y)
reference = np.sort(model.predict(train[fn]))
path = Path(os.getenv("CRMS_SCORE_PATH", "/data/phase31_score.joblib"))
path.parent.mkdir(parents=True, exist_ok=True)
temp = path.with_suffix(".tmp")
joblib.dump({"model": model, "reference": reference, "features": fn,
             "trained_through": str(train.ts.max()), "n": len(train)}, temp)
os.replace(temp, path)
print("SCORE_MODEL|READY|path=%s|n=%d|through=%s" %
      (path, len(train), train.ts.max()), flush=True)
