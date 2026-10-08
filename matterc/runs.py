"""Every experiment gets results/<timestamp>_<name>/{logs,figures,data}/ + config.json + metrics.json."""
import json, os, time
import numpy as np
from .logutil import setup_logger

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    return o

class Run:
    def __init__(self, name, config=None, root="results"):
        self.dir = os.path.join(root, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}")
        for sub in ("logs", "figures", "data"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.log = setup_logger(name, os.path.join(self.dir, "logs"))
        self.metrics = {}
        self.save_json("config.json", config or {})
        self.log.info("run dir: %s", self.dir)

    def path(self, sub, fname): return os.path.join(self.dir, sub, fname)
    def save_json(self, fname, obj):
        with open(os.path.join(self.dir, fname), "w") as f: json.dump(_clean(obj), f, indent=2)
    def log_metric(self, key, value):
        self.metrics[key] = value; self.log.info("metric %s = %s", key, value)
        self.save_json("metrics.json", self.metrics)
    def save_array(self, fname, arr): np.save(self.path("data", fname), arr)
