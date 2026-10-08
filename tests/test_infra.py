import json, os, logging, numpy as np
from matterc.runs import Run
from matterc.logutil import setup_logger
from matterc import viz

def test_run_creates_structure(tmp_path):
    r = Run("t", {"a": np.float64(1.5)}, root=str(tmp_path))
    for sub in ("logs", "figures", "data"): assert os.path.isdir(r.path(sub, ""))
    r.log_metric("x", np.array([1, 2]))
    assert json.load(open(os.path.join(r.dir, "metrics.json")))["x"] == [1, 2]
    assert os.path.exists(os.path.join(r.dir, "logs", "t.log"))

def test_logger_writes_file(tmp_path):
    lg = setup_logger("tt", str(tmp_path)); lg.info("hello")
    for h in lg.handlers: h.flush()
    assert "hello" in open(tmp_path / "tt.log").read()

def test_viz_files(tmp_path):
    viz.plot_fitness([0, 1, 2], str(tmp_path / "f.png"))
    viz.plot_heatmap(np.ones((2, 2)), [1, 2], [0, 1], str(tmp_path / "h.png"))
    assert (tmp_path / "f.png").stat().st_size > 0 and (tmp_path / "h.png").stat().st_size > 0
