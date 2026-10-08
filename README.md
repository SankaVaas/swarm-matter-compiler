# swarm-matter-compiler

Compile local rules for minimal, communication-free robot swarms from a target
continuum material law, and study the limits (universality) of what such swarms can emulate.

Robots: 1-bit contact + 1-bit global light + tiny FSM. No radio, no map. The object is the only coordination channel.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python experiments/sindy_demo.py        # SINDy sanity check
python experiments/phase_diagram.py     # first (small) phase sweep
```
See `docs/ROADMAP.md`. Layout: `matterc/` (sim, controller, evolve, coarse_grain, discover), `experiments/`, `tests/`.

## Every step = test + log + figure
```bash
pytest                                  # -> results/tests/{junit.xml,pytest.log}
python experiments/phase0_transport.py  # -> results/<time>_phase0_transport/{logs,figures,data,config.json,metrics.json}
python experiments/phase_diagram.py
```
