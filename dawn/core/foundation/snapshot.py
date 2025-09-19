import json, time, pathlib
from .state import get_state, set_state

SNAP_DIR = pathlib.Path("runtime/snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)

def snapshot(tag="auto"):
    s = get_state().__dict__
    p = SNAP_DIR / f"{int(time.time())}_{tag}.json"
    p.write_text(json.dumps(s, indent=2))
    return str(p)

def restore(path: str):
    data = json.loads(pathlib.Path(path).read_text())
    set_state(**{k:v for k,v in data.items() if k in get_state().__dict__})