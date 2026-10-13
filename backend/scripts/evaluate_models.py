import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'models/metadata.json'
if not p.exists(): raise SystemExit('Train models first.')
meta=json.loads(p.read_text()); print(json.dumps(meta['metrics'],indent=2))
