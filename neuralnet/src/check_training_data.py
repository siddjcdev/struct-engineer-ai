# check_training_data.py
import json
from pathlib import Path
p = Path('tmd_training_data_peer.json')
if not p.exists():
    print("MISSING")
else:
    data = json.loads(p.read_text())
    print("top keys:", list(data.keys()))
    samples = data.get('samples', [])
    print("samples count:", len(samples))
    if samples:
        print("sample[0] keys:", list(samples[0].keys()))