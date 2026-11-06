# -*- coding: utf-8 -*-
"""P23-8: embed f32 slot_resid weights into occ_resid30.py.

Reads smart_buildings/results/slot_runs/slot_resid.npz, keeps the six
head arrays in float32, packs raw savez -> zlib(9) -> base64, and
replaces the RESID30_W_B64 placeholder in the ship module. Verifies
the roundtrip (shapes + exact f32 bytes) before writing.
"""
import base64
import io
import sys
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIP = (PROJECT_ROOT / "smart_buildings" / "baselines"
        / "occ_resid30.py")
SRC = (PROJECT_ROOT / "smart_buildings" / "results" / "slot_runs"
       / "slot_resid.npz")
KEYS = ("Wz", "bz", "Wa", "ba", "Wg", "bg")
MARK = 'RESID30_W_B64 = "PLACEHOLDER"'


def main():
    z = np.load(SRC)
    arrs = {k: z[k].astype(np.float32) for k in KEYS}
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    payload = base64.b64encode(zlib.compress(buf.getvalue(), 9)).decode()
    print(f"payload: {len(payload) / 1024:.1f} KB", flush=True)
    # roundtrip check before touching the ship module
    back = np.load(io.BytesIO(zlib.decompress(base64.b64decode(payload))))
    for k in KEYS:
        assert back[k].shape == arrs[k].shape, k
        assert np.array_equal(back[k], arrs[k]), k
    text = SHIP.read_text()
    assert MARK in text, "placeholder already replaced?"
    SHIP.write_text(text.replace(MARK, f'RESID30_W_B64 = "{payload}"'))
    print(f"embedded into {SHIP}", flush=True)


if __name__ == "__main__":
    main()
