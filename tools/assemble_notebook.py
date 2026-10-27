#!/usr/bin/env python3
"""Assemble the submission notebook from repo policy sources, verbatim.

Reads class blocks from smart_buildings/baselines/*.py and appends them as a
new controller cell after the existing chain cell, then re-points
SubmissionPolicy at the requested class. Verifies every block is copied
exactly (modulo trailing whitespace).

Usage: assemble_notebook.py <src_ipynb> <dst_ipynb> <SubmitClass> <CLASS...> <file:CLASS...>
Each CLASS is taken from zone_pi.py unless prefixed with file:.
"""
import json
import re
import sys


def block(text, name):
    m = re.search(r"^class " + name + r"\b.*?(?=^class |\Z)", text, re.S | re.M)
    if not m:
        raise KeyError(f"class {name} not found")
    return m.group(0).rstrip() + "\n"


def main():
    src, dst, submit_cls, wants = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    repo = {}
    cells_src = ["zone_pi", "occ_gated", "occ_gated2", "occ_gated3"]

    def load(mod):
        if mod not in repo:
            repo[mod] = open(f"smart_buildings/baselines/{mod}.py").read()
        return repo[mod]

    parts = []
    for w in wants:
        if ":" in w:
            mod, cls = w.split(":")
        else:
            mod, cls = "zone_pi", w
        if cls == "<consts>":
            consts = [l for l in load(mod).splitlines()
                      if re.match(r"^[A-Z_][A-Z_0-9]* *=", l)]
            parts.append("\n".join(consts))
        else:
            parts.append(block(load(mod), cls))
    new_cell = "\n\n".join(parts)

    nb = json.load(open(src))
    cells = nb["cells"]
    # Find the controller chain cell (defines DsatGen31Policy).
    chain_idx = next(
        i for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "class DsatGen31Policy" in "".join(c["source"])
    )
    # Remove any previously assembled occ cells (idempotent re-run).
    ASSEMBLE_MARK = "# >>> assembled occ cells (verbatim from repo) >>>"
    cells[:] = [c for c in cells
                if not (c["cell_type"] == "code" and "".join(c["source"]).startswith(ASSEMBLE_MARK))]
    chain_idx = next(
        i for i, c in enumerate(cells)
        if c["cell_type"] == "code" and "class DsatGen31Policy" in "".join(c["source"])
    )
    occ_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ASSEMBLE_MARK + "\n", new_cell],
    }
    cells.insert(chain_idx + 1, occ_cell)
    # Re-point SubmissionPolicy.
    for c in cells:
        if c["cell_type"] == "code" and "class SubmissionPolicy" in "".join(c["source"]):
            src_text = "".join(c["source"])
            src_text = re.sub(r"class SubmissionPolicy\(\w+(, [^)]*)?\):",
                              f"class SubmissionPolicy({submit_cls}\\1):", src_text)
            c["source"] = [src_text]
            c["execution_count"] = None
            c["outputs"] = []
    json.dump(nb, open(dst, "w"), indent=1)
    print(f"wrote {dst} with SubmissionPolicy({submit_cls})")


if __name__ == "__main__":
    main()
