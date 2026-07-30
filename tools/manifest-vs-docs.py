#!/usr/bin/env python3
"""Does every figure in `az-manifest.toml` actually appear in the document it names?

    python whoop-rs/tools/manifest-vs-docs.py

`az-sweep.py` compares the manifest against what the harnesses PRINT. Nothing compared the manifest
against the DOCUMENTS, and the two can drift apart in either direction: a value synced to a fresh
measurement without the document being corrected sweeps green while the document still claims the old
number, and a figure filed under the wrong document leaves the sweep pointing a reader at a section
that never held it.

Every numeric token of a `documented` value must appear in the named document; a composite value like
`96.4 / 0.7 / 94.9` is checked token by token, because the document spells those across a table. When a
value is absent from its own document but present in another, that other document is named — a wrong
pointer, not a wrong number. Exit status is 1 when anything is unaccounted for.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    print("needs Python 3.11+ for tomllib", file=sys.stderr)
    raise SystemExit(2)

HERE = Path(__file__).resolve().parent
WHOOP = HERE.parent.parent
MANIFEST = HERE / "az-manifest.toml"
DOCS = WHOOP / "dev-notes" / "noop-tan"

TOKEN = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")


def spellings(tok: str) -> list[str]:
    """Every way a document may legitimately write one numeric token."""
    out = {tok, tok.lstrip("+")}
    for s in list(out):
        out.add(s.replace("-", "−"))
        out.add(s.replace("−", "-"))
    for s in list(out):
        bare = s.lstrip("-−+").replace(",", "")
        if bare.replace(".", "").isdigit() and len(bare.split(".")[0]) > 3:
            whole, _, frac = bare.partition(".")
            grouped = f"{int(whole):,}" + (f".{frac}" if frac else "")
            sign = s[0] if s[0] in "-−" else ""
            out |= {sign + grouped, sign + bare}
    return sorted(out)


def present(tok: str, text: str) -> bool:
    return any(re.search(rf"(?<![\d.,]){re.escape(s)}(?![\d])", text) for s in spellings(tok))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    man = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    texts = {p.name: p.read_text(encoding="utf-8") for p in DOCS.glob("*.md")}
    problems = []
    for f in man["figure"]:
        text = texts.get(f["doc"])
        if text is None:
            problems.append(("NO SUCH DOCUMENT", f, f["doc"]))
            continue
        toks = [t.rstrip(",") for t in TOKEN.findall(str(f["documented"]))]
        missing = [t for t in toks if not present(t, text)]
        if not missing:
            continue
        elsewhere = [n for n, o in texts.items() if n != f["doc"] and all(present(t, o) for t in missing)]
        problems.append((f"FILED UNDER THE WRONG DOCUMENT — it is in {elsewhere[0]}" if elsewhere else "IN NO DOCUMENT",
                         f, ", ".join(missing)))
    for kind, f, detail in problems:
        print(f"{kind}  [{f.get('step','-')}] {f['doc']} · {f['section']}")
        print(f"   figure      {f['label']}")
        print(f"   documented  {f['documented']}")
        print(f"   unaccounted {detail}\n")
    print(f"{len(man['figure'])} figures | {len(problems)} whose value does not appear in the document they name")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
