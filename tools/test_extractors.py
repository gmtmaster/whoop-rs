#!/usr/bin/env python3
"""Adversarial tests for the gates' EXTRACTORS, which nothing else covers.

    python whoop-rs/tools/test_extractors.py

Every revert-test so far mutated a document, or mutated the source and checked the document still
disagreed. Both prove the COMPARISON works. Neither proves the extraction is right, and that is
where all three wrong published numbers came from:

  - round 2  `clientHello` matched `DeviceFamily.clientHello`, an unrelated ByteArray property
  - round 3  tightened it to require a call parenthesis
  - round 7  `.feed(` matched `reassembler.feed(bytes)`; published 18 called against a true 16

Each fix addressed the instance. The class is: **a gate may only count a name it can qualify.**
`free_fns_called` matches `uniffi.whoop_ffi.<name>`, a namespace, so a namesake cannot reach it.
`codec_methods_called` now binds the receiver. Any future extractor that matches a bare name over a
blob belongs here with a namesake case beside it, or it will find namesakes too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
gate = __import__("docs-vs-code")

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok     {name:<44} {got}")
    else:
        print(f"  FAILED {name:<44} got {got!r}, want {want!r}")
        FAILED.append(name)


# The real shape of RustCodec.kt: two codecs bound by name, and a namesake that is NOT one.
HOLDER = """
import uniffi.whoop_ffi.WhoopCodec
class RustCodec {
    private val gen5 by lazy { WhoopCodec(Gen.GEN5) }
    private val gen4 by lazy { WhoopCodec(Gen.GEN4) }
    private val reassembler = Reassembler()
    fun decode(raw: ByteArray) = gen5.decodeHistory(raw)
    fun live(raw: ByteArray) = gen4.decodeLive(raw)
    fun chunk(b: ByteArray) { for (f in reassembler.feed(b)) handle(f) }
    fun clear() { bondGiveUp.reset(); clockReference.reset() }
}
"""


def main() -> int:
    print("receiver binding")
    check("binds both codec identifiers", gate.codec_receivers(HOLDER), {"gen5", "gen4"})
    check("does not bind the namesake", "reassembler" in gate.codec_receivers(HOLDER), False)
    check("binds nothing in a file with no codec", gate.codec_receivers("val x = Reassembler()"), set())

    print("\ncall detection, the three regressions that shipped")
    orig = gate.kotlin_files
    gate.kotlin_files = lambda: [HOLDER]
    try:
        # decodeHistory and decodeLive are real; `new` is the construction; feed/reset are namesakes.
        called = gate.codec_methods_called(["decode_history", "decode_live", "feed", "reset", "new"])
        check("counts the two real calls plus new", called, 3)
        check("feed alone is not counted", gate.codec_methods_called(["feed"]), 0)
        check("reset alone is not counted", gate.codec_methods_called(["reset"]), 0)
        check("a method nobody calls is not counted", gate.codec_methods_called(["offload_abort"]), 0)

        # A file that names the type only in prose must not construct one.
        gate.kotlin_files = lambda: ["// WhoopCodec (decode history) is described here.\nval r = Reassembler()\nr.feed(b)"]
        check("prose mention constructs nothing", gate.codec_methods_called(["new", "feed"]), 0)
    finally:
        gate.kotlin_files = orig

    print("\nqualified matching, the property that makes a namesake impossible")
    orig_blob = gate.kotlin_blob
    gate.kotlin_blob = lambda: "val x = uniffi.whoop_ffi.hrvRmssd(rr)\nval y = someOther.hrvSdnn(rr)"
    try:
        check("qualified call counts", gate.free_fns_called(["hrv_rmssd"]), 1)
        check("unqualified namesake does not", gate.free_fns_called(["hrv_sdnn"]), 0)
    finally:
        gate.kotlin_blob = orig_blob

    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all extractor checks pass'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
