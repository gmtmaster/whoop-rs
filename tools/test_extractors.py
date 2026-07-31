#!/usr/bin/env python3
"""Adversarial tests for the gates' EXTRACTORS, which nothing else covers.

    python whoop-rs/tools/test_extractors.py

Every revert-test so far mutated a document, or mutated the source and checked the document still
disagreed. Both prove the COMPARISON works. Neither proves the extraction is right, and that is
where all four wrong published numbers came from:

  - round 2  `clientHello` matched `DeviceFamily.clientHello`, an unrelated ByteArray property
  - round 3  tightened it to require a call parenthesis; still receiver-blind
  - round 7  `.feed(` matched `reassembler.feed(bytes)`; published 18 called against a true 16
  - round 8  bound the receivers, and missed every call routed through `codec(gen)`; published 8

Each fix addressed the instance. The class is: **a gate may only count a name it can qualify.**
`free_fns_called` matches `uniffi.whoop_ffi.<name>`, a namespace, so a namesake cannot reach it.
`WhoopCodec`'s methods have no namespace, so the answer is not a tighter matcher: the document NAMES
both halves, a receiver the extractor cannot resolve is reported rather than counted, and the two
historical matchers run below as controls so a regression is visible beside the current answer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
gate = __import__("docs-vs-code")

FAILED: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok     {name:<48} {got}")
    else:
        print(f"  FAILED {name:<48} got {got!r}, want {want!r}")
        FAILED.append(name)


# The real shape of RustCodec.kt: two codecs bound by name, reached both directly and through a
# selector, plus two namesakes that are NOT codecs.
HOLDER = """
import uniffi.whoop_ffi.WhoopCodec
object RustCodec {
    private val gen5 by lazy { WhoopCodec(Gen.GEN5) }
    private val gen4 by lazy { WhoopCodec(Gen.GEN4) }
    private fun codec(gen: Gen) = if (gen == Gen.GEN5) gen5 else gen4
    private val reassembler = Reassembler()
    fun history(gen: Gen, raw: ByteArray) = codec(gen).decodeHistory(raw)
    fun live(raw: ByteArray) = gen4.decodeLive(raw)
    fun buzz(seq: Int): ByteArray = gen5.buzzFrame(seq.toUByte())
    fun chunk(b: ByteArray) { for (f in reassembler.feed(b)) handle(f) }
    fun clear() { bondGiveUp.reset(); clockReference.reset() }
}
"""

# The same file as the caller spelled it one refactor earlier. The selector took a `Boolean`, so a
# rule that allowed the declaration's own parameters would pass here and fail on HOLDER above.
HOLDER_BOOLEAN = HOLDER.replace(
    "private fun codec(gen: Gen) = if (gen == Gen.GEN5) gen5 else gen4",
    "private fun codec(isGen5: Boolean) = if (isGen5) gen5 else gen4",
).replace("codec(gen)", "codec(true)")

METHODS = ["new", "decode_history", "decode_live", "buzz_frame", "feed", "reset", "offload_abort"]


def round7_called(names: list[str], src: str) -> set[str]:
    """The round-7 matcher: scope to files naming `WhoopCodec`, then match `.<name>(` on any receiver."""
    body = gate.strip_comments(src)
    hit = set(re.findall(r"\.\s*(\w+)\s*\(", body))
    return {n for n in names if gate.camel(n) in hit} | ({"new"} if "WhoopCodec(" in body else set())


def round8_called(names: list[str], src: str) -> set[str]:
    """The round-8 matcher: bind only identifiers CONSTRUCTED as a codec, with no accessor closure."""
    body = gate.strip_comments(src)
    bound = set(gate.CODEC_BOUND.findall(body))
    alt = "|".join(sorted(map(re.escape, bound)))
    hit = set(re.findall(rf"\b(?:{alt})\s*\.\s*(\w+)\s*\(", body)) if bound else set()
    return {n for n in names if gate.camel(n) in hit} | ({"new"} if "WhoopCodec(" in body else set())


def with_kotlin(src: str, fn):
    orig = gate.kotlin_files
    gate.kotlin_files = lambda: [src]
    try:
        return fn()
    finally:
        gate.kotlin_files = orig


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("receiver binding")
    check("binds both constructed codecs", gate.codec_receivers(HOLDER) >= {"gen5", "gen4"}, True)
    check("binds the selector that returns one", "codec" in gate.codec_receivers(HOLDER), True)
    check("binds it under the earlier Boolean spelling", "codec" in gate.codec_receivers(HOLDER_BOOLEAN), True)
    check("does not bind the namesake", "reassembler" in gate.codec_receivers(HOLDER), False)
    check("does not bind a builder that returns bytes", "buzz" in gate.codec_receivers(HOLDER), False)
    check("binds nothing in a file with no codec", gate.codec_receivers("val x = Reassembler()"), set())

    print("\ncall detection, the four regressions that shipped")
    called, unresolved, escaped = with_kotlin(HOLDER, lambda: gate.codec_calls(METHODS))
    check("counts the direct call, the routed call and new",
          called, {"new", "decode_history", "decode_live", "buzz_frame"})
    check("feed is not counted", "feed" in called, False)
    check("reset is not counted", "reset" in called, False)
    check("a method nobody calls is not counted", "offload_abort" in called, False)
    check("the two namesakes are REPORTED, not silently dropped",
          {n: sorted(v) for n, v in sorted(unresolved.items())},
          {"feed": ["reassembler.feed"], "reset": ["bondGiveUp.reset", "clockReference.reset"]})
    check("nothing escapes a file that keeps its codecs private", escaped, [])

    published = with_kotlin(HOLDER_BOOLEAN, lambda: gate.codec_calls(METHODS))[0]
    check("the earlier Boolean spelling gives the same answer", published, called)

    print("\nthe file-scope assumption, which is what lets a bare `.name(` be read at all")
    leaky = HOLDER.replace("private fun codec(gen: Gen)", "fun codec(gen: Gen)")
    check("a published selector is reported as escaping",
          with_kotlin(leaky, lambda: gate.codec_calls(METHODS))[2], ["codec"])
    # A block body is what the expression-body inference structurally cannot read, and it is exactly
    # where a codec would leave the file unseen: the consumer never names the type, so it is skipped
    # whole and its real calls are neither counted nor reported. Kotlin makes the annotation mandatory
    # there, so the return type is the one handle that cannot be omitted.
    check("a block-bodied public accessor is reported as escaping",
          gate.codec_escapes(HOLDER + "\nfun pick(gen: Gen): WhoopCodec { return codec(gen) }\n"), ["pick"])
    check("a block-bodied public getter is reported as escaping",
          gate.codec_escapes(HOLDER + "\nval current: WhoopCodec get() { return gen5 }\n"), ["current"])
    check("keeping either private is still no escape",
          gate.codec_escapes(HOLDER + "\nprivate fun pick(gen: Gen): WhoopCodec { return codec(gen) }\n"), [])
    check("a builder returning bytes off a codec is not an escape",
          gate.codec_escapes(HOLDER + "\nfun raw(s: Int): ByteArray { return gen5.buzzFrame(s) }\n"), [])
    check("prose mention constructs nothing", with_kotlin(
        "// WhoopCodec (decode history) is described here.\nval r = Reassembler()\nr.feed(b)",
        lambda: gate.codec_calls(["new", "feed"]))[0], set())

    print("\ncontrol: the two shipped matchers, on the same input")
    r7, r8 = round7_called(METHODS, HOLDER), round8_called(METHODS, HOLDER)
    for label, got in (("round 7  (any receiver)", r7), ("round 8  (constructed only)", r8), ("now", called)):
        print(f"    {label:<30} {sorted(got)}")
    check("round 7 counted both namesakes", sorted(r7 & {"feed", "reset"}), ["feed", "reset"])
    check("round 8 missed the routed call", "decode_history" in r8, False)
    check("now: neither namesake, and the routed call is found",
          (called & {"feed", "reset"}, "decode_history" in called), (set(), True))

    print("\nthe document's two lists")
    text = ("intro\n\n**Called from Kotlin** — `new`, `decode_history`,\n`decode_live`.\n\n"
            "**Not called** — `feed`, `reset`. Kept for the CLI.\n\nprose with `decode_history` in it\n")
    check("reads both halves, and stops at the paragraph",
          gate.codec_doc_lists(text), ({"new", "decode_history", "decode_live"}, {"feed", "reset"}))
    check("a deleted heading reads as empty, which the caller fails on",
          gate.codec_doc_lists(text.replace("**Not called**", "Not called")), ({"new", "decode_history", "decode_live"}, set()))

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
