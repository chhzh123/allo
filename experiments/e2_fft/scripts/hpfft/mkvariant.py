#!/usr/bin/env python3
"""Derive a variant of FFT.cpp by applying named pragma / storage-class edits.

Every edit asserts that it matched something, so a silent no-op is impossible:
a variant that failed to apply raises rather than quietly building the baseline.
"""
import sys, re, os, shutil


def rev_fn_span(c):
    m = re.search(r"^void reverse_input_stream_UF\d+ ?\(", c, re.M)
    assert m, "reverse_input_stream_UFx not found"
    e = re.compile(r"^\}", re.M).search(c, m.end())
    assert e, "closing brace of reverse stage not found"
    return m.start(), e.end()


def static_arrays(c):
    """A1: give the reverse stage's two scratch arrays static storage, so the
    complex<float> default constructor stops emitting per-call init loops."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    # Only the scratch buffers declared at function scope, i.e. ahead of the first
    # labelled loop. Arrays declared inside a loop body (block_data, cyclic_data)
    # are rewritten every iteration and must stay automatic.
    first_loop = re.search(r"^[ \t]*\w+: for ", body, re.M)
    cut = first_loop.start() if first_loop else len(body)
    head, tail = body[:cut], body[cut:]
    out, n = [], 0
    for line in head.splitlines(True):
        if re.match(r"^[ \t]*complex<float> \w+\[", line):   # skips // comments
            line = re.sub(r"^([ \t]*)complex<float>", r"\1static complex<float>", line)
            n += 1
        out.append(line)
    assert n >= 1, "no non-static complex<float> array declarations in reverse stage"
    return c[:s] + "".join(out) + tail + c[e:], n


def dataflow_pipo(c):
    """A2: make the reverse stage a dataflow region with pipo channels, so its
    loops overlap instead of running back to back."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    # anchor on the TIME_STEP declaration: present in every UF variant and not
    # removed by any other edit, so edit order does not matter
    a = re.search(r"^([ \t]*)const int TIME_STEP", body, re.M)
    assert a, "TIME_STEP anchor not found"
    ind = a.group(1)
    ins = (ind + "#pragma HLS stream type=pipo variable=data_rev_stream\n" +
           ind + "#pragma HLS stream type=pipo variable=data_in_cyclic\n" +
           ind + "#pragma HLS dataflow disable_start_propagation\n")
    body = body[:a.start()] + ins + body[a.start():]
    return c[:s] + body + c[e:], 1


def pipeline_ii1(c):
    """A3: pin II=1 on the reverse stage's pipelines rather than letting the
    scheduler pick the interval."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    body, k = re.subn(r"^([ \t]*)#pragma HLS pipeline[ \t]*$",
                      r"\1#pragma HLS pipeline II=1", body, flags=re.M)
    assert k >= 1, "no bare pipeline pragma in reverse stage"
    return c[:s] + body + c[e:], k


def no_cyclic_dim2(c):
    """A5: drop the cyclic dim=2 partition on data_in_cyclic. After the complete
    dim=1 partition each lane is already its own memory taking one write and one
    read per cycle, so the extra banking buys nothing -- and a channel array that
    is banked this finely cannot become a dataflow PIPO."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    body, k = re.subn(r"^[ \t]*#pragma HLS array_partition variable=data_in_cyclic"
                      r" type=cyclic factor=UF dim=2[ \t]*\n", "", body, flags=re.M)
    assert k == 1, "expected 1 cyclic dim=2 partition, got %d" % k
    return c[:s] + body + c[e:], k


def no_fabric_binding(c):
    """C1: stop forcing the butterfly's float add/sub into fabric, so the tool can
    pick a lower-latency implementation. Same IEEE operations either way."""
    c, n = re.subn(r"^[ \t]*#pragma HLS bind_op variable=\w+ op=f(add|sub) impl=fabric[ \t]*\n",
                   "", c, flags=re.M)
    assert n >= 6, "expected the fabric bind_op pragmas, got %d" % n
    return c, n


def no_complete_rev(c):
    """A7: drop the complete dim=1 partition on data_rev_stream. The lane index is
    a constant at every access, so Vitis splits the array anyway -- and an array
    carrying an explicit partition cannot become a dataflow PIPO channel."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    body, k = re.subn(r"^[ \t]*#pragma HLS array_partition variable=data_rev_stream"
                      r" type=complete dim=1[ \t]*\n", "", body, flags=re.M)
    assert k == 1, "expected 1 complete dim=1 partition on data_rev_stream, got %d" % k
    return c[:s] + body + c[e:], k


def fn_pipeline(c):
    """A8: pipeline the reverse stage as a whole function at II = beats-per-transform.
    This is the transformation Vitis picks by itself for reverse_input_stream_UF8,
    where it yields interval == trip count; here we ask for it explicitly."""
    s, e = rev_fn_span(c)
    body = c[s:e]
    # drop the per-loop pipeline pragmas: function pipelining unrolls the loops
    body, dropped = re.subn(r"^[ \t]*#pragma HLS pipeline([ \t]+II=\d+)?[ \t]*\n", "",
                            body, flags=re.M)
    assert dropped >= 1, "no loop pipeline pragmas found in reverse stage"
    o = body.index("{")
    ins = "\n    #pragma HLS pipeline II=FFT_NUM/(2*UF)\n"
    body = body[:o + 1] + ins + body[o + 1:]
    return c[:s] + body + c[e:], dropped


def overunroll(c, k):
    """D: widen the butterfly unroll from UF to UF*k lanes, leaving the streaming
    interface at UF*2 samples per beat. Each stage's interval is trip + iteration
    latency; over-unrolling cuts the trip count so the stage drops below the
    interval the interface itself imposes. Pragma-only, but it costs area."""
    k = int(k)
    n = 0
    c, a = re.subn(r"^([ \t]*)#pragma HLS unroll factor=UF[ \t]*$",
                   r"\1#pragma HLS unroll factor=UF*%d" % k, c, flags=re.M)
    n += a
    c, b = re.subn(r"^([ \t]*)#pragma HLS unroll factor=UF>>\(stage-1\)[ \t]*$",
                   r"\1#pragma HLS unroll factor=(UF*%d)>>(stage-1)" % k, c, flags=re.M)
    n += b
    assert a >= 1 and b >= 1, "unroll pragmas not found (factor=UF x%d, shifted x%d)" % (a, b)
    return c, n


def repart_all(c, spec):
    """D-companion: rescale every inter-stage buffer's partition at once. An
    over-unrolled butterfly needs proportionally more banks or its II rises to
    compensate, which cancels the gain exactly."""
    c, n = re.subn(r"^([ \t]*)#pragma HLS array_partition variable=(data_\d+)"
                   r" type=\S+(?: factor=\S+)? dim=1[ \t]*$",
                   r"\1#pragma HLS array_partition variable=\2 type=" + spec + " dim=1",
                   c, flags=re.M)
    assert n >= 6, "expected the data_N partitions, got %d" % n
    return c, n


def repart(c, var, spec):
    """B: replace one inter-stage buffer's array_partition with `spec`
    (e.g. "cyclic factor=UF", "complete"). Targets a single named array so the
    effect is attributable to it."""
    pat = (r"^([ \t]*)#pragma HLS array_partition variable=%s"
           r" type=\S+(?: factor=\S+)? dim=1[ \t]*$" % re.escape(var))
    c, n = re.subn(pat, r"\1#pragma HLS array_partition variable=%s type=%s dim=1"
                   % (var, spec), c, flags=re.M)
    assert n == 1, "expected 1 partition pragma for %s, got %d" % (var, n)
    return c, n


def bindstore(c, var, spec):
    """B: (re)bind one inter-stage buffer's memory core, e.g. RAM_T2P impl=bram.
    Replaces an existing bind_storage for that array if there is one."""
    pat = r"^[ \t]*#pragma HLS bind_storage variable=%s [^\n]*\n" % re.escape(var)
    c, dropped = re.subn(pat, "", c, flags=re.M)
    ap = re.search(r"^([ \t]*)#pragma HLS array_partition variable=%s[^\n]*\n"
                   % re.escape(var), c, re.M)
    assert ap, "no array_partition for %s to anchor bind_storage on" % var
    ins = "%s#pragma HLS bind_storage variable=%s type=%s\n" % (ap.group(1), var, spec)
    return c[:ap.end()] + ins + c[ap.end():], 1 + dropped


EDITS = {
    "static": static_arrays,
    "dataflow": dataflow_pipo,
    "ii1": pipeline_ii1,
    "nocyclic2": no_cyclic_dim2,
    "nofabric": no_fabric_binding,
    "nocomplete_rev": no_complete_rev,
    "fnpipe": fn_pipeline,
    "overunroll": overunroll,
    "repart_all": repart_all,
    "repart": repart,
    "bindstore": bindstore,
}

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    names = sys.argv[3:]
    c = open(os.path.join(src, "FFT.cpp")).read()
    for nm in names:
        # "repart:data_2:complete" -> repart(c, "data_2", "complete")
        parts = nm.split(":")
        c, k = EDITS[parts[0]](c, *parts[1:])
        print("  applied %s (%d site(s))" % (nm, k))
    os.makedirs(dst, exist_ok=True)
    for f in ("FFT.h", "project.tcl", "testbench.cpp"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    open(os.path.join(dst, "FFT.cpp"), "w").write(c)
    print("  wrote %s/FFT.cpp" % dst)
