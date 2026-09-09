p = "/scratch/hc676/agentstudy/arms/spmw/build.py"
s = open(p).read()

old = """    def stage_time(label, fn):
        start = time.time()
        value = fn()
        print("STUDY STAGE %s %.2f" % (label, time.time() - start))
        return value
"""
new = '''    def stage_time(label, fn):
        """Run one stage, time it, and stop at the first one that fails.

        The compiler checks structure before it generates anything, so a design
        with an unbound port or a mismatched link is rejected in seconds. That
        error is worth reporting on its own terms rather than as a traceback
        after a synthesis run that was never going to happen.
        """
        start = time.time()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - the message is the product
            seconds = time.time() - start
            print("STUDY STAGE %s %.2f" % (label, seconds))
            print("STUDY BUILD FAIL %s" % label)
            print("%s: %s" % (type(exc).__name__, exc))
            sys.stdout.flush()
            sys.exit(1)
        print("STUDY STAGE %s %.2f" % (label, time.time() - start))
        return value
'''
assert s.count(old) == 1
s = s.replace(old, new)

old = """    fabric = load(path)"""
new = """    fabric = stage_time("import", lambda: load(path))"""
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    sba._write(os.path.join(sim, "dut_norm.sv"), wrapper(graph))'''
new = '''    sba._write(os.path.join(sim, "dut_norm.sv"),
               stage_time("wrap", lambda: wrapper(graph)))'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("spmw arm: every stage fails fast, with the compiler's own message")

# The wrapper reports contract violations by exiting; make those a build
# failure the agent can read rather than a bare exit.
s = open(p).read()
s = s.replace('sys.exit(f"STUDY BUILD FAIL boundary', 'sys.exit(f"STUDY BUILD FAIL wrap: boundary')
s = s.replace('sys.exit(f"STUDY BUILD FAIL `{name}` channel', 'sys.exit(f"STUDY BUILD FAIL wrap: `{name}` channel')
s = s.replace('sys.exit(f"STUDY BUILD FAIL no boundary channel', 'sys.exit(f"STUDY BUILD FAIL wrap: no boundary channel')
open(p, "w").write(s)

# And the salvage runner needs the same environment a trial gets.
p = "/scratch/hc676/agentstudy/harness/grade_salvaged.sh"
s = open(p).read()
old = "source /work/shared/common/allo/vitis_2023.2_u280.sh >/dev/null 2>&1"
new = ("export PATH=/scratch/hc676/allo-agent/bin:$PATH "
       "LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build\n" + old)
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("salvage runner: the same environment a trial gets")
