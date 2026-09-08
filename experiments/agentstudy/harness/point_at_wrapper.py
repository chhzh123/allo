import json
p = "/scratch/hc676/agentstudy/harness/arms.json"
arms = json.load(open(p))
arms["spmw"]["build"] = "bash /scratch/hc676/agentstudy/arms/spmw/build.sh {trial} visible"
json.dump(arms, open(p, "w"), indent=1)
print("arms.json: spmw goes through its own wrapper")

# And the builder should name an absent tool as an environment fault rather
# than let it read as a failed design.
p = "/scratch/hc676/agentstudy/arms/spmw/build.py"
s = open(p).read()
old = '''        except Exception as exc:  # noqa: BLE001 - the message is the product'''
new = '''        except FileNotFoundError as exc:
            print("STUDY STAGE %s %.2f" % (label, time.time() - start))
            print("STUDY BUILD ENVIRONMENT %s: %s" % (label, exc))
            sys.stdout.flush()
            sys.exit(2)
        except Exception as exc:  # noqa: BLE001 - the message is the product'''
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("spmw build: a missing tool is an environment fault, exit 2")

p = "/scratch/hc676/agentstudy/harness/grade_salvaged.sh"
s = open(p).read()
old = "    spmw) (cd /scratch/hc676/allo && timeout 3600 python3 -u \"$S/arms/spmw/build.py\" \"$T\" visible 2>&1)"
if old in s:
    s = s.replace(old, "    spmw) timeout 3600 bash \"$S/arms/spmw/build.sh\" \"$T\" visible 2>&1")
    open(p, "w").write(s)
    print("salvage runner: uses the wrapper too")
else:
    print("salvage runner: anchor not found, left alone")
