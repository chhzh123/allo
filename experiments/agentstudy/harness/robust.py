p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()

# 1. A transient network read error killed a trial outright: IncompleteRead is
#    an HTTPException, which the retry did not catch. Retry on anything.
old = """            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:"""
new = """            except Exception as exc:  # noqa: BLE001 - any transport fault retries"""
assert s.count(old) == 1
s = s.replace(old, new)

# 2. Record why a turn ended and how much of it was reasoning, so a turn that
#    returns nothing usable is diagnosable rather than mysterious.
old = """        usage = body.get("usage") or {}
        spent = int(usage.get("total_tokens") or 0)
        self.tokens += spent
        self.seg_tokens += spent
        return body["choices"][0]["message"], usage"""
new = """        usage = body.get("usage") or {}
        spent = int(usage.get("total_tokens") or 0)
        self.tokens += spent
        self.seg_tokens += spent
        choice = body["choices"][0]
        usage = dict(usage)
        usage["finish_reason"] = choice.get("finish_reason")
        usage["reasoning_chars"] = len(str(choice["message"].get("reasoning") or ""))
        return choice["message"], usage"""
assert s.count(old) == 1
s = s.replace(old, new)

# 3. A model that answers with neither text nor a tool call has spent a turn
#    for nothing. One or two is noise; several in a row means the trial is
#    burning its budget without progressing, and it should stop and say so.
old = """        if not calls:
            messages.append({"role": "user", "content":
                             "Use write_file or run. Write SUBMIT alone on a line when done."})
            continue"""
new = """        if not calls:
            empty = empty + 1 if not text.strip() else 0
            if empty >= 3:
                trial.record("stalled", empty_turns=empty)
                break
            messages.append({"role": "user", "content":
                             "Use write_file or run. Write SUBMIT alone on a line when done."})
            continue
        empty = 0"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    messages = [{"role": "system", "content": system},"""
new = """    empty = 0
    messages = [{"role": "system", "content": system},"""
assert s.count(old) == 1
s = s.replace(old, new)

for anchor in ('"tokens" if trial.tokens >= args.tokens else',):
    assert s.count(anchor) == 2
s = s.replace('"tokens" if trial.tokens >= args.tokens else',
              '"empty_replies" if empty >= 3 else\n                             "tokens" if trial.tokens >= args.tokens else')
open(p, "w").write(s)
print("agent: retries any transport fault, records finish reason, stops after three empty turns")
