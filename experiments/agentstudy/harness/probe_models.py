import json, os, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/models",
    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
with urllib.request.urlopen(req, timeout=60) as r:
    models = {m["id"]: m for m in json.load(r)["data"]}
print("--- slugs containing 5.6 or sol:")
for i in sorted(models):
    if "5.6" in i or "sol" in i.lower():
        print("   ", i)
print("--- candidates:")
for slug in ("anthropic/claude-opus-5", "moonshotai/kimi-k3", "z-ai/glm-5.3",
             "deepseek/deepseek-v4-pro", "openai/gpt-5.1", "openai/gpt-5-pro"):
    m = models.get(slug)
    if not m:
        print("    %-28s NOT FOUND" % slug); continue
    params = m.get("supported_parameters") or []
    price = m.get("pricing") or {}
    pin = float(price.get("prompt") or 0) * 1e6
    pout = float(price.get("completion") or 0) * 1e6
    print("    %-28s tools=%-5s ctx=%-8s in=$%.2f/M out=$%.2f/M"
          % (slug, "tools" in params, m.get("context_length"), pin, pout))
