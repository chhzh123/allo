import json, os, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/models",
    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
with urllib.request.urlopen(req, timeout=60) as r:
    models = {m["id"]: m for m in json.load(r)["data"]}
for slug in ("z-ai/glm-5.3", "moonshotai/kimi-k3", "deepseek/deepseek-v4-pro",
             "anthropic/claude-opus-5", "openai/gpt-5.6-sol"):
    m = models.get(slug) or {}
    tp = m.get("top_provider") or {}
    params = m.get("supported_parameters") or []
    print("%-28s ctx=%-9s max_completion=%-9s reasoning_param=%s" % (
        slug, m.get("context_length"), tp.get("max_completion_tokens"),
        [p for p in params if "reason" in p or p == "max_tokens"]))
