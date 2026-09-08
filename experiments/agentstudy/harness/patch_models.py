p = "/scratch/hc676/agentstudy/harness/run_all.sh"
s = open(p).read()
old = 'MODELS="anthropic/claude-opus-5 openai/gpt-5.6-sol moonshotai/kimi-k3 z-ai/glm-5.3 deepseek/deepseek-v4-pro"'
new = ('MODELS=${MODELS:-"anthropic/claude-opus-5 openai/gpt-5.6-sol moonshotai/kimi-k3 '
       'z-ai/glm-5.3 deepseek/deepseek-v4-pro"}')
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("run_all: model list is overridable")
