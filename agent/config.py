import os

# Groq's OpenAI-compatible chat model used by the planner, specialists,
# synthesizer, and critic. llama-3.3-70b-versatile is a strong, free-tier
# general-purpose model with solid tool-calling support.
DEFAULT_MODEL = os.environ.get("SEO_AGENT_MODEL", "llama-3.3-70b-versatile")
PLANNER_MODEL = os.environ.get("SEO_AGENT_PLANNER_MODEL", DEFAULT_MODEL)
CRITIC_MODEL = os.environ.get("SEO_AGENT_CRITIC_MODEL", DEFAULT_MODEL)

# Groq's per-model daily token quota (TPD) is tracked separately per model.
# If the primary model's quota is exhausted, agents automatically fall back
# to this smaller/faster model instead of failing outright, since it draws
# from a completely separate quota pool. Set to "" to disable fallback.
FALLBACK_MODEL = os.environ.get("SEO_AGENT_FALLBACK_MODEL", "llama-3.1-8b-instant")

# Groq's built-in agentic "Compound" system, used only by the competitive/
# benchmarking specialist. It performs live web search server-side, so no
# custom tool schema is needed (and Groq does not allow mixing custom tools
# with Compound systems at this time).
COMPETITIVE_MODEL = os.environ.get("SEO_AGENT_COMPETITIVE_MODEL", "groq/compound-mini")

# One or more Groq API keys, tried in order. Each key has its own separate
# daily quota, so a second key (e.g. from a second free Groq account) gives
# a fresh quota pool once the first is exhausted on both models. Comma-
# separate multiple keys: GROQ_API_KEYS=key1,key2. Falls back to the
# standard single GROQ_API_KEY if GROQ_API_KEYS isn't set.
_keys_env = os.environ.get("GROQ_API_KEYS")
if _keys_env:
    GROQ_API_KEYS = [k.strip() for k in _keys_env.split(",") if k.strip()]
else:
    _single_key = os.environ.get("GROQ_API_KEY")
    GROQ_API_KEYS = [_single_key] if _single_key else []

MAX_TOOL_ITERATIONS = int(os.environ.get("SEO_AGENT_MAX_ITER", 10))
MAX_REFLECTION_ROUNDS = int(os.environ.get("SEO_AGENT_MAX_REFLECTION_ROUNDS", 2))

# Groq's free tier shares one tokens-per-minute budget across all concurrent
# requests, so running many specialist agents at once easily triggers 429s.
# Keep this low by default; raise it if you're on a paid tier with more TPM.
MAX_PARALLEL_SPECIALISTS = int(os.environ.get("SEO_AGENT_MAX_WORKERS", 2))
SPECIALIST_DISPATCH_STAGGER_SECONDS = float(os.environ.get("SEO_AGENT_DISPATCH_STAGGER", 2.0))
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("SEO_AGENT_RATE_LIMIT_RETRIES", 4))

DB_PATH = os.environ.get("SEO_AGENT_DB_PATH", os.path.join(os.getcwd(), "data", "audit_history.db"))