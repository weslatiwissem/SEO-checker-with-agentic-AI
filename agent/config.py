import os

# Groq's OpenAI-compatible chat model used by the planner, specialists,
# synthesizer, and critic. llama-3.3-70b-versatile is a strong, free-tier
# general-purpose model with solid tool-calling support.
DEFAULT_MODEL = os.environ.get("SEO_AGENT_MODEL", "llama-3.3-70b-versatile")
PLANNER_MODEL = os.environ.get("SEO_AGENT_PLANNER_MODEL", DEFAULT_MODEL)
CRITIC_MODEL = os.environ.get("SEO_AGENT_CRITIC_MODEL", DEFAULT_MODEL)

# Groq's built-in agentic "Compound" system, used only by the competitive/
# benchmarking specialist. It performs live web search server-side, so no
# custom tool schema is needed (and Groq does not allow mixing custom tools
# with Compound systems at this time).
COMPETITIVE_MODEL = os.environ.get("SEO_AGENT_COMPETITIVE_MODEL", "groq/compound-mini")

MAX_TOOL_ITERATIONS = int(os.environ.get("SEO_AGENT_MAX_ITER", 10))
MAX_REFLECTION_ROUNDS = int(os.environ.get("SEO_AGENT_MAX_REFLECTION_ROUNDS", 2))

# Groq's free tier shares one tokens-per-minute budget across all concurrent
# requests, so running many specialist agents at once easily triggers 429s.
# Keep this low by default; raise it if you're on a paid tier with more TPM.
MAX_PARALLEL_SPECIALISTS = int(os.environ.get("SEO_AGENT_MAX_WORKERS", 2))
SPECIALIST_DISPATCH_STAGGER_SECONDS = float(os.environ.get("SEO_AGENT_DISPATCH_STAGGER", 2.0))
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("SEO_AGENT_RATE_LIMIT_RETRIES", 4))

DB_PATH = os.environ.get("SEO_AGENT_DB_PATH", os.path.join(os.getcwd(), "data", "audit_history.db"))
