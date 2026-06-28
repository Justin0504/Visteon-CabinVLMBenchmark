"""Central configuration for the Cabin-VLM exterior caption pipeline.
Everything portable comes from environment variables (with sensible defaults) so the
pipeline runs on any machine without code edits.

Required env:
  VULTR_KEYS   path to a file with lines `VULTR_KEY_1=...` (one or more keys, round-robined)
               OR a comma-separated list of keys directly in the variable.

Optional env (override models / concurrency / token budgets):
  VISION_MODEL        primary vision model        (default Qwen/Qwen3.5-397B-A17B)
  VISION_MODEL_FB     fast vision fallback        (default nvidia/Nemotron-...-Omni...)
  GEN_MODEL           text caption/QA generator   (default deepseek-ai/DeepSeek-V4-Flash)
  XCHECK_MODEL        cross-check fact-checker     (default deepseek-ai/DeepSeek-V3.2-NVFP4)
  WORKERS             concurrent requests         (default 4)
  VISION_MAX_TOKENS   primary vision token budget (default 2600; lower for reasoning fallbacks)
"""
import os

VULTR_URL = "https://api.vultrinference.com/v1/chat/completions"

# ---- API keys -------------------------------------------------------------
def load_keys():
    src = os.environ.get("VULTR_KEYS")
    if not src:
        raise SystemExit("Set VULTR_KEYS to a key-file path or a comma-separated list of Vultr API keys.")
    if os.path.isfile(src):
        keys = [l.split("=", 1)[1].strip() for l in open(src) if l.strip().startswith("VULTR_KEY")]
        if not keys:  # plain file, one key per line
            keys = [l.strip() for l in open(src) if l.strip() and not l.startswith("#")]
    else:
        keys = [k.strip() for k in src.split(",") if k.strip()]
    if not keys:
        raise SystemExit(f"No keys found in VULTR_KEYS={src}")
    return keys

# ---- models ---------------------------------------------------------------
VISION_MODEL    = os.environ.get("VISION_MODEL",    "Qwen/Qwen3.5-397B-A17B")
VISION_MODEL_FB = os.environ.get("VISION_MODEL_FB", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
GEN_MODEL       = os.environ.get("GEN_MODEL",       "deepseek-ai/DeepSeek-V4-Flash")
XCHECK_MODEL    = os.environ.get("XCHECK_MODEL",    "deepseek-ai/DeepSeek-V3.2-NVFP4")

# ---- runtime knobs --------------------------------------------------------
WORKERS          = int(os.environ.get("WORKERS", "4"))
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "2600"))
GEN_MAX_TOKENS    = int(os.environ.get("GEN_MAX_TOKENS", "1100"))
RETRIES          = int(os.environ.get("RETRIES", "3"))
