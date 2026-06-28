"""Thin Vultr Inference client shared by both pipeline stages.
Handles: OpenAI-compatible chat, retry/backoff, reasoning-model fallback (content empty ->
salvage from the `reasoning` field), and robust JSON extraction. Stateless and thread-safe."""
import json, time, re, urllib.request
from . import config

def _post(model, messages, key, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(config.VULTR_URL, data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]

def _salvage(reasoning):
    """Reasoning models sometimes spend the whole budget thinking and leave `content` empty.
    Pull the descriptive sentences out of the reasoning trace as a fallback."""
    r = re.sub(r"\*\*|`|#+", "", reasoning)
    sents = re.split(r"(?<=[.!?])\s+", r.replace("\n", " "))
    kw = ("scene", "road", "street", "weather", "sky", "building", "lane", "pavement", "asphalt",
          "daytime", "night", "sunny", "cloudy", "overcast", "urban", "highway", "lighting",
          "surface", "vegetation", "intersection", "parking")
    desc = [s.strip(" -*") for s in sents if len(s.strip()) > 40 and any(w in s.lower() for w in kw)]
    if desc:
        return " ".join(desc[-3:])[:600]
    return " ".join(s.strip(" -*") for s in sents[-3:])[:600] if sents else ""

def chat_text(model, sys, usr, key, max_tokens, timeout=180):
    """Return assistant text (content, or salvaged reasoning). '' on failure."""
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
    for att in range(config.RETRIES):
        try:
            m = _post(model, msgs, key, max_tokens, timeout)
            c = (m.get("content") or "").strip()
            if c:
                return c
            r = (m.get("reasoning") or "").strip()
            if r:
                d = _salvage(r)
                if d:
                    return d
        except Exception:
            time.sleep(2 * (att + 1))
    return ""

def chat_vision(model, sys, prompt, img_b64, key, max_tokens, timeout=240):
    """Vision call with a base64 JPEG. Return description text. '' on failure."""
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}]}]
    for att in range(config.RETRIES):
        try:
            m = _post(model, msgs, key, max_tokens, timeout)
            c = (m.get("content") or "").strip()
            if c:
                return c
            r = (m.get("reasoning") or "").strip()
            if r:
                d = _salvage(r)
                if d:
                    return d
        except Exception:
            time.sleep(2 * (att + 1))
    return ""

def parse_json(t):
    """Tolerant JSON extraction from a model reply (strips code fences, grabs outermost braces)."""
    if not t:
        return None
    t = t.replace("```json", "").replace("```", "").strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None
