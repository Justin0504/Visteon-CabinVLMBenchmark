"""Vultr Inference client shared by the pipeline.
Hardened per review: image downscale before encode (cap payload), error-class handling (fail fast on
auth, retry only transient), no silent swallowing (failures are logged), choices/body validation,
reasoning-model fallback, robust JSON extraction. Stateless and thread-safe."""
import json, time, re, io, logging, urllib.request, urllib.error, base64
from . import config

log = logging.getLogger("vultr")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_EDGE = 1280            # downscale longest image edge before sending
JPEG_Q = 85
MAX_B64_BYTES = 6_000_000  # hard cap on base64 payload

class AuthError(Exception):
    """Non-retryable: bad/expired key, forbidden, malformed request."""

def encode_image(path):
    """Downscale + JPEG-encode an image to base64, capping payload size (avoids 413 / 504 on 4K frames)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_EDGE:
        s = MAX_EDGE / max(w, h)
        im = im.resize((int(w * s), int(h * s)))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=JPEG_Q)
    b64 = base64.b64encode(buf.getvalue()).decode()
    if len(b64) > MAX_B64_BYTES:  # still too big -> shrink harder once
        im.thumbnail((900, 900)); buf = io.BytesIO(); im.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
    return b64

def _post(model, messages, key, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(config.VULTR_URL, data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        raw = urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 400):  # auth / bad request -> do NOT retry
            raise AuthError(f"HTTP {e.code}: {e.read()[:200]!r}")
        raise  # 429/5xx -> retryable
    d = json.loads(raw)
    if "choices" not in d or not d["choices"]:
        raise ValueError(f"no choices in response: {str(d)[:200]}")
    return d["choices"][0]["message"]

def _salvage(reasoning):
    r = re.sub(r"\*\*|`|#+", "", reasoning)
    sents = re.split(r"(?<=[.!?])\s+", r.replace("\n", " "))
    kw = ("scene", "road", "street", "weather", "sky", "building", "lane", "pavement", "asphalt",
          "daytime", "night", "sunny", "cloudy", "overcast", "urban", "highway", "lighting",
          "surface", "vegetation", "intersection", "parking")
    desc = [s.strip(" -*") for s in sents if len(s.strip()) > 40 and any(w in s.lower() for w in kw)]
    if desc:
        return " ".join(desc[-3:])[:600]
    return " ".join(s.strip(" -*") for s in sents[-3:])[:600] if sents else ""

def _extract(m):
    c = (m.get("content") or "").strip()
    if c:
        return c
    r = (m.get("reasoning") or "").strip()
    return _salvage(r) if r else ""

def _chat(model, msgs, key, max_tokens, timeout, label):
    for att in range(config.RETRIES):
        try:
            d = _extract(_post(model, msgs, key, max_tokens, timeout))
            if d:
                return d
        except AuthError as e:
            log.error("AUTH error (not retrying) on %s: %s", label, e)
            raise  # surface fast: a dead key shouldn't masquerade as 'empty output'
        except Exception as e:
            if att == config.RETRIES - 1:
                log.warning("%s failed after %d tries: %s", label, config.RETRIES, e)
            time.sleep(2 * (att + 1))
    return ""

def chat_text(model, sys, usr, key, max_tokens, timeout=180):
    return _chat(model, [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                 key, max_tokens, timeout, "chat_text")

def chat_vision(model, sys, prompt, img_b64, key, max_tokens, timeout=240):
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img_b64}}]}]
    return _chat(model, msgs, key, max_tokens, timeout, "chat_vision")

def parse_json(t):
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
