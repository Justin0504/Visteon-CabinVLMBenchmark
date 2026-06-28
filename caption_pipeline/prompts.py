"""Unified prompt module for the Cabin-VLM exterior caption pipeline.

ONE place for every prompt, so caption / QA / cross-check stay consistent and reviewable.
Design = fusion (vision detail + ground-truth anchoring + frontier causal reasoning + cross-check)
         x IntelliCockpitBench rigor (driver persona, must-be-multimodal, no "in the image",
           answer carries a reason, perspective + capability diversity, reject-if-unanswerable).

Two layers feed the text models:
  GT (authoritative): object class / count / distance / bbox from sensor labels — trust for objects.
  VISION (Qwen3.5-397B et al.): weather, time, road type, lane markings, surface, buildings — trust for context.
The rule "objects from GT, context from VISION, never invent" is what keeps captions hallucination-free.
"""

# ---------------------------------------------------------------- Stage A: vision description
VISION_SYS = "You describe ONLY the visual context of an exterior driving scene. You never count or list dynamic objects."

VISION_PROMPT = (
    "Describe the VISUAL CONTEXT of this exterior driving/street image in 2-3 sentences: weather, time of day, "
    "road/setting type (urban street / highway / intersection / parking / rural), lane-marking type "
    "(dashed / solid / double-yellow / none), road-surface condition (dry / wet / snow), lighting, and roadside "
    "buildings / vegetation / terrain. Do NOT count or list vehicles or pedestrians — only the scene context. "
    "Output ONLY the final description, no preamble."
)

# ---------------------------------------------------------------- Stage B: causal caption
CAPTION_SYS = (
    "You are the reasoning module of an autonomous-driving cockpit. You receive (1) authoritative GROUND-TRUTH "
    "objects (sensor truth: class / count / distance / bbox) and (2) a VISION description of the same image. "
    "Rule: for objects / counts / positions, trust GROUND-TRUTH only. For visual context not in GT (weather, "
    "time-of-day, road type, lane markings, surface, lighting), you MAY use the VISION description. "
    "Never invent anything. Output only JSON."
)

CAPTION_SCHEMA = (
    'Use CHAIN-OF-CAUSATION. Output STRICT JSON {"scene","risk","decision","prediction"}: '
    'scene = road type + weather/time/lane (from VISION) + each GT object with ego-relative position & distance (from GT); '
    'risk = causal hazard, phrased "because <GT object> at <pos/dist> and <vision context>, it may <hazard>"; '
    'decision = therefore the ego action (proceed / slow / stop / yield / lane-change), justified by that cause; '
    'prediction = 1-3s intent of the most safety-critical object, or "none".'
)

# ---------------------------------------------------------------- Stage B: diverse QA (IntelliCockpitBench rigor)
PERSPECTIVES = ["Why", "What", "Where", "When", "Who/Which", "How", "How-many", "Is/Can/Do"]
CAPABILITIES = ["counting", "spatial", "distance", "recognition", "risk", "action", "intent", "weather/scene", "reject"]

QA_SYS = (
    "You are a DRIVER operating a vehicle, speaking about what you see THROUGH the windshield right now. "
    "You build question-answer pairs for an automotive multimodal (VLM) training set. Output only JSON."
)

def qa_schema(n=4):
    persp = ", ".join(PERSPECTIVES)
    caps = ", ".join(CAPABILITIES)
    return (
        f"Generate exactly {n} DIVERSE QA grounded in the GROUND-TRUTH objects (+ VISION context). RULES:\n"
        f"1. Each question must refer to concrete, perceivable content in THIS scene.\n"
        f"2. Must require MULTIMODAL perception — NOT answerable by an LLM alone or by a maps/weather/nav app.\n"
        f'3. Do NOT use phrases like "in the image" / "in the background" — speak as a driver.\n'
        f"4. Vary perspective across: {persp}.\n"
        f"5. Cover >=4 distinct capabilities from: {caps}.\n"
        f'6. Include exactly ONE "reject" item that asks about something NOT supported by GT/VISION; its answer '
        f'   must be "not visible" (teaches the model to refuse hallucination).\n'
        f"7. Every answer is grounded and concise, and carries a short reason.\n"
        f'Output STRICT JSON: {{"qa":[{{"q":"","a":"","reason":"","perspective":"","capability":""}}]}}'
    )

# ---------------------------------------------------------------- Stage B: cross-check
XCHECK_SYS = "You are a strict fact-checker. Remove any claim that contradicts the GROUND-TRUTH objects (counts/classes/positions). Output only JSON."

def xcheck_schema():
    return ('Remove claims contradicting GT object counts/classes/positions. Keep everything consistent. '
            'STRICT JSON {"corrected":{"scene","risk","decision","prediction"}}')
