"""Pydantic output schemas (industrial-grade field construction + output validation).
Per the standard: every model output is validated against a typed schema, not hand-parsed from a string.
Use `schema_hint(Model)` to inject the field spec into a prompt, and `validate(Model, raw)` to parse +
validate the model's JSON reply (returns the typed object or None, never a half-broken dict)."""
from typing import List, Optional, Literal
import json
from pydantic import BaseModel, Field, field_validator, ValidationError


# ---------------- Stage B: caption + QA ----------------
class Caption(BaseModel):
    scene: str = Field(..., description="road type + weather/time/lane (from VISION) + each GT object w/ ego-relative pos & distance (from GT)")
    risk: str = Field("", description='causal hazard: "because <GT object> at <pos/dist> ..., it may <hazard>"')
    decision: str = Field("", description="ego action (proceed/slow/stop/yield/lane-change), justified by the cause")
    prediction: str = Field("", description="1-3s intent of the most safety-critical object, or 'none'")

    @field_validator("scene", "risk", "decision", "prediction", mode="before")
    @classmethod
    def _stringify(cls, v):
        # models sometimes return a nested dict for 'scene' — flatten to readable prose, never keep a dict
        if isinstance(v, dict):
            return ", ".join(f"{k}: {x}" for k, x in v.items() if x)
        return "" if v is None else str(v)


class QAItem(BaseModel):
    q: str
    a: str
    reason: str = ""
    perspective: str = ""
    capability: str = ""


class QASet(BaseModel):
    qa: List[QAItem] = []


# ---------------- per-use-case extraction ----------------
class TrafficLight(BaseModel):
    state: Literal["red", "yellow", "green", "off"]
    shape: str = "circle"
    position: str = ""
    for_lane: str = "unknown"


class POIItem(BaseModel):
    name: str
    category: str = ""
    position: str = ""

    @field_validator("name")
    @classmethod
    def _real_name(cls, v):
        if not v or str(v).strip().lower() in ("unknown", "sign", "store", "shop", ""):
            raise ValueError("placeholder name")
        return str(v).strip()


class OCRText(BaseModel):
    text: str
    bbox: List[int] = Field(..., min_length=4, max_length=4)  # [x1,y1,x2,y2] normalized 0-1000
    position: str = ""


class VRUCounts(BaseModel):
    pedestrians: int = 0
    cyclists: int = 0
    motorcyclists: int = 0


class SceneAttrs(BaseModel):
    weather: str = ""
    time_of_day: str = ""
    road_type: str = ""


class MultiExtract(BaseModel):
    scene: SceneAttrs = SceneAttrs()
    lights: List[TrafficLight] = []
    signs: List[dict] = []
    poi: List[POIItem] = []
    text: List[OCRText] = []
    vru: VRUCounts = VRUCounts()


# ---------------- helpers ----------------
def schema_hint(model: type[BaseModel]) -> str:
    """Compact JSON-schema hint to embed in a prompt (field names + types)."""
    props = model.model_json_schema().get("properties", {})
    def t(p):
        return p.get("type") or (p.get("$ref", "").split("/")[-1]) or "any"
    return "{" + ", ".join(f'"{k}": {t(v)}' for k, v in props.items()) + "}"


def validate(model: type[BaseModel], raw):
    """Parse a model reply (str or dict) and validate against the schema. Returns typed obj or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.replace("```json", "").replace("```", "").strip()
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0:
            return None
        try:
            raw = json.loads(raw[i:j + 1])
        except Exception:
            return None
    try:
        return model.model_validate(raw)
    except ValidationError:
        return None


def validate_list(model: type[BaseModel], items) -> list:
    """Validate a list of dicts, dropping invalid ones (e.g. POI placeholder names, bad bbox length)."""
    out = []
    for it in (items or []):
        try:
            out.append(model.model_validate(it))
        except ValidationError:
            continue
    return out
