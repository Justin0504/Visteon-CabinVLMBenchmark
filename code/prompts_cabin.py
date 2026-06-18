"""High-fidelity prompts adapted from IntelliCockpitBench (Lane315/IntelliCockpitBench)
paradigm -> Cabin VLM Benchmark exterior (out-of-vehicle) use cases.
Faithful to: driver persona, 10 question perspectives, 5x19 taxonomy, must-be-multimodal,
reject-if-unanswerable, answer carries primary/secondary tag + reason."""

PERSPECTIVES = ["Why","What","Where","When","Who/Which","How","How much/How many","How feel","Can/Have","Is/Do/Others"]

# Exterior-focused subset of the IntelliCockpitBench taxonomy + our 8 Category-2 use cases
TAXONOMY = {
 "Description": "describe the exterior driving scene",
 "Recognition/Vehicle Make&Model": "e.g. What is the make/model of the car ahead?",
 "Recognition/Object Recognition": "e.g. What is on the road on the right?",
 "Recognition/Information Extraction(Text/Ad/OCR)": "e.g. What does the billboard on the right say?",
 "Recognition/Pedestrian&VRU": "e.g. Is there a cyclist crossing ahead?",
 "WorldKnowledge/Traffic Sign&Law": "e.g. What does the sign ahead mean? Can I turn left here?",
 "WorldKnowledge/Building Landmark POI": "e.g. What building is on the left? What is this landmark?",
 "WorldKnowledge/Geospatial&Scene": "e.g. Is this a commercial or residential area?",
 "Reasoning/Quantitative": "e.g. How many lanes are ahead?",
 "Reasoning/Distance": "e.g. How far is the pedestrian from my car?",
 "Reasoning/Intent": "e.g. What is that person standing in the road trying to do?",
 "Reasoning/Driving Decision": "e.g. Given the road ahead, how should I proceed safely?",
}

def caption_prompt():
    return ("You are an in-car intelligent agent looking OUTSIDE the vehicle via the front camera. "
            "Describe the exterior driving scene in ONE clear, factual sentence: road type, traffic, "
            "pedestrians/vehicles, weather, notable buildings/signs. Output ONLY the sentence.")

def qa_prompt(n=5):
    cats = "\n".join(f"   - {k}: {v}" for k,v in TAXONOMY.items())
    persp = ", ".join(PERSPECTIVES)
    return f"""You are a DRIVER operating a vehicle. Based ONLY on what the onboard front camera sees,
generate exactly {n} question-answer pairs for building an automotive multimodal (VLM) training set.

RULES (from the IntelliCockpitBench paradigm):
1. Clarity: each question must refer to concrete, perceivable content in THIS image.
2. Must require MULTIMODAL capability — NOT answerable by an LLM alone or by maps/weather/navigation apps.
3. Do NOT use phrases like "in the image"/"in the background" (not how a driver speaks). If a question
   cannot be answered from the image, do not ask it.
4. Diverse perspectives — vary across: {persp}.
5. Coverage: across the {n} pairs include >=2 Recognition, >=2 Reasoning, >=1 WorldKnowledge.
6. Each answer must be grounded in the image and concise; also give a short 'reason'.

Pick {n} aspects from this exterior classification system (use the exact key as 'category'):
{cats}

Output STRICT JSON only:
{{"qa":[{{"question":"","answer":"","reason":"","perspective":"","category":""}}]}}
Output ONLY the JSON, no extra text."""
