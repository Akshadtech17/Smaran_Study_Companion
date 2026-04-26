import os
import tempfile
import json
import aiofiles
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
import docx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import requests

# Load env variables
load_dotenv()

# Gemini setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Rachel voice, you can change

# Path to Tesseract (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize FastAPI
app = FastAPI()

# Allow frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # update for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- File Text Extractors -------- #
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        safe = raw.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        texts.append(safe)
    return " ".join(texts)


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)


# Optional: show a preview of available models at startup (not a route)
def available_models_preview(limit=10):
    try:
        models = genai.list_models()
        return [m.name for m in models][:limit]
    except Exception as e:
        print("⚠️ Could not list models:", e)
        return []

print("Available Gemini/GenAI models preview:", available_models_preview())


# -------- API Route: Upload & Extract -------- #
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        print("📂 Uploaded file:", file.filename, "Suffix:", suffix)

        # Use TemporaryDirectory for automatic cleanup
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, file.filename)

            async with aiofiles.open(file_path, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)

            print("✅ File saved at:", file_path)

            if suffix == ".pdf":
                extracted_text = extract_text_from_pdf(file_path)
            elif suffix == ".docx":
                extracted_text = extract_text_from_docx(file_path)
            elif suffix == ".txt":
                extracted_text = extract_text_from_txt(file_path)
            elif suffix in [".jpg", ".jpeg", ".png"]:
                extracted_text = extract_text_from_image(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

            print("📝 Extracted text (preview):", extracted_text[:200])

            if not extracted_text.strip():
                raise HTTPException(status_code=400, detail="No readable text found in file")

            # Ensure safe return with proper Unicode handling
            safe_text = extracted_text.encode("utf-8", "replace").decode("utf-8")
            return JSONResponse(content={"text": safe_text})

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -------- API Route: Preprocess with Gemini -------- #
class TextPayload(BaseModel):
    text: str


@app.post("/preprocess/")
async def preprocess_text(payload: TextPayload):
    try:
        user_text = payload.text.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="No text provided")

        exam_prompt = """
You are an exam-focused AI tutor helping students revise.
Carefully read the provided notes line by line.

IMPORTANT: Use ONLY the notes provided below. 
DO NOT add content not present in the notes. 
Focus entirely on the concepts contained in the provided notes.

Your task:
1. Identify ONLY exam-relevant content:
   - Core concepts
   - Definitions
   - Key formulas
   - High-yield facts
2. Ignore:
   - Filler, long stories, or low-importance details
   - Redundant repetition
3. Transform important content into a clear lecture script that feels like a teacher explaining to students before an exam.

VERY IMPORTANT: The lecture script should be long enough so the total podcast duration is 20 to 30 minutes. 
Expand explanations with examples, comparisons, and short clarifications when needed to reach this duration.

Output format (STRICT JSON only, no extra text outside JSON):
[
  {
    "title": "Topic Name",
    "content": "Lecture-style explanation of exam-relevant points, written in a natural spoken style.",
    "importance": "high" | "medium",
    "duration": <estimated time in seconds to read aloud>
  }
]

Guidelines:
- Split into multiple sections (one section per concept/topic).
- Each section must feel like a mini lecture (not just bullet points).
- Keep "content" natural, like spoken teaching (e.g., “Now let’s look at…”).
- Ensure enough depth/detail to make the whole lecture 20–30 minutes when read aloud.
- Estimate duration realistically (average ~120 words ≈ 60 seconds).
- Distribute time across sections (some topics may need more, some less).
"""

        preferred_model = "models/gemini-1.5-flash"  # change if not available

        # defensive model listing
        try:
            model_names = [m.name for m in genai.list_models()]
        except Exception as e:
            print("⚠️ Warning: could not fetch model list:", e)
            model_names = []

        if preferred_model not in model_names:
            # try to fall back
            fallback = None
            for candidate in model_names:
                if "gemini" in candidate.lower() or "bison" in candidate.lower():
                    fallback = candidate
                    break

            if fallback:
                print(f"⚠️ Preferred model {preferred_model} not available, falling back to {fallback}")
                use_model = fallback
            else:
                preview = model_names[:10] if model_names else "no models returned"
                raise HTTPException(
                    status_code=500,
                    detail=f"Preferred model '{preferred_model}' not found. Available (preview): {preview}. "
                           "Try updating your SDK or check your API access."
                )
        else:
            use_model = preferred_model

        full_input = exam_prompt + "\n\nNotes:\n" + user_text
        response_text = None
        raw_response = None

        try:
            ModelClass = getattr(genai, "GenerativeModel", None)
            if ModelClass:
                print("Using GenerativeModel API path with model:", use_model)
                model_obj = ModelClass(use_model)
                raw_response = model_obj.generate_content(
                    full_input,
                    generation_config={"response_mime_type": "application/json"}
                )
                if hasattr(raw_response, "text"):
                    response_text = raw_response.text
                elif hasattr(raw_response, "content"):
                    response_text = raw_response.content
                elif isinstance(raw_response, dict) and "candidates" in raw_response:
                    response_text = raw_response["candidates"][0].get("content", "")
                else:
                    response_text = str(raw_response)
            else:
                print("Using genai.generate_text API path with model:", use_model)
                raw = genai.generate_text(model=use_model, prompt=full_input, max_output_tokens=1024)
                if hasattr(raw, "text"):
                    response_text = raw.text
                elif isinstance(raw, dict) and raw.get("candidates"):
                    response_text = raw["candidates"][0].get("content", "")
                else:
                    response_text = str(raw)
        except Exception as api_exc:
            print("❌ Gemini API call failed:", api_exc)
            raise HTTPException(status_code=500, detail=f"Gemini API call failed: {api_exc}")

        # Validate JSON safely
        try:
            segments = json.loads(response_text)
            if not isinstance(segments, list):
                raise ValueError("Expected a JSON list at top level")
        except Exception as parse_err:
            print("❌ JSON Parse Error. Raw response preview:", (response_text or "")[:1000])
            raise HTTPException(status_code=500, detail="Invalid JSON from Gemini or unexpected response shape")

        return JSONResponse(content={"status": "success", "segments": segments})

    except HTTPException:
        raise
    except Exception as e:
        print("❌ AI Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -------- API Route: Convert to Audio with ElevenLabs -------- #
class AudioPayload(BaseModel):
    text: str

@app.post("/generate-audio/")
async def generate_audio(payload: AudioPayload):
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ElevenLabs API key not set")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"

        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": ELEVENLABS_API_KEY,
        }

        body = {
            "text": payload.text,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.7
            }
        }

        # NOTE: requests is blocking. For production consider httpx.AsyncClient
        response = requests.post(url, headers=headers, json=body)

        if response.status_code != 200:
            print("❌ ElevenLabs Error:", response.text)
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # Save file
        output_path = os.path.join(tempfile.gettempdir(), "output.mp3")
        with open(output_path, "wb") as f:
            f.write(response.content)

        return FileResponse(output_path, media_type="audio/mpeg", filename="output.mp3")

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Audio Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
