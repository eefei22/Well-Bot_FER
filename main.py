import os
import logging
import datetime
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fer_model import predict_emotion
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# --- Connection Setup ---
#testing
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_validated_uuid(request_id: str) -> str:
    """Helper to handle the hierarchy: Request Form -> Env Var -> Error."""
    # 1. Check Request ID
    if request_id:
        try:
            return str(uuid.UUID(request_id))
        except ValueError:
            logging.warning(f"Invalid UUID in request: {request_id}")

    # 2. Check Environment Fallback
    env_id = os.environ.get("DEV_USER_ID")
    if env_id:
        try:
            return str(uuid.UUID(env_id))
        except ValueError:
            logging.error(f"Invalid UUID in DEV_USER_ID env: {env_id}")

    return None

@app.post("/emotion")
async def detect_emotion(
    file: UploadFile = File(...), 
    user_id: str = Form(None) 
):
    # 1. Identity Validation
    validated_id = get_validated_uuid(user_id)
    if not validated_id:
        raise HTTPException(status_code=400, detail="Valid UUID required (from form or env)")

    try:
        # 2. Image Processing
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # 3. Model Inference
        result = predict_emotion(image)
        logging.info(f"User {validated_id} result: {result['emotion']}")

        # 4. Database Persistence
        if result["emotion"].lower() != "none" and supabase:
            # Use timezone-aware UTC
            now = datetime.datetime.now(datetime.timezone.utc)
            
            db_record = {
                "user_id": validated_id, # Fixed: using validated variable
                "timestamp": now.isoformat(),
                "predicted_emotion": result["emotion"],
                "emotion_confidence": float(result["confidence"]),
                "date": now.strftime("%Y-%m-%d")
            }

            try:
                supabase.table("face_emotion").insert(db_record).execute()
            except Exception as db_err:
                logging.error(f"Supabase error: {db_err}")
        
        return result

    except HTTPException:
        raise # Re-raise FastAPI-specific errors
    except Exception as e:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail=str(e))