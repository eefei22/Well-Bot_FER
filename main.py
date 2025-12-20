import os
import logging
import datetime
import numpy as np
import cv2
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fer_model import predict_emotion
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ---------------------------------------------------------
# 1. SETUP SUPABASE CONNECTION
# ---------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logging.warning("Supabase keys not found! Database save will fail.")
    supabase = None

# ---------------------------------------------------------
# 2. CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/emotion")
async def detect_emotion(
    file: UploadFile = File(...), 
    user_id: str = Form(...)  
):
    try:
        # This checks if the string is a valid UUID
        uuid_obj = uuid.UUID(user_id)
        # Re-assign to string to ensure it's in a standardized format
        user_id = str(uuid_obj)
    except ValueError:
        logging.warning(f"Invalid UUID format received: {user_id}")
        return JSONResponse(
            content={"error": "Invalid user_id format. Must be a valid UUID string."}, 
            status_code=400
        )
    
    try:
        logging.info(f"Received file from user: {user_id}")
        contents = await file.read()

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(content={"error": "Could not decode image"}, status_code=400)

        # ---------------------------------------------------------
        # 3. RUN FER MODEL
        # ---------------------------------------------------------
        try:
            result = predict_emotion(image)
        except Exception as pe:
            logging.exception("Error inside predict_emotion()")
            return JSONResponse(content={"error": f"Prediction failed: {str(pe)}"}, status_code=500)

        logging.info(f"Prediction result: {result}")

        # ---------------------------------------------------------
        # 4. SAVE TO SUPABASE (Updated for your Schema)
        # ---------------------------------------------------------
        if result["emotion"].lower() != "none" and supabase:
            
            # Get current time
            now = datetime.datetime.utcnow()
            
            # Map data to your exact table columns
            db_record = {
                # Use the user_id from the request body instead of the environment variable
                "user_id": user_id, 
                "timestamp": now.isoformat(),
                "predicted_emotion": result["emotion"],
                "emotion_confidence": float(result["confidence"]),
                "date": now.strftime("%Y-%m-%d")
            }

            try:
                supabase.table("face_emotion").insert(db_record).execute()
                logging.info(f"Logged prediction to Supabase for user {user_id}.")
            except Exception as db_err:
                logging.error(f"Failed to save to Supabase: {db_err}")
        
        return result

    except Exception as e:
        logging.exception("Exception in /emotion")
        return JSONResponse(content={"error": str(e)}, status_code=500)