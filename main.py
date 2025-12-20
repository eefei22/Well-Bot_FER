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
from collections import Counter # Removed 'deque'

# --- 1. The Clean Buffer Logic ---
class EmotionBuffer:
    def __init__(self, aggregation_minutes: int = 5):
        self.buffers = {}
        self.aggregation_minutes = aggregation_minutes

    def add_entry(self, user_id: str, emotion: str):
        """Adds emotion to user's list. Returns Aggregated Result if window closed."""
        now = datetime.datetime.now(datetime.timezone.utc)
        
        if user_id not in self.buffers:
            self.buffers[user_id] = []
        
        self.buffers[user_id].append((now, emotion))
        
        return self._check_and_aggregate(user_id, now)

    def _check_and_aggregate(self, user_id, current_time):
        user_data = self.buffers[user_id]
        if not user_data:
            return None

        # Check time elapsed since the FIRST entry in the current batch
        first_entry_time = user_data[0][0]
        duration = (current_time - first_entry_time).total_seconds()
        
        if duration >= (self.aggregation_minutes * 60):
            emotions_list = [entry[1] for entry in user_data]
            
            # Get the most common emotion
            if emotions_list:
                most_common = Counter(emotions_list).most_common(1)[0][0]
                
                # Reset Buffer for this user (Start next 5 min window)
                self.buffers[user_id] = []
                
                logging.info(f"BUFFER: Aggregated {len(emotions_list)} items for {user_id} -> {most_common}")
                return most_common
            
        return None

# Initialize Global Buffer
buffer_manager = EmotionBuffer(aggregation_minutes=1)

# --- 2. FastAPI Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
app = FastAPI()

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
    if request_id:
        try:
            return str(uuid.UUID(request_id))
        except ValueError:
            logging.warning(f"Invalid UUID in request: {request_id}")
    
    # Fallback to env
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
    validated_id = get_validated_uuid(user_id)
    if not validated_id:
        raise HTTPException(status_code=400, detail="Valid UUID required")

    try:
        # Read Image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # --- MODEL RUNS HERE (Untouched) ---
        result = predict_emotion(image)
        current_emotion = result['emotion']
        server_message = "Processed. No aggregation yet."
        
        # Buffer Logic
        if current_emotion.lower() != "none":
            
            # A. Add to Memory
            aggregated_emotion = buffer_manager.add_entry(validated_id, current_emotion)
            
            # B. Check if Aggregation Happened
            if aggregated_emotion:
                server_message = f"AGGREGATION COMPLETE: Saved '{aggregated_emotion}' to DB."
                
                # Save to DB
                if supabase:
                    now = datetime.datetime.now(datetime.timezone.utc)
                    db_record = {
                        "user_id": validated_id,
                        "timestamp": now.isoformat(),
                        "predicted_emotion": aggregated_emotion,
                        "emotion_confidence": 1.0, 
                        "date": now.strftime("%Y-%m-%d")
                    }
                    try:
                        supabase.table("face_emotion").insert(db_record).execute()
                    except Exception as e:
                        logging.error(f"DB Error: {e}")

        # Attach the message to the response
        result["server_message"] = server_message
        
        return result

    except HTTPException:
        raise 
    except Exception as e:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail=str(e))