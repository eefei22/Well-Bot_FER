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
from status_tracker import status_tracker, router as status_router
# removed 'collections' because we use max() now, not Counter()

# --- 1. Buffer Logic (Max Confidence Strategy) ---
class EmotionBuffer:
    def __init__(self, aggregation_minutes: int = 5):
        self.buffers = {}
        self.aggregation_minutes = aggregation_minutes

    def add_entry(self, user_id: str, emotion: str, confidence: float):
        """
        Adds entry and returns (Aggregated_Emotion, Aggregated_Confidence) 
        if the time window has closed. Otherwise returns None.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        
        if user_id not in self.buffers:
            self.buffers[user_id] = []
        
        # Store tuple: (timestamp, emotion, confidence)
        self.buffers[user_id].append((now, emotion, confidence))
        
        return self._check_and_aggregate(user_id, now)

    def _check_and_aggregate(self, user_id, current_time):
        user_data = self.buffers[user_id]
        if not user_data:
            return None

        # Check time elapsed since the FIRST entry
        first_entry_time = user_data[0][0]
        duration = (current_time - first_entry_time).total_seconds()
        
        if duration >= (self.aggregation_minutes * 60):
            # --- AGGREGATION LOGIC START ---
            
            # 1. Filter out 'none' to find real emotions
            # item[1] is emotion, item[2] is confidence
            real_emotions = [item for item in user_data if item[1].lower() != 'none']
            
            final_emotion = "none"
            final_confidence = 0.0

            if real_emotions:
                # 2. Find the entry with the HIGHEST confidence score
                # key=lambda x: x[2] tells max() to look at the confidence value
                best_entry = max(real_emotions, key=lambda x: x[2])
                final_emotion = best_entry[1]
                final_confidence = best_entry[2]
            else:
                # If everything was 'none', we default to 'none' with 0.0 confidence
                final_emotion = "none"
                final_confidence = 0.0
            
            # --- AGGREGATION LOGIC END ---

            # Reset Buffer
            self.buffers[user_id] = []
            
            logging.info(f"BUFFER: Aggregated {len(user_data)} items. Winner: {final_emotion} ({final_confidence})")
            return (final_emotion, final_confidence)
            
        return None

# Initialize Global Buffer (Set to 5 mins for production)
buffer_manager = EmotionBuffer(aggregation_minutes=5)

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

# Include status tracking routes
app.include_router(status_router)

def get_validated_uuid(request_id: str) -> str:
    if request_id:
        try:
            return str(uuid.UUID(request_id))
        except ValueError:
            logging.warning(f"Invalid UUID in request: {request_id}")
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

    # Track request
    request_timestamp = datetime.datetime.now(datetime.timezone.utc)
    status_tracker.log_request(validated_id, request_timestamp, file.filename)

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # --- MODEL INFERENCE ---
        result = predict_emotion(image)
        current_emotion = result['emotion']
        # Ensure confidence is a float
        current_confidence = float(result.get('confidence', 0.0))
        
        server_message = "Processed. No aggregation yet."
        db_write_success = False
        aggregation_complete = False
        
        # --- BUFFER LOGIC (Always Run) ---
        if True:
            # Pass confidence into the buffer now
            aggregation_result = buffer_manager.add_entry(validated_id, current_emotion, current_confidence)
            
            # If aggregation_result is not None, it contains the tuple (emotion, score)
            if aggregation_result:
                agg_emotion, agg_conf = aggregation_result
                aggregation_complete = True
                
                server_message = f"AGGREGATION COMPLETE: Saved '{agg_emotion}' ({agg_conf:.2f}) to DB."
                
                if supabase:
                    now = datetime.datetime.now(datetime.timezone.utc)
                    db_record = {
                        "user_id": validated_id,
                        "timestamp": now.isoformat(),
                        "predicted_emotion": agg_emotion,
                        "emotion_confidence": agg_conf, # Saving the ACTUAL max confidence
                        "date": now.strftime("%Y-%m-%d")
                    }
                    try:
                        supabase.table("face_emotion").insert(db_record).execute()
                        db_write_success = True
                    except Exception as e:
                        logging.error(f"DB Error: {e}")
                        db_write_success = False

        # Track result
        result_timestamp = datetime.datetime.now(datetime.timezone.utc)
        status_tracker.log_result(
            validated_id,
            result_timestamp,
            current_emotion,
            current_confidence,
            db_write_success=db_write_success,
            aggregation_complete=aggregation_complete
        )

        result["server_message"] = server_message
        return result

    except HTTPException:
        raise 
    except Exception as e:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail=str(e))