"""
FER Service Status Tracker

Tracks recent requests and results for status monitoring.
Provides endpoints for health checks and service status.
"""

import logging
import datetime
from collections import deque
from typing import Dict, List, Optional
from threading import Lock
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Initialize router for status endpoints
router = APIRouter()


class StatusTracker:
    """Track recent requests and results for status monitoring."""
    
    def __init__(self, max_recent_requests: int = 100, max_recent_results: int = 100):
        self.recent_requests = deque(maxlen=max_recent_requests)
        self.recent_results = deque(maxlen=max_recent_results)
        self.lock = Lock()
    
    def log_request(self, user_id: str, timestamp: datetime.datetime, filename: Optional[str] = None):
        """Log an incoming request."""
        with self.lock:
            self.recent_requests.append({
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "filename": filename or "image.jpg",
                "status": "received"
            })
    
    def log_result(
        self,
        user_id: str,
        timestamp: datetime.datetime,
        emotion: str,
        confidence: float,
        db_write_success: bool = False,
        aggregation_complete: bool = False
    ):
        """Log a processing result."""
        with self.lock:
            self.recent_results.append({
                "user_id": user_id,
                "timestamp": timestamp.isoformat(),
                "emotion": emotion,
                "emotion_confidence": confidence,
                "db_write_success": db_write_success,
                "aggregation_complete": aggregation_complete,
                "status": "completed"
            })
    
    def get_recent_requests(self, limit: int = 20, minutes: int = 10) -> List[Dict]:
        """Get recent requests within the last N minutes."""
        with self.lock:
            cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
            recent = []
            for req in reversed(self.recent_requests):  # Start from newest
                try:
                    req_time = datetime.datetime.fromisoformat(req["timestamp"].replace('Z', '+00:00'))
                    if req_time >= cutoff_time:
                        recent.append(req)
                        if len(recent) >= limit:
                            break
                except Exception:
                    continue
            return recent
    
    def get_recent_results(self, limit: int = 20) -> List[Dict]:
        """Get recent results (newest first)."""
        with self.lock:
            return list(reversed(list(self.recent_results)[-limit:]))


# Global status tracker instance
status_tracker = StatusTracker()


@router.get("/health")
async def health():
    """Health check endpoint for Cloud Run."""
    try:
        return {
            "status": "healthy",
            "service": "fer",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "fer",
            "error": str(e),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }


@router.get("/fer/status")
async def get_fer_service_status():
    """
    Get detailed FER service status for cloud dashboard monitoring.
    
    Returns real-time information about:
    - Recent requests received
    - Processing results and database write status
    """
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        
        # Get recent requests (last 10 minutes)
        recent_requests = status_tracker.get_recent_requests(limit=20, minutes=10)
        
        # Get recent results
        recent_results = status_tracker.get_recent_results(limit=20)
        
        return {
            "service": "fer",
            "timestamp": now.isoformat(),
            "status": "healthy",
            "recent_requests": recent_requests[:10],  # Last 10 requests (most recent)
            "recent_results": recent_results[:10],  # Last 10 results (most recent)
            "uptime": "unknown"  # Could be enhanced with actual uptime tracking
        }
        
    except Exception as e:
        logger.error(f"Error getting FER service status: {e}", exc_info=True)
        return {
            "service": "fer",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "error",
            "error": str(e),
            "recent_requests": [],
            "recent_results": []
        }
