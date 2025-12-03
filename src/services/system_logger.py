import cv2
import base64
import logging
import numpy as np
from datetime import datetime
from typing import Optional

from src.services.remote_logger import RemoteLogWorker
from src.infrastructure.data.repository import UnifiedRepository
from src.infrastructure.data.models import DrowsinessEvent

UNKNOWN_USER_ID = 0

class SystemLogger:
    """
    The Coordinator. Handles Buzzer, Local DB, and Remote Push.
    """
    def __init__(
        self, 
        buzzer=None, 
        remote_worker: Optional[RemoteLogWorker] = None, 
        event_repo: Optional[UnifiedRepository] = None,
        vehicle_vin: str = "VIN-0001",
        local_quality: int = 85,
        remote_quality: int = 70
    ):
        self.buzzer = buzzer
        self.remote = remote_worker
        self.repo = event_repo
        self.vehicle_vin = vehicle_vin
        self.local_quality = local_quality
        self.remote_quality = remote_quality

    def log_event(self, user_id: int, event_type: str, duration: float = 0.0, 
                  value: float = 0.0, frame: Optional[np.ndarray] = None):
        timestamp = datetime.now()

        # Skip remote for unknown user
        remote_allowed = self.remote and self.remote.enabled and user_id != UNKNOWN_USER_ID

        jpeg_local = None
        jpeg_remote = None

        if frame is not None:
            jpeg_local = self._encode_jpeg(frame, self.local_quality)
            if remote_allowed:
                # Use a reasonable resolution for remote (about 480p width)
                h, w = frame.shape[:2]
                target_w = 640  # was 320; higher but still fast over Wi‑Fi
                if w > target_w:
                    scale = target_w / float(w)
                    resized = cv2.resize(frame, (target_w, int(round(h * scale))))
                else:
                    resized = frame
                jpeg_remote = self._encode_jpeg(resized, self.remote_quality)

        if self.repo:
            event = DrowsinessEvent(
                vehicle_identification_number=self.vehicle_vin,
                user_id=user_id,
                status=event_type.lower(),
                time=timestamp,
                img_drowsiness=jpeg_local,
                img_path=None,
            )
            setattr(event, "duration", duration)
            setattr(event, "value", value)
            self.repo.add_event(event)
            logging.info(f"[LOG] Local: {event_type}")

        if remote_allowed and self.remote:
            # Pass raw JPEG bytes (remote thread will base64 + send)
            self.remote.send_or_queue(
                vehicle_vin=self.vehicle_vin,
                user_id=user_id,
                status=event_type,
                time_dt=timestamp,
                raw_jpeg=jpeg_remote
            )

    def alert(self, level: str = "warning"):
        """Triggers buzzer. Fixed TypeError."""
        if not self.buzzer: return
        
        if level == "warning":     
            self.buzzer.beep(0.1, 0.1, background=True)
        elif level == "critical":  
            self.buzzer.beep(0.5, 0.5, background=True)
        elif level == "distraction": 
            self.buzzer.beep(0.1, 0.1, background=True)
            
    def stop_alert(self):
        if self.buzzer: self.buzzer.off()

    def _encode_jpeg(self, frame, quality):
        try:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                                 [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            return bytes(buf) if ok else None
        except: return None