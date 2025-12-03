import logging
import threading
import sqlite3
import queue
from datetime import datetime
from typing import Optional
from src.api_client.api_service import ApiService
from src.api_client.event import DrowsinessEvent as ApiEvent
from src.api_client import config

log = logging.getLogger(__name__)

class RemoteLogWorker:
    RETRY_INTERVAL_SEC = 30
    MAX_QUEUE_SIZE = 100
    SEND_BATCH_SIZE = 5

    def __init__(self, db_path: str, remote_api_url: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        target_base_url = remote_api_url if remote_api_url else config.SERVER_BASE_URL
        self.api_service = ApiService(base_url=target_base_url) if enabled else None
        if self.api_service:
            log.info(f"[REMOTE] Worker started (Base: {target_base_url})")
        else:
            log.info("[REMOTE] Worker disabled")

        # Create the simple queue table (matches your schema)
        self.queue_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.queue_conn.execute("""
            CREATE TABLE IF NOT EXISTS remote_event_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                vehicle_vin TEXT, 
                user_id INTEGER, 
                status TEXT, 
                time TEXT
            )
        """)
        self.queue_conn.commit()

        # In‑memory immediate queue
        self._immediate_q: "queue.Queue[tuple]" = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()

        self._retry_thread = None
        self._send_thread = None
        if self.enabled and self.api_service:
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
            self._retry_thread = threading.Thread(target=self._retry_loop, daemon=True)
            self._retry_thread.start()

    def send_or_queue(self, vehicle_vin: str, user_id: int, status: str, 
                      time_dt: datetime, raw_jpeg: Optional[bytes]):
        """Push into immediate queue; follow DB time format and ApiEvent expectations."""
        if not (self.enabled and self.api_service):
            return
        try:
            self._immediate_q.put_nowait((vehicle_vin, user_id, status, time_dt, raw_jpeg))
        except queue.Full:
            log.warning("[REMOTE] Immediate queue full; falling back to persistent queue")
            self._queue(vehicle_vin, user_id, status, time_dt, raw_jpeg)

    def _send_event(self, vin, uid, status, dt, jpeg_bytes) -> bool:
        """Send to API. ApiService expects event.time as datetime; event.py formats to 'YYYY-MM-DD HH:MM:SS'."""
        try:
            # Coerce dt -> datetime
            if isinstance(dt, datetime):
                dt_obj = dt
            else:
                # If caller provided a string or anything else, ignore and use now
                dt_obj = datetime.now()

            # Base64 if image present
            b64 = None
            if jpeg_bytes:
                import base64
                b64 = base64.b64encode(jpeg_bytes).decode("ascii")

            event = ApiEvent(
                vehicle_identification_number=vin,
                user_id=uid,
                status=status,
                time=dt_obj,           # datetime object (event.py will strftime it)
                img_drowsiness=b64,
                img_path=None
            )

            # Final safety: ensure event.time is datetime (not str)
            if not isinstance(event.time, datetime):
                event.time = datetime.now()

            res = self.api_service.send_drowsiness_event(event)
            if getattr(res, "success", False):
                log.info(f"[REMOTE] ✓ Sent (CID: {getattr(res, 'correlation_id', '-')})")
                return True
            log.warning(f"[REMOTE] Send failed: {getattr(res, 'error', 'unknown error')}")
            return False
        except Exception as e:
            log.error(f"[REMOTE] Send exception: {e}")
            return False

    def _send_loop(self):
        while not self._stop_event.is_set():
            try:
                vin, uid, status, dt, jpeg_bytes = self._immediate_q.get(timeout=0.25)
                ok = self._send_event(vin, uid, status, dt, jpeg_bytes)
                if not ok:
                    self._queue(vin, uid, status, dt, jpeg_bytes)
            except queue.Empty:
                pass
            except Exception as e:
                log.error(f"[REMOTE] Send loop error: {e}")

    def _queue(self, vin, uid, status, dt, jpeg_bytes):
        """Persist to DB using simple time format 'YYYY-mm-dd HH:MM:SS' (matches your table)."""
        try:
            # Normalize dt to DB string
            if isinstance(dt, datetime):
                t_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cur = self.queue_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM remote_event_queue")
            if cur.fetchone()[0] >= self.MAX_QUEUE_SIZE:
                cur.execute("DELETE FROM remote_event_queue WHERE id = (SELECT id FROM remote_event_queue ORDER BY id ASC LIMIT 1)")
            cur.execute("""
                INSERT INTO remote_event_queue (vehicle_vin, user_id, status, time)
                VALUES (?,?,?,?)
            """, (vin, uid, status, t_str))
            self.queue_conn.commit()
            log.info("[REMOTE] ⧖ Queued")
        except Exception as e:
            log.error(f"[REMOTE] Queue error: {e}")

    def _retry_loop(self):
        while not self._stop_event.wait(self.RETRY_INTERVAL_SEC):
            self._process_queue()

    def _process_queue(self):
        """Read rows, parse time with strptime, send, then delete on success."""
        try:
            cur = self.queue_conn.cursor()
            cur.execute("SELECT id, vehicle_vin, user_id, status, time FROM remote_event_queue ORDER BY id ASC LIMIT ?", (self.SEND_BATCH_SIZE,))
            rows = cur.fetchall()
            if not rows:
                return
            for row in rows:
                qid, vin, uid, stat, t_str = row
                try:
                    dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.now()

                if self._send_event(vin, uid, stat, dt, None):
                    cur.execute("DELETE FROM remote_event_queue WHERE id=?", (qid,))
                    log.info(f"[REMOTE] ✓ Retry Success ID:{qid}")
            self.queue_conn.commit()
        except Exception as e:
            log.error(f"[REMOTE] Retry error: {e}")

    def close(self):
        self._stop_event.set()
        if self._send_thread:
            self._send_thread.join(1.0)
        if self._retry_thread:
            self._retry_thread.join(1.0)
        self.queue_conn.close()