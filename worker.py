"""Ядро обработки: БД, гео-привязка, дедупликация, фоновая задача.
Отделено от Flask, чтобы тестировать без веб-сервера."""
import os, uuid, datetime
import cv2
from srt_parser import parse_srt, geo_at_time
from detector import Detector
from geo import box_to_geo

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "detections.db")
UPLOADS = os.path.join(BASE, "uploads")
CROPS = os.path.join(BASE, "static", "crops")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(CROPS, exist_ok=True)
CANCEL = set()  # id заданий, которые просили остановить

import sqlite3
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT, video_name TEXT,
        lat REAL, lon REAL, alt REAL, rtk TEXT,
        confidence REAL, frame_ts_ms INTEGER,
        image_path TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, video_name TEXT,
        status TEXT, total_frames INTEGER DEFAULT 0,
        done_frames INTEGER DEFAULT 0, detections INTEGER DEFAULT 0,
        created_at TEXT);
    """)
    for col, typ in (("status", "TEXT DEFAULT 'new'"), ("frame_path", "TEXT")):
        try:
            conn.execute(f"ALTER TABLE detections ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # колонка уже есть
    # задания, прерванные перезапуском сервера, не должны висеть как «идёт»
    conn.execute("UPDATE jobs SET status='error' WHERE status IN ('running','queued')")
    conn.commit()
    conn.close()

def is_duplicate(conn, lat, lon, radius_m=5.0, job_id=None, ts=None):
    """Дедупликация находок ближе radius_m метров (RTK-точность)."""
    if lat is None:
        return False
    d = radius_m / 111320.0
    row = conn.execute(
        "SELECT id FROM detections WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
        "AND NOT (job_id IS ? AND frame_ts_ms IS ?) LIMIT 1",   # объекты одного кадра не дубли
        (lat - d, lat + d, lon - d, lon + d, job_id, ts)).fetchone()
    return row is not None

def iou(a, b):
    """Пересечение рамок (x1,y1,x2,y2) / объединение."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1)

def process_job(job_id, video_path, srt_path, interval_ms, conf_threshold, model_path,
                progress=None):
    """Нарезка видео + детекция + геопривязка + сохранение находок.
    progress(dict) — опциональный словарь для обновления статуса в памяти."""
    conn = get_db()
    if progress is not None:
        progress[job_id] = {"status": "running", "detections": 0}
    try:
        detector = Detector(model_path)
        telemetry = parse_srt(srt_path) if srt_path else []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_ms = total / fps * 1000
        frames_to_check = max(1, int(total_ms // interval_ms))
        conn.execute("UPDATE jobs SET total_frames=? WHERE id=?", (frames_to_check, job_id))
        conn.commit()

        found, checked, frame_idx, prev, stopped = 0, 0, 0, [], False
        while True:
            if job_id in CANCEL:
                CANCEL.discard(job_id); stopped = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = frame_idx / fps * 1000
            if ts_ms >= checked * interval_ms:
                checked += 1
                cur = []
                for (x1, y1, x2, y2, conf) in detector.detect(frame, conf_threshold):
                    if conf < conf_threshold:
                        continue
                    box = (x1, y1, x2, y2); cur.append(box)
                    # без SRT: та же куча на соседнем кадре не считается новой находкой
                    if not telemetry and any(iou(box, pb) > 0.3 for pb in prev):
                        continue
                    geo = geo_at_time(telemetry, telemetry[0][0] + ts_ms) if telemetry \
                        else (None,) * 6
                    lat, lon, alt, rtk, yaw, pitch = geo
                    fh, fw = frame.shape[:2]
                    # координата самого объекта (по положению рамки в кадре), а не дрона
                    lat, lon, _ = box_to_geo(lat, lon, alt, yaw, pitch, (x1, y1, x2, y2), fw, fh)
                    if is_duplicate(conn, lat, lon, job_id=job_id, ts=int(ts_ms)):
                        continue
                    crop_name = f"{uuid.uuid4().hex}.jpg"
                    h, w = frame.shape[:2]
                    full = frame.copy()
                    cv2.rectangle(full, (x1, y1), (x2, y2), (0, 200, 255), max(3, w // 300))
                    frame_name = f"{uuid.uuid4().hex}_full.jpg"
                    cv2.imwrite(os.path.join(CROPS, frame_name), full)
                    cv2.imwrite(os.path.join(CROPS, crop_name),
                                frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)])
                    conn.execute(
                        "INSERT INTO detections(job_id,video_name,lat,lon,alt,rtk,"
                        "confidence,frame_ts_ms,image_path,frame_path,created_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (job_id, os.path.basename(video_path), lat, lon, alt, rtk,
                         conf, int(ts_ms), f"static/crops/{crop_name}", f"static/crops/{frame_name}",
                         datetime.datetime.now().isoformat(timespec="seconds")))
                    conn.commit()
                    found += 1
                prev = cur
                conn.execute("UPDATE jobs SET done_frames=?, detections=? WHERE id=?",
                             (checked, found, job_id))
                conn.commit()
                if progress is not None:
                    progress[job_id].update(done=checked, detections=found)
            frame_idx += 1
        cap.release()
        final = "stopped" if stopped else "done"
        conn.execute("UPDATE jobs SET status=? WHERE id=?", (final, job_id))
        conn.commit()
        if progress is not None:
            progress[job_id]["status"] = final
    except Exception as e:
        conn.execute("UPDATE jobs SET status='error' WHERE id=?", (job_id,))
        conn.commit()
        if progress is not None:
            progress[job_id] = {"status": "error", "error": str(e)}
        raise
    finally:
        conn.close()
        if not os.environ.get("KEEP_UPLOADS"):  # видео после обработки не нужно
            for p in (video_path, srt_path):
                try:
                    if p: os.remove(p)
                except OSError:
                    pass
