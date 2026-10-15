import os, io, csv, json, hmac, math, urllib.request, urllib.parse, zipfile, uuid, threading, datetime
from flask import (Flask, request, jsonify, render_template, send_file, Response)
from geo import in_geometry as _in_geom, dist_m
from worker import get_db, init_db, process_job, CANCEL, DB_PATH, CROPS

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
JOBS = {}  # job_id -> статус в памяти
init_db()
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3   # до 4 ГБ на загрузку
PASSWORD = os.environ.get("APP_PASSWORD")           # если задан, сайт просит пароль
RUN_LOCK = threading.Lock()                         # задания идут по одному

@app.before_request
def _auth():
    if PASSWORD:
        a = request.authorization
        if not a or not hmac.compare_digest(a.password or "", PASSWORD):
            return Response("Нужен пароль", 401, {"WWW-Authenticate": 'Basic realm="kimo"'})

@app.errorhandler(413)
def _too_big(e):
    return jsonify({"error": "Файл слишком большой (максимум 4 ГБ)"}), 413

def _set_status(job_id, status):
    conn = get_db()
    conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))
    conn.commit(); conn.close()

def _notify(text):
    """Сообщение в Telegram, если заданы TELEGRAM_TOKEN и TELEGRAM_CHAT."""
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT")
    if tok and chat:
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", timeout=10,
                                   data=urllib.parse.urlencode({"chat_id": chat, "text": text}).encode())
        except Exception:
            pass

def _run(job_id, *args):
    with RUN_LOCK:
        if job_id in CANCEL:                         # остановили, пока ждало очереди
            CANCEL.discard(job_id); _set_status(job_id, "stopped")
            for p in args[:2]:
                if p and os.path.exists(p): os.remove(p)
            return
        _set_status(job_id, "running")
        try:
            process_job(job_id, *args)
        finally:
            conn = get_db()
            j = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            conn.close()
            if j:
                _notify(f"Обработка «{j['video_name']}» завершена ({j['status']}). Находок: {j['detections']}")

# ---------- маршруты ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "нет видео"}), 400
    job_id = uuid.uuid4().hex[:12]
    vname = os.path.basename(video.filename or "")
    if os.path.splitext(vname)[1].lower() not in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
        return jsonify({"error": "Нужен видеофайл MP4, MOV, AVI или MKV"}), 400
    srt = request.files.get("srt")
    if srt and srt.filename and not srt.filename.lower().endswith(".srt"):
        return jsonify({"error": "Телеметрия должна быть файлом .srt"}), 400
    try:
        interval = max(200, int(request.form.get("interval", 1000)))
        conf = min(0.95, max(0.05, float(request.form.get("conf", 0.4))))
    except ValueError:
        return jsonify({"error": "Неверные значения интервала или порога"}), 400
    from worker import UPLOADS
    video_path = os.path.join(UPLOADS, f"{job_id}_{vname}")
    video.save(video_path)
    srt_path = None
    srt = request.files.get("srt")
    if srt and srt.filename:
        srt_path = os.path.join(UPLOADS, f"{job_id}_{os.path.basename(srt.filename)}")
        srt.save(srt_path)
    model_path = request.form.get("model", "") or None
    conn = get_db()
    conn.execute("INSERT INTO jobs(id,video_name,status,created_at) VALUES(?,?,?,?)",
                 (job_id, vname, "queued",
                  datetime.datetime.now().isoformat(timespec="seconds")))
    conn.commit(); conn.close()
    threading.Thread(target=_run,
                     args=(job_id, video_path, srt_path, interval, conf, model_path, JOBS),
                     daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/job/<job_id>")
def job_status(job_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return jsonify(dict(row) if row else {"error": "not found"})

@app.route("/api/detections")
def detections():
    conn = get_db()
    rows = conn.execute("SELECT * FROM detections ORDER BY id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/report")
def report():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
    by_video = [dict(r) for r in conn.execute(
        "SELECT video_name, COUNT(*) cnt, ROUND(AVG(confidence),2) avg_conf "
        "FROM detections GROUP BY video_name")]
    by_rtk = [dict(r) for r in conn.execute(
        "SELECT COALESCE(rtk,'нет RTK') rtk, COUNT(*) cnt FROM detections GROUP BY rtk")]
    recent = [dict(r) for r in conn.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT 50")]
    jobs = [dict(r) for r in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")]
    conn.close()
    return jsonify({"total": total, "by_video": by_video,
                    "by_rtk": by_rtk, "recent": recent, "jobs": jobs})

@app.route("/api/download/zip")
def download_zip():
    conn = get_db()
    rows = conn.execute("SELECT image_path, video_name, id FROM detections").fetchall()
    conn.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            full = os.path.join(BASE, r["image_path"])
            if os.path.exists(full):
                z.write(full, f"{r['video_name']}/detection_{r['id']}.jpg")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="detections.zip")

@app.route("/api/detection/<int:det_id>/delete", methods=["POST"])
def delete_detection(det_id):
    conn = get_db()
    row = conn.execute("SELECT image_path, frame_path FROM detections WHERE id=?", (det_id,)).fetchone()
    if row:
        for key in ("image_path", "frame_path"):
            full = os.path.join(BASE, row[key] or "")
            if row[key] and os.path.exists(full):
                os.remove(full)
        conn.execute("DELETE FROM detections WHERE id=?", (det_id,))
        conn.commit()
    conn.close()
    return jsonify({"ok": True})

@app.route("/api/detection/<int:det_id>/status", methods=["POST"])
def set_status(det_id):
    st = (request.get_json(silent=True) or {}).get("status")
    if st not in ("new", "checked", "confirmed", "false"):
        return jsonify({"error": "bad status"}), 400
    conn = get_db()
    conn.execute("UPDATE detections SET status=? WHERE id=?", (st, det_id))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

def _export_rows():
    conn = get_db()
    rows = [dict(r) for r in conn.execute("SELECT * FROM detections ORDER BY id")]
    conn.close()
    for r in rows:
        r["video_name"] = r["video_name"].split("_", 1)[-1]
        r["status"] = r["status"] or "new"
    return rows

COLS = ["id", "video_name", "lat", "lon", "alt", "rtk", "confidence", "status",
        "frame_ts_ms", "created_at"]

@app.route("/api/export/csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLS)
    for r in _export_rows():
        w.writerow([r[c] for c in COLS])
    return Response("\ufeff" + buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=detections.csv"})

@app.route("/api/export/geojson")
def export_geojson():
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
              "properties": {c: r[c] for c in COLS if c not in ("lat", "lon")}}
             for r in _export_rows() if r["lat"] is not None]
    return Response(json.dumps({"type": "FeatureCollection", "features": feats},
                               ensure_ascii=False),
                    mimetype="application/geo+json", headers={
        "Content-Disposition": "attachment; filename=detections.geojson"})

# ---------- охранные зоны и свалки ----------
ZONES = os.path.join(BASE, "zones.json")
SITE_RADIUS_M = 20          # находки ближе этого радиуса считаются одной свалкой
INBOX = os.path.join(BASE, "inbox")
os.makedirs(INBOX, exist_ok=True)

def _zones():
    try:
        with open(ZONES, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"type": "FeatureCollection", "features": []}

@app.route("/api/zones", methods=["GET", "POST"])
def zones():
    if request.method == "POST":
        try:
            data = json.load(request.files["file"])
            if data.get("type") == "Feature":
                data = {"type": "FeatureCollection", "features": [data]}
            assert data["type"] == "FeatureCollection"
        except Exception:
            return jsonify({"error": "Нужен корректный файл GeoJSON с полигонами зон"}), 400
        with open(ZONES, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return jsonify(_zones())

@app.route("/api/sites")
def sites():
    """Находки ближе SITE_RADIUS_M объединяются в свалку; для каждой считаются охранные зоны."""
    conn = get_db()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, lat, lon, status FROM detections WHERE lat IS NOT NULL "
        "AND COALESCE(status,'new')!='false' ORDER BY id")]
    conn.close()
    out = []
    for r in rows:
        for s in out:
            if dist_m(s["lat"], s["lon"], r["lat"], r["lon"]) <= SITE_RADIUS_M:
                s["ids"].append(r["id"]); s["n"] += 1; s["st"].append(r["status"] or "new")
                s["lat"] += (r["lat"] - s["lat"]) / s["n"]; s["lon"] += (r["lon"] - s["lon"]) / s["n"]
                break
        else:
            out.append({"lat": r["lat"], "lon": r["lon"], "n": 1, "ids": [r["id"]], "st": [r["status"] or "new"]})
    feats = [f for f in _zones()["features"] if f.get("geometry")]
    for i, s in enumerate(out, 1):
        s["no"] = i
        s["status"] = "confirmed" if "confirmed" in s["st"] else "checked" if "checked" in s["st"] else "new"
        s["zones"] = [((f.get("properties") or {}).get("name") or (f.get("properties") or {}).get("Name") or "Зона")
                      for f in feats if _in_geom(s["lon"], s["lat"], f["geometry"])]
        del s["st"]
    return jsonify({"sites": sorted(out, key=lambda s: -s["n"])})

# ---------- папка-приёмник: положили видео в inbox/, обработка стартует сама ----------
def _watch():
    import time
    from worker import UPLOADS
    seen = {}
    while True:
        time.sleep(10)
        for n in os.listdir(INBOX):
            p = os.path.join(INBOX, n)
            if os.path.splitext(n)[1].lower() not in (".mp4", ".mov", ".avi", ".mkv", ".m4v") or not os.path.isfile(p):
                continue
            size = os.path.getsize(p)
            if seen.get(p) != size:          # файл ещё копируется
                seen[p] = size
                continue
            seen.pop(p, None)
            job_id = uuid.uuid4().hex[:12]
            vpath, spath = os.path.join(UPLOADS, f"{job_id}_{n}"), None
            srt = os.path.splitext(p)[0] + ".srt"
            try:
                if os.path.exists(srt):
                    spath = os.path.join(UPLOADS, f"{job_id}_{os.path.basename(srt)}")
                    os.replace(srt, spath)
                os.replace(p, vpath)
            except OSError:
                continue
            conn = get_db()
            conn.execute("INSERT INTO jobs(id,video_name,status,created_at) VALUES(?,?,?,?)",
                         (job_id, n, "queued", datetime.datetime.now().isoformat(timespec="seconds")))
            conn.commit(); conn.close()
            threading.Thread(target=_run, args=(job_id, vpath, spath, 1000, 0.4, None, JOBS), daemon=True).start()

@app.route("/api/job/<job_id>/stop", methods=["POST"])
def stop_job(job_id):
    CANCEL.add(job_id)
    return jsonify({"ok": True})

def _rm_files(row):
    for k in ("image_path", "frame_path"):
        if row[k] and os.path.exists(os.path.join(BASE, row[k])):
            os.remove(os.path.join(BASE, row[k]))

@app.route("/api/job/<job_id>/delete", methods=["POST"])
def delete_job(job_id):
    conn = get_db()
    job = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    if job and job["status"] in ("running", "queued"):
        conn.close()
        return jsonify({"error": "Сначала остановите задание"}), 400
    for r in conn.execute("SELECT image_path, frame_path FROM detections WHERE job_id=?", (job_id,)).fetchall():
        _rm_files(r)
    conn.execute("DELETE FROM detections WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit(); conn.close()
    return jsonify({"ok": True})

@app.route("/api/backup")
def backup():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(DB_PATH, "detections.db")
        for n in os.listdir(CROPS):
            z.write(os.path.join(CROPS, n), f"static/crops/{n}")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="kimo_backup.zip")

@app.route("/api/export/false")
def export_false():
    """Снимки с отметкой «Ложная» для дообучения модели."""
    conn = get_db()
    rows = conn.execute("SELECT id, image_path, frame_path FROM detections WHERE status='false'").fetchall()
    conn.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            f = os.path.join(BASE, r["frame_path"] or r["image_path"])
            if os.path.exists(f):
                z.write(f, f"false_{r['id']}.jpg")
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="false_positives.zip")

@app.route("/report/print")
def report_print():
    """Страница отчёта: откройте и нажмите «Печать → Сохранить как PDF»."""
    from markupsafe import escape as e
    job = request.args.get("job")
    sql = "SELECT * FROM detections WHERE COALESCE(status,'new')!='false'" + (" AND job_id=?" if job else "") + " ORDER BY id"
    conn = get_db()
    rows = conn.execute(sql, (job,) if job else ()).fetchall()
    conn.close()
    ST = {"new": "Новая", "checked": "Проверена", "confirmed": "Подтверждена"}
    cards = "".join(
        f'<div class="c"><img src="/{e(r["frame_path"] or r["image_path"])}"><p><b>№{r["id"]}</b> '
        f'{e(r["video_name"].split("_", 1)[-1])}<br>Уверенность {r["confidence"]*100:.0f}%, RTK: {e(r["rtk"] or "нет")}<br>'
        + (f'{r["lat"]:.6f}, {r["lon"]:.6f}' if r["lat"] is not None else "без координат")
        + f'<br>{ST.get(r["status"] or "new", "")}</p></div>' for r in rows)
    geo = sum(r["lat"] is not None for r in rows)
    conf = sum((r["status"] == "confirmed") for r in rows)
    return (f'<!doctype html><meta charset="utf-8"><title>Отчёт</title><style>'
            'body{font:14px Segoe UI,Arial;margin:24px;color:#18231d}.g{display:grid;grid-template-columns:1fr 1fr;gap:14px}'
            '.c{break-inside:avoid;border:1px solid #ccd;border-radius:8px;padding:8px}img{width:100%;border-radius:6px}'
            '@media print{button{display:none}}</style>'
            f'<h1>Отчёт о несанкционированных свалках</h1><p>{datetime.date.today():%d.%m.%Y}. Находок: {len(rows)}, '
            f'с координатами: {geo}, подтверждено: {conf}.</p><button onclick="print()">Печать или PDF</button>'
            f'<div class="g">{cards}</div>')

if __name__ == "__main__":
    threading.Thread(target=_watch, daemon=True).start()
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=5000, threaded=True)
