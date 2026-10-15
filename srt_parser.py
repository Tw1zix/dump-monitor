"""Парсинг RTK-телеметрии из SRT-файла DJI + привязка кадров к координатам."""
import re
from bisect import bisect_left

RE_LAT = re.compile(r"\[latitude:\s*([\d.\-]+)\]", re.I)
RE_LON = re.compile(r"\[longitude:\s*([\d.\-]+)\]", re.I)
RE_ALT = re.compile(r"\[(?:rel_alt|altitude):\s*([\d.\-]+)\]", re.I)
RE_TS = re.compile(r"\[timestamp:\s*(\d+)\]")
RE_RTK = re.compile(r"\[rtk_status:\s*(\w+)\]", re.I)
RE_YAW = re.compile(r"\[gb_yaw:\s*([\d.\-]+)\]", re.I)      # курс подвеса, градусы от севера
RE_PITCH = re.compile(r"\[gb_pitch:\s*([\d.\-]+)\]", re.I)  # наклон подвеса, -90 = строго вниз


def parse_srt(path):
    """-> отсортированный список (ts_ms, lat, lon, alt, rtk_status)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        blocks = re.split(r"\n\s*\n", f.read())
    points = []
    for b in blocks:
        m_ts, m_lat, m_lon = RE_TS.search(b), RE_LAT.search(b), RE_LON.search(b)
        if not (m_ts and m_lat and m_lon):
            continue
        m_alt = RE_ALT.search(b)
        m_rtk = RE_RTK.search(b)
        m_yaw, m_pitch = RE_YAW.search(b), RE_PITCH.search(b)
        points.append((
            int(m_ts.group(1)),
            float(m_lat.group(1)),
            float(m_lon.group(1)),
            float(m_alt.group(1)) if m_alt else 0.0,
            (m_rtk.group(1).lower() if m_rtk else "unknown"),
            float(m_yaw.group(1)) if m_yaw else None,
            float(m_pitch.group(1)) if m_pitch else None,
        ))
    points.sort()
    return points


def geo_at_time(points, frame_ts_ms):
    """Линейная интерполяция координат на момент кадра. -> (lat, lon, alt, rtk, yaw, pitch)"""
    if not points:
        return (None,) * 6
    if len(points) == 1:
        p = points[0]
        return p[1], p[2], p[3], p[4], p[5], p[6]
    ts = [p[0] for p in points]
    i = bisect_left(ts, frame_ts_ms)
    i = min(max(i, 1), len(points) - 1)
    t0, t1 = points[i - 1][0], points[i][0]
    k = (frame_ts_ms - t0) / max(t1 - t0, 1)
    lerp = lambda a, b: a + (b - a) * k
    return (lerp(points[i - 1][1], points[i][1]),
            lerp(points[i - 1][2], points[i][2]),
            lerp(points[i - 1][3], points[i][3]),
            points[i][4], points[i][5], points[i][6])
