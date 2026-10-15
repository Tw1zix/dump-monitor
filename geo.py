"""Геомодуль: расстояния, привязка рамки в кадре к точке на земле, охранные зоны."""
import math, os

M_PER_DEG = 111320.0
# Горизонтальный угол обзора камеры дрона, градусы. Задайте под вашу модель (CAMERA_HFOV_DEG).
HFOV_DEG = float(os.environ.get("CAMERA_HFOV_DEG", 70))
NADIR_MIN_PITCH = 80   # камера считается смотрящей вниз при наклоне подвеса от 80° до 90°


def dist_m(lat1, lon1, lat2, lon2):
    """Расстояние между двумя точками в метрах (для малых расстояний)."""
    dy = (lat2 - lat1) * M_PER_DEG
    dx = (lon2 - lon1) * M_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, dy)


def offset(lat, lon, north_m, east_m):
    """Сдвиг точки на заданное число метров к северу и востоку."""
    return (lat + north_m / M_PER_DEG,
            lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def box_to_geo(lat, lon, alt, yaw, pitch, box, w, h, hfov_deg=None):
    """Координата центра рамки на земле. -> (lat, lon, precise).

    Работает для камеры, смотрящей вниз. Нужны высота над землёй, курс (gb_yaw)
    и наклон подвеса (gb_pitch). Если чего-то нет или камера смотрит под углом,
    возвращается позиция дрона и precise=False.
    """
    if None in (lat, lon, alt, yaw, pitch) or alt <= 0 or abs(pitch) < NADIR_MIN_PITCH:
        return lat, lon, False
    hfov = math.radians(hfov_deg or HFOV_DEG)
    m_per_px = 2 * alt * math.tan(hfov / 2) / w          # метров земли в одном пикселе
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    fwd, right = (h / 2 - cy) * m_per_px, (cx - w / 2) * m_per_px
    a = math.radians(yaw)                                  # курс: от севера по часовой
    north = fwd * math.cos(a) - right * math.sin(a)
    east = fwd * math.sin(a) + right * math.cos(a)
    return (*offset(lat, lon, north, east), True)


def in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def in_geometry(x, y, g):
    """Точка (lon=x, lat=y) внутри Polygon или MultiPolygon из GeoJSON (с учётом дыр)."""
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]] if g["type"] == "Polygon" else []
    pts = lambda r: [(p[0], p[1]) for p in r]
    return any(in_ring(x, y, pts(p[0])) and not any(in_ring(x, y, pts(hole)) for hole in p[1:]) for p in polys)
