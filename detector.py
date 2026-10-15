"""Детектор мусора на базе вашей обученной YOLOv8 (models/best.pt).

Модель: YOLOv8s, 1 класс ("waste"), обучена на imgsz=640.

Порядок выбора весов:
  1) путь из поля «Путь к модели» в веб-интерфейсе
  2) переменная окружения MODEL_PATH
  3) models/best.pt (лежит в проекте, используется по умолчанию)
Если веса не найдены вообще — включается демо-эвристика (цветовая маска),
чтобы сайт запускался. Если путь указан явно, но файл не загрузился —
выбрасывается ошибка, а не тихий откат в демо (иначе легко принять
демо-результаты за работу нейросети).
"""
import os
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE, "models", "best.pt")
IMGSZ = int(os.environ.get("IMGSZ", 640))        # размер, на котором обучалась модель
DEVICE = os.environ.get("DEVICE") or None        # "cpu", "0" (GPU) и т.д.; None = авто

_MODEL_CACHE = {}  # path -> YOLO, чтобы не грузить веса на каждое видео


def _load_yolo(path):
    if path not in _MODEL_CACHE:
        from ultralytics import YOLO
        _MODEL_CACHE[path] = YOLO(path)
    return _MODEL_CACHE[path]


class Detector:
    def __init__(self, model_path=None):
        self.model = None
        self.kind = "demo"
        explicit = model_path or os.environ.get("MODEL_PATH", "")
        path = explicit or DEFAULT_MODEL

        if explicit and not os.path.exists(explicit):
            raise FileNotFoundError(f"Файл модели не найден: {explicit}")

        if os.path.exists(path):
            try:
                self.model = _load_yolo(path)
                self.kind = "yolo"
                self.path = path
                print(f"[detector] YOLO загружена: {path}, классы: {self.model.names}")
                return
            except Exception as e:
                if explicit:
                    raise RuntimeError(f"Не удалось загрузить модель {path}: {e}") from e
                print(f"[detector] не удалось загрузить {path} ({e}), демо-режим")
        else:
            print("[detector] веса не найдены, демо-режим")

    def detect(self, frame_bgr, conf_threshold=0.4):
        """-> список (x1, y1, x2, y2, confidence)"""
        if self.kind == "yolo":
            res = self.model.predict(frame_bgr, conf=conf_threshold, imgsz=IMGSZ,
                                     device=DEVICE, verbose=False)[0]
            out = []
            for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = map(int, box)
                out.append((x1, y1, x2, y2, float(conf)))
            return out
        return self._demo_detect(frame_bgr)

    def _demo_detect(self, frame_bgr):
        """Демо: ищем тёмно-зелёные/серые кучи (HSV-маска) — имитация находок."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 40, 20]), np.array([95, 255, 160]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame_bgr.shape[:2]
        out = []
        for c in cnts:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw * ch > 0.005 * w * h:
                area = cv2.contourArea(c) / max(cw * ch, 1)
                out.append((x, y, x + cw, y + ch, round(0.4 + 0.6 * area, 3)))
        return out
