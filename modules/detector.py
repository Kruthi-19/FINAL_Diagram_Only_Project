import cv2
import os
import numpy as np

RESULT_DIR = "static/results"


def detect_venue_objects(image_path, event_data):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read uploaded image.")

    h, w = img.shape[:2]

    detections = []

    # =========================================================
    # STAGE DETECTION
    # =========================================================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    stage_found = False

    best_stage = None
    best_area = 0

    for cnt in contours:

        x, y, cw, ch = cv2.boundingRect(cnt)

        area = cw * ch

        aspect_ratio = cw / max(ch, 1)

        # Better stage filtering
        if (
            area > 20000 and
            aspect_ratio > 2.5 and
            h * 0.20 < y < h * 0.70
        ):

            if area > best_area:

                best_area = area

                best_stage = [x, y, x + cw, y + ch]

                stage_found = True

    # Use detected stage
    if stage_found:

        detections.append({
            "label": "stage",
            "bbox": best_stage
        })

    # Fallback stage
    else:

        detections.append({
            "label": "stage",
            "bbox": [
                int(w * .25),
                int(h * .35),
                int(w * .75),
                int(h * .55)
            ]
        })

    # =========================================================
    # ENTRY
    # =========================================================

    detections.append({
        "label": "entry",
        "bbox": [
            int(w * .40),
            int(h * .88),
            int(w * .60),
            int(h * .98)
        ]
    })

    # =========================================================
    # EXITS
    # =========================================================

    detections.append({
        "label": "exit",
        "bbox": [
            0,
            int(h * .38),
            int(w * .10),
            int(h * .62)
        ]
    })

    detections.append({
        "label": "exit",
        "bbox": [
            int(w * .90),
            int(h * .38),
            int(w * .99),
            int(h * .62)
        ]
    })

    # =========================================================
    # NO PILLARS
    # =========================================================

    # Removed completely

    # =========================================================
    # NO HELPDESK
    # =========================================================

    # Removed completely

    return detections, draw_detections(image_path, detections)


def draw_detections(image_path, detections):

    img = cv2.imread(image_path)

    colors = {
        "stage": (180, 80, 255),
        "exit": (0, 180, 0),
        "entry": (255, 100, 0)
    }

    for det in detections:

        label = det["label"]

        x1, y1, x2, y2 = map(int, det["bbox"])

        color = colors.get(label, (255, 255, 255))

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            img,
            label.upper(),
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    out = os.path.join(
        RESULT_DIR,
        f"{name}_01_detected.jpg"
    )

    cv2.imwrite(out, img)

    return out