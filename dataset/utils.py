import json
from pathlib import Path
from .doclaynet import LABELS

def xywh_to_xyxy_1000(bbox_xywh, W, H):
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    return [
        int(round(x1 / W * 1000)),
        int(round(y1 / H * 1000)),
        int(round(x2 / W * 1000)),
        int(round(y2 / H * 1000)),
    ]

# labels = "|".join(LABELS)
PROMPT = (
    f"Detect the layout of this document. Report JSON list ONLY. Each JSON includes label and bbox_2d."
)

def to_messages(data: dict):
    w, h = data["width"], data["height"]
    objects = data["objects"]

    pairs = [
        (cid, xywh_to_xyxy_1000(bbox_xywh, w, h)) for cid, bbox_xywh in zip(objects["category_id"], objects["bbox"])
        if cid is not None and cid >= 0
        ]

    answer = "[\n"
    for cid, bbox in pairs:
        answer += " "*2 + json.dumps({"bbox_2d": bbox, "label": LABELS[cid]}, ensure_ascii = False) + ",\n"
    answer = answer.rstrip(",\n") + "\n]"
    
    return {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": data["image"]}, {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }

def to_messages_img_path_only(data: dict, base_dir: str):
    w, h = data["width"], data["height"]
    objects = data["objects"]

    pairs = [
        (cid, xywh_to_xyxy_1000(bbox_xywh, w, h)) for cid, bbox_xywh in zip(objects["category_id"], objects["bbox"])
        if cid is not None and cid >= 0
        ]

    answer = "[\n"
    for cid, bbox in pairs:
        answer += " "*2 + json.dumps({"bbox_2d": bbox, "label": LABELS[cid]}, ensure_ascii = False) + ",\n"
    answer = answer.rstrip(",\n") + "\n]"
    
    file_name = data["file_name"]
    image_path = str(Path(base_dir).joinpath("PNG", file_name))

    return {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }