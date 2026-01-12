from PIL import Image, ImageDraw
from typing import Literal

def absolute_to_relative_coords(
        absolute_coords: list[list[int]], 
        width: int, 
        height: int
        ):
    relative_points = []
    for x0_abs, y0_abs, x1_abs, y1_abs in absolute_coords:
        x0_rel = int((x0_abs / width)  * 1000)
        y0_rel = int((y0_abs / height) * 1000)
        x1_rel = int((x1_abs / width)  * 1000)
        y1_rel = int((y1_abs / height) * 1000)
        relative_points.append([x0_rel, y0_rel, x1_rel, y1_rel])
    return relative_points

def relative_to_absolute_coords(
        relative_coords: list[list[int]], 
        width: list, 
        height: list
        ):
    absolute_points = []
    for coord in relative_coords:
        if coord:
            x0_rel, y0_rel, x1_rel, y1_rel = coord
            x0_abs = int((x0_rel / 1000) * width)
            y0_abs = int((y0_rel / 1000) * height)
            x1_abs = int((x1_rel / 1000) * width)
            y1_abs = int((y1_rel / 1000) * height)
            absolute_points.append([x0_abs, y0_abs, x1_abs, y1_abs])
    return absolute_points

def draw_bbox(
        image: Image, 
        bboxes: list[list[int]],
        labels: list[str] | None = None,
        line_color = "red", 
        line_width = 2, 
        offset: Literal["factor", "pixel"] = "pixel", 
        offset_value = 10
    ):

    draw = ImageDraw.Draw(image)
    for i, bbox in enumerate(bboxes):
        x0_abs, y0_abs, x1_abs, y1_abs = bbox

        if offset == "factor":
            center_x = (x0_abs + x1_abs) / 2.0
            center_y = (y0_abs + y1_abs) / 2.0
            w_new = (x1_abs - x0_abs) * (1 + offset_value)
            h_new = (y1_abs - y0_abs) * (1 + offset_value)
            x0_new = center_x - w_new / 2.0
            y0_new = center_y - h_new / 2.0
            x1_new = center_x + w_new / 2.0
            y1_new = center_y + h_new / 2.0
        else:
            x0_new = x0_abs - offset_value
            y0_new = y0_abs - offset_value
            x1_new = x1_abs + offset_value
            y1_new = y1_abs + offset_value

        x0_abs = int(max(0, x0_new))
        y0_abs = int(max(0, y0_new))
        x1_abs = int(min(image.width,  x1_new))
        y1_abs = int(min(image.height, y1_new))
        draw.rectangle([(x0_abs, y0_abs), (x1_abs, y1_abs)], outline = line_color, width = line_width)

        if labels:
            draw.text((x0_abs, y0_abs), text = labels[i], fill = line_color, font_size = 20)
    return image