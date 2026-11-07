"""Small numeric SVG plots using only escaped XML and the standard library."""

import math
import xml.etree.ElementTree as ET

PALETTE = ("#126b76", "#c55a32", "#46659a", "#85723a", "#477953", "#925c70")
INK = "#23333b"
PAPER = "#faf7f0"


def _text(root, x, y, text, size=12, color=INK):
    node = ET.SubElement(root, "text", {
        "x": str(x), "y": str(y), "fill": color, "font-size": str(size),
        "font-family": "Consolas, monospace",
    })
    node.text = str(text)


def _canvas(title, subtitle, height=390):
    root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 900 {height}",
        "width": "900", "height": str(height), "role": "img",
    })
    ET.SubElement(root, "title").text = title
    ET.SubElement(root, "rect", {"width": "900", "height": str(height), "fill": PAPER})
    _text(root, 28, 34, title, 20)
    _text(root, 28, 58, subtitle, 11)
    return root


def _save(root, path):
    with path.open("xb") as handle:
        handle.write(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return path


def line_chart(path, title, series, *, x_label="index", y_label="value", x_start=0):
    if not series or len(series) > len(PALETTE):
        raise ValueError("Supply one to six numeric series.")
    values = [float(value) for curve in series.values() for value in curve]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Plot values must be nonempty and finite.")
    if any(not 1 <= len(curve) <= 4096 for curve in series.values()):
        raise ValueError("Each curve needs 1-4096 values.")
    root = _canvas(title, y_label)
    left, top, width, height = 85, 88, 760, 215
    minimum, maximum = min(0.0, min(values)), max(0.0, max(values))
    if minimum == maximum:
        maximum = minimum + 1
    count = max(len(curve) for curve in series.values())
    for tick in range(5):
        y = top + height * tick / 4
        value = maximum - (maximum - minimum) * tick / 4
        ET.SubElement(root, "line", {
            "x1": str(left), "x2": str(left + width), "y1": str(y), "y2": str(y),
            "stroke": "#d9ded8", "stroke-width": "1",
        })
        _text(root, 10, y + 4, f"{value:.3g}", 11)
        _text(root, left + width * tick / 4 - 8, top + height + 21,
              f"{x_start + (count - 1) * tick / 4:.0f}", 11)
    for index, (name, curve) in enumerate(series.items()):
        points = " ".join(
            f"{left + i * width / max(1, count - 1):.2f},"
            f"{top + height * (maximum - float(value)) / (maximum - minimum):.2f}"
            for i, value in enumerate(curve)
        )
        ET.SubElement(root, "polyline", {
            "points": points, "fill": "none", "stroke": PALETTE[index], "stroke-width": "1.8",
        })
        _text(root, 30 + index * 140, 375, name, 11, PALETTE[index])
    _text(root, left, 345, x_label, 12)
    return _save(root, path)


def heatmap(path, title, matrix):
    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if not 1 <= rows <= 128 or not 1 <= columns <= 128 or any(len(row) != columns for row in matrix):
        raise ValueError("Heatmaps require a rectangular 1-128 by 1-128 matrix.")
    values = [float(value) for row in matrix for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Heatmap values must be finite.")
    limit = max(abs(value) for value in values) or 1.0
    root = _canvas(title, "Row means over 16 residual dimensions; each plot has its own color scale.", 590)
    left, top, width, height = 100, 92, 700, 400
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            fraction = abs(float(value)) / limit
            target = (197, 90, 50) if value >= 0 else (18, 107, 118)
            color = "#%02x%02x%02x" % tuple(round(255 + (channel - 255) * fraction) for channel in target)
            ET.SubElement(root, "rect", {
                "x": f"{left + column_index * width / columns:.2f}",
                "y": f"{top + row_index * height / rows:.2f}",
                "width": f"{width / columns + 0.1:.2f}", "height": f"{height / rows + 0.1:.2f}",
                "fill": color,
            })
    _text(root, 100, 518, f"Changed-column ordinal: 0 to {columns - 1} (not original column index)")
    _text(root, 100, 542, f"Top to bottom: {rows} row bins. Scale: {-limit:.4g} to {limit:.4g}.")
    _text(root, 100, 566, "Teal = negative change; orange = positive change.", 11)
    return _save(root, path)
