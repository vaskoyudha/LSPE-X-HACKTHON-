#!/usr/bin/env python3
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "proposal" / "figures"

FONT_REGULAR_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PALETTE = {
    "bg": "#F7F8FC",
    "title": "#0F172A",
    "text": "#334155",
    "muted": "#64748B",
    "stroke": "#94A3B8",
    "connector": "#475569",
    "data": "#E0F2FE",
    "process": "#EDE9FE",
    "model": "#FEF3C7",
    "explain": "#DCFCE7",
    "decision": "#FCE7F3",
    "risk_low": "#DCFCE7",
    "risk_watch": "#FEF3C7",
    "risk_high": "#FED7AA",
    "risk_critical": "#FECACA",
    "shadow": "#D7DCEA",
}

@dataclass
class Node:
    id: str
    x: int
    y: int
    w: int
    h: int
    label: str
    fill: str
    stroke: str = PALETTE["stroke"]
    text_color: str = PALETTE["title"]
    kind: Literal["box", "note"] = "box"

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

@dataclass
class Edge:
    start: str
    end: str
    label: str | None = None
    style: Literal["solid", "dashed"] = "solid"
    color: str = PALETTE["connector"]
    elbow: Literal["horizontal", "vertical", "straight"] = "horizontal"

@dataclass
class Badge:
    text: str
    fill: str
    text_color: str = PALETTE["title"]

@dataclass
class Diagram:
    slug: str
    title: str
    subtitle: str
    mermaid: str
    width: int
    height: int
    nodes: list[Node]
    edges: list[Edge]
    badges: list[Badge] = field(default_factory=list)
    footer: str = "LPSE-X | Find IT! 2026 Track C | Generated offline"


FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = ("bold" if bold else "regular", size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_REGULAR_PATH, size)
    return FONT_CACHE[key]


def estimate_text_width(text: str, fnt: ImageFont.FreeTypeFont) -> float:
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    return draw.textlength(text, font=fnt)


def split_long_token(token: str) -> list[str]:
    if len(token) <= 18:
        return [token]
    parts: list[str] = []
    current = ""
    for char in token:
        current += char
        if char in "_/-" and len(current) >= 10:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts or [token]


def wrap_text(text: str, max_width: int, fnt: ImageFont.FreeTypeFont) -> list[str]:
    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        tokens: list[str] = []
        for raw in paragraph.split():
            if estimate_text_width(raw, fnt) > max_width:
                tokens.extend(split_long_token(raw))
            else:
                tokens.append(raw)
        if not tokens:
            lines.append("")
            continue
        current = tokens[0]
        for token in tokens[1:]:
            trial = f"{current} {token}" if current else token
            if estimate_text_width(trial, fnt) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = token
        lines.append(current)
    return lines or [""]


def hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def svg_multiline_text(x: float, y: float, lines: Iterable[str], *, size: int, color: str, weight: str = "400", anchor: str = "middle", line_height: float = 1.32) -> str:
    tspans = []
    dy = 0
    for idx, line in enumerate(lines):
        escaped = html.escape(line)
        if idx == 0:
            tspans.append(f'<tspan x="{x}" dy="0">{escaped}</tspan>')
        else:
            dy = size * line_height
            tspans.append(f'<tspan x="{x}" dy="{dy}">{escaped}</tspan>')
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}">' + "".join(tspans) + "</text>"
    )


def node_anchor(node: Node, other: Node, axis: Literal["horizontal", "vertical", "straight"]) -> tuple[float, float, float, float]:
    sx, sy = node.center
    ex, ey = other.center
    if axis == "vertical":
        if ey >= sy:
            start = (sx, node.y + node.h)
            end = (ex, other.y)
        else:
            start = (sx, node.y)
            end = (ex, other.y + other.h)
    elif axis == "straight":
        angle = math.atan2(ey - sy, ex - sx)
        start = (sx + math.cos(angle) * node.w / 2.2, sy + math.sin(angle) * node.h / 2.2)
        end = (ex - math.cos(angle) * other.w / 2.2, ey - math.sin(angle) * other.h / 2.2)
    else:
        if ex >= sx:
            start = (node.x + node.w, sy)
            end = (other.x, ey)
        else:
            start = (node.x, sy)
            end = (other.x + other.w, ey)
    return start[0], start[1], end[0], end[1]


def orthogonal_path(start: tuple[float, float], end: tuple[float, float], mode: Literal["horizontal", "vertical", "straight"]) -> list[tuple[float, float]]:
    sx, sy = start
    ex, ey = end
    if mode == "straight":
        return [start, end]
    if mode == "vertical":
        mid_y = (sy + ey) / 2
        return [start, (sx, mid_y), (ex, mid_y), end]
    mid_x = (sx + ex) / 2
    return [start, (mid_x, sy), (mid_x, ey), end]


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], *, color: str, dashed: bool = False) -> None:
    if dashed:
        for a, b in zip(points, points[1:]):
            draw_dashed_segment(draw, a, b, color=color, width=4)
    else:
        draw.line(points, fill=color, width=4, joint="curve")
    a, b = points[-2], points[-1]
    arrow = arrow_head(a, b, size=12)
    draw.polygon(arrow, fill=color)


def draw_dashed_segment(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], *, color: str, width: int) -> None:
    sx, sy = start
    ex, ey = end
    distance = math.dist(start, end)
    if distance == 0:
        return
    dash = 12
    gap = 8
    steps = int(distance // (dash + gap)) + 1
    ux = (ex - sx) / distance
    uy = (ey - sy) / distance
    cursor = 0.0
    for _ in range(steps):
        a = (sx + ux * cursor, sy + uy * cursor)
        cursor = min(cursor + dash, distance)
        b = (sx + ux * cursor, sy + uy * cursor)
        draw.line([a, b], fill=color, width=width)
        cursor += gap
        if cursor >= distance:
            break


def arrow_head(a: tuple[float, float], b: tuple[float, float], size: int) -> list[tuple[float, float]]:
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    left = (b[0] - size * math.cos(angle - math.pi / 6), b[1] - size * math.sin(angle - math.pi / 6))
    right = (b[0] - size * math.cos(angle + math.pi / 6), b[1] - size * math.sin(angle + math.pi / 6))
    return [b, left, right]


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if idx == 0 else "L") + f" {x:.1f} {y:.1f}" for idx, (x, y) in enumerate(points))


def edge_label_position(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        return points[0]
    mid = len(points) // 2
    a = points[mid - 1]
    b = points[mid]
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 10)


def render_png(diagram: Diagram, output: Path) -> None:
    image = Image.new("RGBA", (diagram.width, diagram.height), hex_to_rgba(PALETTE["bg"]))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((24, 24, diagram.width - 24, diagram.height - 24), radius=28, fill="#FFFFFF", outline=PALETTE["shadow"], width=2)
    draw.text((56, 46), diagram.title, font=font(30, bold=True), fill=PALETTE["title"])
    draw.text((56, 90), diagram.subtitle, font=font(15), fill=PALETTE["muted"])

    badge_x = diagram.width - 58
    for badge in reversed(diagram.badges):
        fnt = font(13, bold=True)
        pad_x = 14
        text_w = estimate_text_width(badge.text, fnt)
        width = int(text_w + pad_x * 2)
        badge_x -= width
        draw.rounded_rectangle((badge_x, 46, badge_x + width, 74), radius=14, fill=badge.fill, outline=badge.fill)
        draw.text((badge_x + pad_x, 53), badge.text, font=fnt, fill=badge.text_color)
        badge_x -= 10

    for edge in diagram.edges:
        start_node = next(node for node in diagram.nodes if node.id == edge.start)
        end_node = next(node for node in diagram.nodes if node.id == edge.end)
        sx, sy, ex, ey = node_anchor(start_node, end_node, edge.elbow)
        points = orthogonal_path((sx, sy), (ex, ey), edge.elbow)
        draw_arrow(draw, points, color=edge.color, dashed=edge.style == "dashed")
        if edge.label:
            lx, ly = edge_label_position(points)
            fnt = font(12, bold=True)
            lines = wrap_text(edge.label, 180, fnt)
            box_w = max(estimate_text_width(line, fnt) for line in lines) + 16
            box_h = 12 + len(lines) * 16
            draw.rounded_rectangle((lx - box_w / 2, ly - box_h + 2, lx + box_w / 2, ly + 10), radius=10, fill="#FFFFFF", outline=PALETTE["shadow"], width=1)
            ty = ly - box_h + 10
            for line in lines:
                tw = estimate_text_width(line, fnt)
                draw.text((lx - tw / 2, ty), line, font=fnt, fill=edge.color)
                ty += 16

    for node in diagram.nodes:
        shadow = (node.x + 4, node.y + 6, node.x + node.w + 4, node.y + node.h + 6)
        draw.rounded_rectangle(shadow, radius=22, fill=hex_to_rgba(PALETTE["shadow"], 90))
        if node.kind == "note":
            draw.rounded_rectangle((node.x, node.y, node.x + node.w, node.y + node.h), radius=22, fill=node.fill, outline=node.stroke, width=2)
        else:
            draw.rounded_rectangle((node.x, node.y, node.x + node.w, node.y + node.h), radius=22, fill=node.fill, outline=node.stroke, width=3)
        title_font = font(15, bold=True)
        lines = wrap_text(node.label, node.w - 30, title_font)
        line_gap = 8
        line_height = 20
        total_height = len(lines) * line_height + (len(lines) - 1) * 2
        start_y = node.y + (node.h - total_height) / 2
        for idx, line in enumerate(lines):
            tw = estimate_text_width(line, title_font)
            draw.text((node.x + (node.w - tw) / 2, start_y + idx * (line_height + 2)), line, font=title_font, fill=node.text_color)

    draw.text((56, diagram.height - 56), diagram.footer, font=font(13), fill=PALETTE["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_svg(diagram: Diagram, output: Path) -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{diagram.width}" height="{diagram.height}" viewBox="0 0 {diagram.width} {diagram.height}">',
        f'<rect width="100%" height="100%" fill="{PALETTE["bg"]}"/>',
        f'<rect x="24" y="24" width="{diagram.width - 48}" height="{diagram.height - 48}" rx="28" fill="#FFFFFF" stroke="{PALETTE["shadow"]}" stroke-width="2"/>',
        f'<text x="56" y="68" fill="{PALETTE["title"]}" font-size="30" font-weight="700" font-family="DejaVu Sans, Arial, sans-serif">{html.escape(diagram.title)}</text>',
        f'<text x="56" y="102" fill="{PALETTE["muted"]}" font-size="15" font-family="DejaVu Sans, Arial, sans-serif">{html.escape(diagram.subtitle)}</text>',
    ]

    badge_x = diagram.width - 58
    for badge in reversed(diagram.badges):
        width = estimate_text_width(badge.text, font(13, bold=True)) + 28
        badge_x -= width
        parts.append(f'<rect x="{badge_x}" y="46" width="{width}" height="28" rx="14" fill="{badge.fill}"/>')
        parts.append(f'<text x="{badge_x + 14}" y="64" fill="{badge.text_color}" font-size="13" font-weight="700" font-family="DejaVu Sans, Arial, sans-serif">{html.escape(badge.text)}</text>')
        badge_x -= 10

    parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" /></marker></defs>')

    for edge in diagram.edges:
        start_node = next(node for node in diagram.nodes if node.id == edge.start)
        end_node = next(node for node in diagram.nodes if node.id == edge.end)
        sx, sy, ex, ey = node_anchor(start_node, end_node, edge.elbow)
        points = orthogonal_path((sx, sy), (ex, ey), edge.elbow)
        dash = ' stroke-dasharray="10 7"' if edge.style == "dashed" else ""
        parts.append(
            f'<path d="{svg_path(points)}" fill="none" stroke="{edge.color}" stroke-width="4" marker-end="url(#arrow)"{dash}/>'
        )
        if edge.label:
            lx, ly = edge_label_position(points)
            label_font = font(12, bold=True)
            lines = wrap_text(edge.label, 180, label_font)
            box_w = max(estimate_text_width(line, label_font) for line in lines) + 18
            box_h = 16 + len(lines) * 16
            parts.append(f'<rect x="{lx - box_w / 2:.1f}" y="{ly - box_h + 2:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="10" fill="#FFFFFF" stroke="{PALETTE["shadow"]}"/>')
            parts.append(svg_multiline_text(lx, ly - box_h + 16, lines, size=12, color=edge.color, weight="700"))

    for node in diagram.nodes:
        parts.append(f'<rect x="{node.x + 4}" y="{node.y + 6}" width="{node.w}" height="{node.h}" rx="22" fill="{PALETTE["shadow"]}" opacity="0.35"/>')
        parts.append(f'<rect x="{node.x}" y="{node.y}" width="{node.w}" height="{node.h}" rx="22" fill="{node.fill}" stroke="{node.stroke}" stroke-width="{2 if node.kind == "note" else 3}"/>')
        lines = wrap_text(node.label, node.w - 30, font(15, bold=True))
        total_height = len(lines) * 20 + (len(lines) - 1) * 2
        text_y = node.y + (node.h - total_height) / 2 + 16
        parts.append(svg_multiline_text(node.x + node.w / 2, text_y, lines, size=15, color=node.text_color, weight="700"))

    parts.append(f'<text x="56" y="{diagram.height - 56}" fill="{PALETTE["muted"]}" font-size="13" font-family="DejaVu Sans, Arial, sans-serif">{html.escape(diagram.footer)}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def save_mermaid(diagram: Diagram, output: Path) -> None:
    output.write_text(diagram.mermaid.strip() + "\n", encoding="utf-8")


def box(id: str, x: int, y: int, w: int, h: int, label: str, fill: str, *, kind: Literal["box", "note"] = "box", stroke: str = PALETTE["stroke"]) -> Node:
    return Node(id=id, x=x, y=y, w=w, h=h, label=label, fill=fill, kind=kind, stroke=stroke)


def diagrams() -> list[Diagram]:
    return [
        Diagram(
            slug="pipeline-architecture",
            title="Arsitektur Pipeline LPSE-X",
            subtitle="Alur end-to-end dari data pengadaan mentah menuju risk score yang explainable untuk triase auditor.",
            mermaid="""
flowchart LR
    A[Data OCDS mentah] --> B[Raw temporal split]
    B --> C[Feature engineering split-aware]
    C --> D[Weak labeling risiko]
    D --> E[Training XGBoost + calibration]
    E --> F[SHAP + narasi Bahasa Indonesia]
    F --> G[Risk lane & audit triage]
    H[Artefak output] --> G
    H --> I[Notebook, model, proposal figures]
""",
            width=1800,
            height=980,
            badges=[Badge("Single-model", PALETTE["model"]), Badge("Offline", PALETTE["explain"]), Badge("Track C", PALETTE["data"])],
            nodes=[
                box("A", 90, 270, 220, 120, "Data OCDS mentah", PALETTE["data"]),
                box("B", 360, 270, 240, 120, "Raw temporal split train/test", PALETTE["data"]),
                box("C", 650, 270, 260, 120, "Feature engineering split-aware", PALETTE["process"]),
                box("D", 960, 270, 240, 120, "Weak labeling risiko", PALETTE["process"]),
                box("E", 1250, 270, 260, 120, "Training XGBoost + calibration", PALETTE["model"]),
                box("F", 1270, 500, 240, 120, "SHAP + narasi Bahasa Indonesia", PALETTE["explain"]),
                box("G", 990, 500, 230, 120, "Risk lane & audit triage", PALETTE["decision"]),
                box("H", 690, 500, 240, 120, "Artefak output reproducible", PALETTE["model"]),
                box("I", 390, 500, 250, 120, "Notebook, model, proposal figures", PALETTE["data"]),
                box("N", 90, 685, 1420, 140, "Kontrol utama: train/test dipisah sebelum preprocessing, inferensi berjalan lokal, dan setiap skor wajib memiliki penjelasan yang dapat dibaca auditor.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
            ],
            edges=[
                Edge("A", "B"),
                Edge("B", "C"),
                Edge("C", "D"),
                Edge("D", "E"),
                Edge("E", "F", elbow="vertical", label="model terkunci"),
                Edge("F", "G"),
                Edge("G", "H"),
                Edge("H", "I"),
            ],
        ),
        Diagram(
            slug="anti-leakage-flow",
            title="Validasi Anti-Leakage dan Data Lineage",
            subtitle="Pemisahan train/test dilakukan pada raw data, lalu seluruh fitting dan tuning hanya terjadi di sisi train.",
            mermaid="""
flowchart TD
    A[Raw OCDS rows] --> B[train_data/raw.parquet]
    A --> C[test_data/raw.parquet]
    B --> D[train-only fit / HPO / calibration]
    D --> E[trained model + temperature scaling]
    C --> F[held-out evaluation only]
    E --> G[inference on test_data]
    F --> G
    G --> H[metrics + explainability evidence]
""",
            width=1680,
            height=1020,
            badges=[Badge("No leakage", PALETTE["risk_critical"]), Badge("Held-out test", PALETTE["risk_watch"])],
            nodes=[
                box("A", 610, 150, 360, 110, "Raw OCDS rows dari benchmark riil", PALETTE["data"]),
                box("B", 250, 340, 320, 110, "train_data/raw.parquet", PALETTE["data"]),
                box("C", 1030, 340, 320, 110, "test_data/raw.parquet", PALETTE["data"]),
                box("D", 210, 545, 400, 120, "fit scaler, HPO, clean-label review, calibration", PALETTE["process"]),
                box("E", 250, 760, 320, 110, "model terkunci + temperature scaling", PALETTE["model"]),
                box("F", 990, 545, 400, 120, "evaluasi held-out final saja", PALETTE["decision"]),
                box("G", 650, 760, 360, 110, "inferensi pada test_data dengan model final", PALETTE["explain"]),
                box("H", 650, 900, 360, 80, "metrics + artefak explainability", PALETTE["decision"]),
                box("N1", 660, 530, 270, 90, "Tidak ada HPO / threshold tuning di test_data.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
                box("N2", 170, 900, 370, 80, "train_data menyimpan semua fitting state yang boleh dipelajari pipeline.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
                box("N3", 1110, 900, 380, 80, "test_data hanya membuktikan generalisasi dan kepatuhan kompetisi.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
            ],
            edges=[
                Edge("A", "B", elbow="vertical"),
                Edge("A", "C", elbow="vertical"),
                Edge("B", "D", elbow="vertical"),
                Edge("D", "E", elbow="vertical"),
                Edge("E", "G"),
                Edge("C", "F", elbow="vertical"),
                Edge("F", "G", label="held-out only"),
                Edge("G", "H", elbow="vertical"),
            ],
        ),
        Diagram(
            slug="inference-flow",
            title="Alur Inference dan Explainability",
            subtitle="Satu baris pengadaan diubah menjadi skor risiko, faktor SHAP, dan narasi Bahasa Indonesia yang actionable.",
            mermaid="""
flowchart LR
    A[Input procurement row] --> B[Feature vector]
    B --> C[Probabilitas risiko]
    C --> D[Top SHAP factors]
    D --> E[Narasi Bahasa Indonesia]
    E --> F[Risk lane & reviewer action]
    C --> G[Thresholding operasional]
    G --> F
""",
            width=1760,
            height=920,
            badges=[Badge("CPU-ready", PALETTE["data"]), Badge("Human-readable", PALETTE["explain"])],
            nodes=[
                box("A", 90, 290, 220, 120, "Input procurement row", PALETTE["data"]),
                box("B", 360, 290, 220, 120, "Feature vector siap model", PALETTE["process"]),
                box("C", 640, 290, 250, 120, "Probabilitas risiko per kelas", PALETTE["model"]),
                box("D", 960, 170, 250, 120, "Top SHAP factors", PALETTE["explain"]),
                box("E", 960, 430, 250, 120, "Narasi Bahasa Indonesia", PALETTE["explain"]),
                box("G", 1260, 290, 230, 120, "Thresholding operasional", PALETTE["decision"]),
                box("F", 1540, 290, 170, 120, "Risk lane & reviewer action", PALETTE["decision"]),
                box("N", 320, 650, 1150, 130, "Output final per record: skor risiko, tiga faktor utama, narasi singkat, dan rekomendasi eskalasi manual bila perlu.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
            ],
            edges=[
                Edge("A", "B"),
                Edge("B", "C"),
                Edge("C", "D", elbow="vertical"),
                Edge("C", "E", elbow="vertical"),
                Edge("C", "G"),
                Edge("D", "E", elbow="vertical", style="dashed"),
                Edge("E", "F"),
                Edge("G", "F"),
            ],
        ),
        Diagram(
            slug="submission-package-map",
            title="Peta Paket Pengumpulan Tahap 2",
            subtitle="Proposal PDF diunggah ke Google Form, sedangkan repo/folder teknis memuat artefak yang harus diperiksa juri.",
            mermaid="""
flowchart TD
    A[Proposal PDF] --> P[Google Form Tahap 2]
    B[training.ipynb] --> R[Repo / cloud folder]
    C[inference.ipynb] --> R
    D[model_risk.ubj + onnx] --> R
    E[train_data] --> R
    F[test_data] --> R
    G[README + figures] --> R
""",
            width=1620,
            height=980,
            badges=[Badge("Judge-safe", PALETTE["risk_watch"]), Badge("Named exactly", PALETTE["data"])],
            nodes=[
                box("P", 1120, 140, 330, 120, "Google Form Tahap 2", PALETTE["decision"]),
                box("A", 230, 140, 350, 120, "Proposal_BismillahFirstTry-Phase2\n_Tahap2_FindIT2026.pdf", PALETTE["model"]),
                box("R", 700, 420, 510, 170, "BismillahFirstTry-Phase2_Tahap2_FindIT2026\n(repo / cloud folder teknis)", PALETTE["data"]),
                box("B", 120, 390, 240, 95, "training.ipynb", PALETTE["process"]),
                box("C", 120, 510, 240, 95, "inference.ipynb", PALETTE["process"]),
                box("D", 1230, 390, 240, 95, "model_risk.ubj + model_risk.onnx", PALETTE["model"]),
                box("E", 1230, 510, 240, 95, "train_data/", PALETTE["data"]),
                box("F", 1230, 630, 240, 95, "test_data/", PALETTE["data"]),
                box("G", 120, 630, 240, 95, "README + figures", PALETTE["explain"]),
                box("N", 280, 800, 1060, 110, "Tujuan packaging: juri dapat memeriksa proposal, notebook, model, dan dataset split tanpa terganggu oleh artefak riset yang tidak wajib.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
            ],
            edges=[
                Edge("A", "P"),
                Edge("B", "R"),
                Edge("C", "R"),
                Edge("D", "R"),
                Edge("E", "R"),
                Edge("F", "R"),
                Edge("G", "R"),
            ],
        ),
        Diagram(
            slug="risk-decision-flow",
            title="Risk Decision Flow untuk Auditor",
            subtitle="Skor model diterjemahkan menjadi jalur tindak lanjut yang konsisten dan mudah dipahami.",
            mermaid="""
flowchart LR
    A[Aman] --> B[Perlu Pantauan]
    B --> C[Risiko Tinggi]
    C --> D[Risiko Kritis]
    D --> E[Eskalasi manual + cek bukti resmi]
""",
            width=1760,
            height=900,
            badges=[Badge("Actionable", PALETTE["decision"]), Badge("Bahasa Indonesia", PALETTE["explain"])],
            nodes=[
                box("A", 100, 320, 250, 140, "Aman\nSkor rendah, simpan sebagai baseline", PALETTE["risk_low"]),
                box("B", 430, 320, 250, 140, "Perlu Pantauan\nMonitor pola dan perubahan mendadak", PALETTE["risk_watch"]),
                box("C", 760, 320, 250, 140, "Risiko Tinggi\nPrioritaskan review manual cepat", PALETTE["risk_high"]),
                box("D", 1090, 320, 250, 140, "Risiko Kritis\nButuh triase auditor dan cek artefak", PALETTE["risk_critical"]),
                box("E", 1420, 320, 230, 140, "Eskalasi manual\nCocokkan dengan bukti resmi", PALETTE["decision"]),
                box("N1", 90, 610, 1560, 120, "Setiap lane menyimpan alasan: skor probabilitas, faktor SHAP teratas, dan narasi auditor-friendly. Semakin tinggi lane, semakin ketat verifikasi bukti resminya.", PALETTE["bg"], kind="note", stroke="#CBD5E1"),
            ],
            edges=[
                Edge("A", "B"),
                Edge("B", "C"),
                Edge("C", "D"),
                Edge("D", "E"),
            ],
        ),
    ]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for diagram in diagrams():
        save_mermaid(diagram, FIGURES_DIR / f"{diagram.slug}.mmd")
        render_svg(diagram, FIGURES_DIR / f"{diagram.slug}.svg")
        render_png(diagram, FIGURES_DIR / f"{diagram.slug}.png")
        print(f"generated {diagram.slug}")


if __name__ == "__main__":
    main()
