#!/usr/bin/env python3

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
INPUT_JSON = DATA_ROOT / "standardized_annotations" / "normalized_annotations.json"
IMAGE_DIR = Path("/home/tianruiliu/codespace/data_process/586份数据20260116/586张原图")
OUTPUT_DIR = DATA_ROOT / "standardized_annotations" / "overlays"

HEADER_HEIGHT = 110
TEXT_PAD_X = 6
TEXT_PAD_Y = 4

PALETTE = [
    "#D7263D",
    "#1B998B",
    "#F46036",
    "#2E294E",
    "#C5D86D",
    "#33658A",
    "#7C3AED",
    "#EF476F",
    "#118AB2",
    "#8D99AE",
    "#FB8500",
    "#2A9D8F",
    "#6A4C93",
]


def load_payload() -> dict:
    with INPUT_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def filename_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def draw_label(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle(
        (
            bbox[0] - TEXT_PAD_X,
            bbox[1] - TEXT_PAD_Y,
            bbox[2] + TEXT_PAD_X,
            bbox[3] + TEXT_PAD_Y,
        ),
        radius=6,
        fill=(255, 255, 255, 175),
        outline=color + (230,),
        width=2,
    )
    draw.text((x, y), text, fill=color + (255,), font=font)


def draw_header(canvas: Image.Image, image_name: str, ann_count: int, title_font: ImageFont.ImageFont, body_font: ImageFont.ImageFont) -> None:
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill=(248, 250, 252, 255))
    draw.text((24, 18), "标准化标注可视化", fill=(17, 24, 39, 255), font=title_font)
    draw.text((24, 52), image_name, fill=(31, 41, 55, 255), font=body_font)
    draw.text((24, 78), f"标准框数量: {ann_count}  |  每颗牙最多一个框  |  标签冲突保留在同一标准框上", fill=(55, 65, 81, 255), font=body_font)


def build_color_map(images: list[dict]) -> dict[str, str]:
    tooth_ids = sorted(
        {ann["tooth_id"] for image in images for ann in image["standardized_annotations"]},
        key=lambda value: int(value),
    )
    return {tooth_id: PALETTE[idx % len(PALETTE)] for idx, tooth_id in enumerate(tooth_ids)}


def render_one(image_name: str, annotations: list[dict], color_map: dict[str, str], title_font: ImageFont.ImageFont, body_font: ImageFont.ImageFont, mono_font: ImageFont.ImageFont) -> Image.Image:
    image_path = IMAGE_DIR / image_name
    original = Image.open(image_path).convert("RGBA")
    canvas = Image.new("RGBA", (original.width, original.height + HEADER_HEIGHT), (255, 255, 255, 255))
    canvas.paste(original, (0, HEADER_HEIGHT))
    draw_header(canvas, image_name, len(annotations), title_font, body_font)

    draw = ImageDraw.Draw(canvas, "RGBA")
    for ann in sorted(annotations, key=lambda item: int(item["tooth_id"])):
        color_rgb = hex_to_rgb(color_map[ann["tooth_id"]])
        x1, y1, x2, y2 = ann["standard_box_xyxy"]
        box = (x1, HEADER_HEIGHT + y1, x2, HEADER_HEIGHT + y2)
        draw.rectangle(box, outline=color_rgb + (230,), width=4)

        labels = ",".join(ann["labels"])
        annotators = ",".join(ann["annotators"])
        text = f"T{ann['tooth_id']} | {labels} | {annotators}"
        label_x = max(8, x1 + 4)
        label_y = max(HEADER_HEIGHT + 8, HEADER_HEIGHT + y1 + 4)
        draw_label(draw, label_x, label_y, text, color_rgb, mono_font)

    draw.rectangle(
        (1, HEADER_HEIGHT + 1, original.width - 2, HEADER_HEIGHT + original.height - 2),
        outline=(148, 163, 184, 255),
        width=2,
    )
    return canvas


def build_index(file_names: list[str]) -> str:
    cards = []
    for name in file_names:
        png_name = f"{Path(name).stem}.png"
        cards.append(
            f"""
            <article class="card">
              <a href="./pngs/{html.escape(png_name)}" target="_blank">
                <img src="./pngs/{html.escape(png_name)}" alt="{html.escape(name)}" loading="lazy" />
              </a>
              <div class="meta">
                <strong>{html.escape(name)}</strong>
                <a href="./pngs/{html.escape(png_name)}" target="_blank">打开大图</a>
              </div>
            </article>
            """
        )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>标准化标注可视化</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --card: #fffdf8;
      --ink: #1f2937;
      --line: #d6d3d1;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(217,119,6,0.1), transparent 24%),
        var(--bg);
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
      background: rgba(255,255,255,0.7);
      backdrop-filter: blur(6px);
      position: sticky;
      top: 0;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    p {{
      margin: 0;
      max-width: 980px;
      line-height: 1.55;
    }}
    main {{
      padding: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }}
    .card a {{
      color: var(--accent);
      text-decoration: none;
    }}
    img {{
      display: block;
      width: 100%;
      height: 220px;
      object-fit: cover;
      background: #e5e7eb;
    }}
    .meta {{
      padding: 14px 16px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    strong {{
      font-family: Consolas, monospace;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>标准化标注可视化</h1>
    <p>每张图展示标准化后的唯一牙位框。框标签格式为 `T牙位 | 标签 | 标注人`。</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    payload = load_payload()
    images = sorted(payload["images"], key=lambda item: filename_sort_key(item["file_name"]))
    color_map = build_color_map(images)

    png_dir = OUTPUT_DIR / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    title_font = load_font(24)
    body_font = load_font(16)
    mono_font = load_font(16)

    rendered = []
    missing = []
    for image_data in images:
        image_name = image_data["file_name"]
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            missing.append(image_name)
            continue
        out = render_one(
            image_name,
            image_data["standardized_annotations"],
            color_map,
            title_font,
            body_font,
            mono_font,
        )
        out.convert("RGB").save(png_dir / f"{Path(image_name).stem}.png", quality=95)
        rendered.append(image_name)

    (OUTPUT_DIR / "index.html").write_text(build_index(rendered), encoding="utf-8")

    lines = [
        "# 标准化标注可视化输出",
        "",
        f"- 输入 JSON：`{INPUT_JSON}`",
        f"- 输出目录：`{OUTPUT_DIR}`",
        f"- 总图片数：{len(images)}",
        f"- 成功生成：{len(rendered)}",
        f"- 缺失原图：{len(missing)}",
        "",
        "## 说明",
        "",
        "- 每颗牙最多只保留一个标准框。",
        "- 标签冲突会合并到同一个标准框里显示。",
        "- 图上文本格式：`T牙位 | 标签 | 标注人`。",
        "",
    ]
    if missing:
        lines.extend(["## 缺失原图", ""])
        for name in missing:
            lines.append(f"- `{name}`")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Generated {len(rendered)} standardized overlay PNG files in {png_dir}")
    print(f"Index: {OUTPUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
