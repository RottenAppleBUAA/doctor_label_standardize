import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import torch

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.layers import batched_nms
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from config.add_cfg import add_ssl_config
from mask2former import add_maskformer2_config


TEETH_32_CLASSES = [
    "1",
    "2",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "30",
    "32",
    "18",
    "29",
    "3",
    "14",
    "31",
]

TEETH_32_COLORS = [
    [220, 20, 60],
    [119, 11, 32],
    [0, 0, 142],
    [0, 0, 230],
    [106, 0, 228],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 70],
    [0, 0, 192],
    [250, 170, 30],
    [100, 170, 30],
    [100, 170, 30],
    [220, 220, 0],
    [175, 116, 175],
    [250, 0, 30],
    [165, 42, 42],
    [255, 77, 255],
    [0, 226, 252],
    [182, 182, 255],
    [0, 82, 0],
    [120, 166, 157],
    [110, 76, 0],
    [174, 57, 255],
    [199, 100, 0],
    [72, 0, 118],
    [72, 0, 118],
    [255, 179, 240],
    [0, 125, 92],
    [209, 0, 151],
    [188, 208, 182],
    [0, 220, 176],
    [255, 99, 164],
]


def get_parser():
    parser = argparse.ArgumentParser(description="Run SemiT-SAM inference on local images.")
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        help="Config file path.",
    )
    parser.add_argument("--weights", required=True, help="Path to model weights.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images.")
    parser.add_argument(
        "--output-dir",
        default="demo_output",
        help="Directory to save visualized outputs and prediction summary.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=32,
        help="Number of foreground classes expected by the checkpoint.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum score to keep an instance in saved results. Default keeps all raw predictions.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=-1.0,
        help="IoU threshold for per-class NMS. Set < 0 to disable NMS. Default disables NMS.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=-1,
        help="Maximum detections to keep after score filtering and NMS. Default keeps all detections.",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS instead of per-class NMS.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.70,
        help="Suppress a lower-score box when overlap is above this IoU and centers are very close.",
    )
    parser.add_argument(
        "--duplicate-center-threshold",
        type=float,
        default=0.18,
        help="Normalized center-distance threshold for duplicate suppression.",
    )
    parser.add_argument(
        "--duplicate-containment-threshold",
        type=float,
        default=0.88,
        help="Suppress a lower-score box when it is mostly contained in a larger overlapping box.",
    )
    parser.add_argument(
        "--duplicate-very-high-iou-threshold",
        type=float,
        default=0.88,
        help="Suppress one of two boxes directly when their IoU is extremely high.",
    )
    parser.add_argument(
        "--draw-masks",
        action="store_true",
        help="Draw masks as well. Default behavior is box-only visualization.",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save raw boxes, filtered boxes, and a side-by-side comparison.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of images to process from the input directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on, for example cuda or cpu.",
    )
    return parser


def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ssl_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = args.num_classes
    cfg.SSL.TRAIN_SSL = False
    cfg.SSL.EVAL_WHO = "Teacher"
    cfg.freeze()
    return cfg


def strip_teacher_student_prefix(state_dict):
    stripped = {}
    prefixes = ("modelTeacher.", "modelStudent.")
    changed = False
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                changed = True
                break
        stripped[new_key] = value
    return stripped, changed


def load_model(cfg):
    build_cfg_for_model = cfg
    target_device = str(cfg.MODEL.DEVICE).lower()
    if target_device == "cuda":
        # Build on CPU first. On this WSL setup, letting detectron2 build
        # directly on CUDA can leave CUDA in a bad state before model.to().
        build_cfg_for_model = copy.deepcopy(cfg)
        build_cfg_for_model.defrost()
        build_cfg_for_model.MODEL.DEVICE = "cpu"
        build_cfg_for_model.freeze()

    model = build_model(build_cfg_for_model)
    if target_device == "cuda":
        # Force a tiny CUDA allocation before moving the full model so CUDA
        # runtime initialization completes cleanly.
        _ = torch.tensor([0.0], device="cuda")
        model = model.to(torch.device("cuda"))
    model.eval()

    checkpoint = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        state_dict, stripped = strip_teacher_student_prefix(state_dict)
        if stripped:
            incompatible = model.load_state_dict(state_dict, strict=False)
        else:
            incompatible = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            return model, incompatible
    else:
        incompatible = model.load_state_dict(checkpoint, strict=False)

    return model, incompatible


def list_images(input_dir, limit):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in sorted(Path(input_dir).iterdir()) if p.suffix in image_exts]
    if limit > 0:
        paths = paths[:limit]
    return paths


def build_metadata(name, num_classes):
    metadata = MetadataCatalog.get(name)
    if num_classes == 32:
        metadata.thing_classes = TEETH_32_CLASSES
        metadata.thing_colors = TEETH_32_COLORS
    else:
        metadata.thing_classes = [str(i + 1) for i in range(num_classes)]
        metadata.thing_colors = [[(37 * i) % 255, (97 * i) % 255, (167 * i) % 255] for i in range(num_classes)]
    return metadata


def resize_image(cfg, image):
    min_size = cfg.INPUT.MIN_SIZE_TEST
    aug = T.Resize((min_size, min_size))
    return aug.get_transform(image).apply_image(image)


def predict(model, cfg, image_bgr):
    with torch.no_grad():
        original = image_bgr
        if cfg.INPUT.FORMAT == "RGB":
            original = original[:, :, ::-1]
        height, width = original.shape[:2]
        image = resize_image(cfg, original)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return model([inputs])[0]


def serialize_instances(instances, score_threshold):
    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
    scores = instances.scores.tolist() if instances.has("scores") else []
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

    results = []
    for box, score, klass in zip(boxes, scores, classes):
        if score < score_threshold:
            continue
        results.append(
            {
                "bbox_xyxy": [round(x, 2) for x in box],
                "score": round(float(score), 4),
                "class_id": int(klass),
            }
        )
    return results


def filter_instances(instances, score_threshold, nms_threshold, max_detections, agnostic_nms):
    instances = instances.to("cpu")
    if not instances.has("scores") or len(instances) == 0:
        return instances

    keep = instances.scores >= score_threshold
    instances = instances[keep]
    if len(instances) == 0:
        return instances

    if nms_threshold is not None and nms_threshold >= 0 and instances.has("pred_boxes"):
        idxs = torch.zeros_like(instances.pred_classes) if agnostic_nms else instances.pred_classes
        keep = batched_nms(
            instances.pred_boxes.tensor,
            instances.scores,
            idxs,
            nms_threshold,
        )
        instances = instances[keep]

    if max_detections > 0 and len(instances) > max_detections:
        order = torch.argsort(instances.scores, descending=True)[:max_detections]
        instances = instances[order]

    return instances


def filter_by_score_and_limit(instances, score_threshold, max_detections):
    instances = instances.to("cpu")
    if not instances.has("scores") or len(instances) == 0:
        return instances

    keep = instances.scores >= score_threshold
    instances = instances[keep]
    if len(instances) == 0:
        return instances

    if max_detections > 0 and len(instances) > max_detections:
        order = torch.argsort(instances.scores, descending=True)[:max_detections]
        instances = instances[order]
    return instances


def box_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def intersection_area(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    return inter_w * inter_h


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def normalized_center_distance(box_a, box_b):
    cx_a = (float(box_a[0]) + float(box_a[2])) * 0.5
    cy_a = (float(box_a[1]) + float(box_a[3])) * 0.5
    cx_b = (float(box_b[0]) + float(box_b[2])) * 0.5
    cy_b = (float(box_b[1]) + float(box_b[3])) * 0.5
    dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5

    diag_a = max(1.0, ((float(box_a[2] - box_a[0])) ** 2 + (float(box_a[3] - box_a[1])) ** 2) ** 0.5)
    diag_b = max(1.0, ((float(box_b[2] - box_b[0])) ** 2 + (float(box_b[3] - box_b[1])) ** 2) ** 0.5)
    return dist / max(1.0, min(diag_a, diag_b))


def suppress_duplicate_boxes(
    instances,
    duplicate_iou_threshold,
    duplicate_center_threshold,
    duplicate_containment_threshold,
    duplicate_very_high_iou_threshold,
):
    instances = instances.to("cpu")
    if len(instances) <= 1 or not instances.has("pred_boxes"):
        return instances

    order = sorted(
        range(len(instances)),
        key=lambda idx: (float(instances.scores[idx]), float(instances.pred_boxes.area()[idx])),
        reverse=True,
    )
    boxes = instances.pred_boxes.tensor
    keep = []
    for idx in order:
        candidate = boxes[idx]
        is_duplicate = False
        for kept_idx in keep:
            kept_box = boxes[kept_idx]
            iou = box_iou(candidate, kept_box)
            inter = intersection_area(candidate, kept_box)
            min_area = max(1.0, min(box_area(candidate), box_area(kept_box)))
            # Containment is intentionally softer than strict inclusion so that
            # slightly exposed small boxes can still be treated as duplicates.
            containment = inter / min_area
            old_duplicate = (
                iou >= duplicate_iou_threshold
                and normalized_center_distance(candidate, kept_box) <= duplicate_center_threshold
            )
            contained_duplicate = containment >= duplicate_containment_threshold
            very_high_iou_duplicate = iou >= duplicate_very_high_iou_threshold
            if (
                old_duplicate
                or contained_duplicate
                or very_high_iou_duplicate
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(idx)

    keep_tensor = torch.as_tensor(keep, dtype=torch.long)
    return instances[keep_tensor]


def draw_box_only(metadata, image_bgr, instances):
    image = image_bgr.copy()
    if instances is None or len(instances) == 0:
        return image

    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    for box, score, klass in zip(boxes, scores, classes):
        color = metadata.thing_colors[int(klass) % len(metadata.thing_colors)]
        color = tuple(int(c) for c in color)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = metadata.thing_classes[int(klass)] if int(klass) < len(metadata.thing_classes) else str(int(klass))
        label = f"{class_name}:{score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(text_h + 4, y1)
        cv2.rectangle(
            image,
            (x1, text_y - text_h - 6),
            (x1 + text_w + 6, text_y + baseline - 6),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def save_comparison_image(raw_image, filtered_image, output_path):
    gap = 20
    height = max(raw_image.shape[0], filtered_image.shape[0]) + 50
    width = raw_image.shape[1] + filtered_image.shape[1] + gap
    canvas = 255 * torch.ones((height, width, 3), dtype=torch.uint8).numpy()

    canvas[50 : 50 + raw_image.shape[0], : raw_image.shape[1]] = raw_image
    x_offset = raw_image.shape[1] + gap
    canvas[50 : 50 + filtered_image.shape[0], x_offset : x_offset + filtered_image.shape[1]] = filtered_image

    cv2.putText(canvas, "Raw Boxes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Filtered Boxes", (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas)


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = Path(args.output_dir) / "raw_boxes"
    filtered_dir = Path(args.output_dir) / "filtered_boxes"
    compare_dir = Path(args.output_dir) / "comparison"
    if args.save_comparison:
        raw_dir.mkdir(parents=True, exist_ok=True)
        filtered_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    model, incompatible = load_model(cfg)
    metadata = build_metadata("__semi_t_sam_demo__", args.num_classes)

    print(f"Loaded weights: {args.weights}")
    if hasattr(incompatible, "missing_keys") and incompatible.missing_keys:
        print(f"Missing keys: {len(incompatible.missing_keys)}")
    if hasattr(incompatible, "unexpected_keys") and incompatible.unexpected_keys:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

    image_paths = list_images(args.input_dir, args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    summary = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        predictions = predict(model, cfg, image)
        instances = predictions["instances"] if "instances" in predictions else None
        result = {
            "image": image_path.name,
            "num_instances": 0,
            "instances": [],
        }
        if instances is not None:
            raw_instances = instances.to("cpu")
            filtered_instances = filter_instances(
                raw_instances,
                args.score_threshold,
                args.nms_threshold,
                args.max_detections,
                args.agnostic_nms,
            )
            filtered_instances = suppress_duplicate_boxes(
                filtered_instances,
                args.duplicate_iou_threshold,
                args.duplicate_center_threshold,
                args.duplicate_containment_threshold,
                args.duplicate_very_high_iou_threshold,
            )
            predictions["instances"] = filtered_instances
            serialized = serialize_instances(filtered_instances, args.score_threshold)
            result["num_instances"] = len(serialized)
            result["instances"] = serialized
        else:
            raw_instances = None
            filtered_instances = None

        raw_vis_image = draw_box_only(metadata, image, raw_instances)
        vis_image = draw_box_only(metadata, image, filtered_instances)
        output_path = Path(args.output_dir) / image_path.name
        cv2.imwrite(str(output_path), vis_image)
        if args.save_comparison:
            cv2.imwrite(str(raw_dir / image_path.name), raw_vis_image)
            cv2.imwrite(str(filtered_dir / image_path.name), vis_image)
            save_comparison_image(raw_vis_image, vis_image, compare_dir / image_path.name)
        summary.append(result)
        print(f"{image_path.name}: kept {result['num_instances']} instances")

    summary_path = Path(args.output_dir) / "predictions.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved visualizations to {args.output_dir}")
    print(f"Saved prediction summary to {summary_path}")


if __name__ == "__main__":
    main()
