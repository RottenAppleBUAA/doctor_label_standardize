import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import torch

from demo_infer import (
    TEETH_32_CLASSES,
    build_cfg,
    filter_instances,
    load_model,
    predict,
    suppress_duplicate_boxes,
)

DATA_ROOT = Path("/home/tianruiliu/codespace/data_process")


@dataclass
class DoctorBox:
    doctor_id: str
    source_file: str
    file_name: str
    label_name: str
    label_id: str
    object_index: int
    image_width: int
    image_height: int
    bbox_pct_xywh: list
    bbox_xyxy: list


def get_parser():
    parser = argparse.ArgumentParser(description="Normalize two-doctor annotations to standard model boxes.")
    parser.add_argument(
        "--annotation-dir",
        default="/home/tianruiliu/codespace/data_process/annotations",
        help="Directory containing the two doctor annotation json files.",
    )
    parser.add_argument(
        "--image-dir",
        default="/home/tianruiliu/codespace/data_process/586份数据20260116/586张原图",
        help="Directory containing source images referenced by the annotations.",
    )
    parser.add_argument(
        "--config-file",
        default="configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "--weights",
        default="pretrained_models/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_ROOT / "standardized_annotations"),
        help="Directory to save normalized outputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model inference, for example cpu or cuda.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=32,
        help="Number of tooth classes expected by the checkpoint.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold used before duplicate suppression.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=-1.0,
        help="NMS IoU threshold. Set < 0 to disable.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=-1,
        help="Maximum detections kept before duplicate suppression. Set < 0 to keep all.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.70,
        help="Duplicate suppression IoU threshold with close-center rule.",
    )
    parser.add_argument(
        "--duplicate-center-threshold",
        type=float,
        default=0.18,
        help="Duplicate suppression center-distance threshold.",
    )
    parser.add_argument(
        "--duplicate-containment-threshold",
        type=float,
        default=0.88,
        help="Duplicate suppression containment threshold.",
    )
    parser.add_argument(
        "--duplicate-very-high-iou-threshold",
        type=float,
        default=0.88,
        help="Duplicate suppression very-high-IoU threshold.",
    )
    parser.add_argument(
        "--single-min-query-cover",
        type=float,
        default=0.28,
        help="Minimum fraction of a single doctor box that must be covered by a standard box.",
    )
    parser.add_argument(
        "--single-min-match-score",
        type=float,
        default=0.35,
        help="Minimum match score for a single doctor box.",
    )
    parser.add_argument(
        "--pair-min-query-cover",
        type=float,
        default=0.55,
        help="Minimum fraction of an overlap region that must be covered by a standard box.",
    )
    parser.add_argument(
        "--pair-min-match-score",
        type=float,
        default=0.55,
        help="Minimum match score for an overlap-region match.",
    )
    parser.add_argument(
        "--max-box-width-percent",
        type=float,
        default=18.0,
        help="Ignore doctor boxes wider than this relative percentage.",
    )
    parser.add_argument(
        "--max-box-height-percent",
        type=float,
        default=24.0,
        help="Ignore doctor boxes taller than this relative percentage.",
    )
    parser.add_argument(
        "--multi-hit-std-cover-threshold",
        type=float,
        default=0.55,
        help="A doctor box is considered multi-tooth if it covers more than one standard box above this threshold.",
    )
    return parser


def read_annotation_files(annotation_dir):
    paths = sorted(Path(annotation_dir).glob("*.json"))
    if len(paths) != 2:
        raise ValueError(f"Expected exactly 2 annotation json files in {annotation_dir}, found {len(paths)}")

    doctor_map = {}
    for idx, path in enumerate(paths, start=1):
        data = json.load(open(path, "r", encoding="utf-8"))
        doctor_map[f"doctor_{idx}"] = {
            "source_file": path.name,
            "files": data.get("files", []),
        }
    return doctor_map


def pct_xywh_to_xyxy(coords, width, height):
    x, y, w, h = [float(v) for v in coords]
    x1 = max(0.0, min(width, x / 100.0 * width))
    y1 = max(0.0, min(height, y / 100.0 * height))
    x2 = max(0.0, min(width, (x + w) / 100.0 * width))
    y2 = max(0.0, min(height, (y + h) / 100.0 * height))
    return [x1, y1, x2, y2]


def xyxy_to_pct_xywh(box, width, height):
    x1, y1, x2, y2 = box
    return [
        round(x1 / width * 100.0, 4),
        round(y1 / height * 100.0, 4),
        round((x2 - x1) / width * 100.0, 4),
        round((y2 - y1) / height * 100.0, 4),
    ]


def box_area(box):
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def intersection_box(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def intersection_area(box_a, box_b):
    inter = intersection_box(box_a, box_b)
    return 0.0 if inter is None else box_area(inter)


def box_iou(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_doctor_boxes(doctor_map):
    by_image = defaultdict(lambda: defaultdict(list))

    for doctor_id, doctor_payload in doctor_map.items():
        source_file = doctor_payload["source_file"]
        for file_entry in doctor_payload["files"]:
            file_name = file_entry["fileName"]
            for label_entry in file_entry.get("labels", []):
                label_name = (label_entry.get("labelName") or "").strip()
                if not label_name:
                    continue

                annotations = label_entry.get("annotations") or {}
                image_info = annotations.get("imageInfo") or {}
                width = int(image_info.get("width") or 0)
                height = int(image_info.get("height") or 0)
                objects = annotations.get("objects") or []
                for object_index, obj in enumerate(objects):
                    if obj.get("type") != "rectangle":
                        continue
                    coords = obj.get("coords")
                    if not coords or len(coords) != 4 or width <= 0 or height <= 0:
                        continue
                    bbox_xyxy = pct_xywh_to_xyxy(coords, width, height)
                    by_image[file_name][doctor_id].append(
                        DoctorBox(
                            doctor_id=doctor_id,
                            source_file=source_file,
                            file_name=file_name,
                            label_name=label_name,
                            label_id=str(label_entry.get("labelId") or ""),
                            object_index=object_index,
                            image_width=width,
                            image_height=height,
                            bbox_pct_xywh=[round(float(v), 4) for v in coords],
                            bbox_xyxy=[round(float(v), 2) for v in bbox_xyxy],
                        )
                    )
    return by_image


def standardize_model_boxes(instances):
    by_tooth = {}
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    boxes = instances.pred_boxes.tensor.tolist()

    order = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)
    for idx in order:
        class_id = int(classes[idx])
        if class_id < 0 or class_id >= len(TEETH_32_CLASSES):
            continue
        tooth_id = TEETH_32_CLASSES[class_id]
        if tooth_id in by_tooth:
            continue
        box = [round(float(v), 2) for v in boxes[idx]]
        by_tooth[tooth_id] = {
            "tooth_id": tooth_id,
            "class_id": class_id,
            "score": round(float(scores[idx]), 6),
            "bbox_xyxy": box,
        }
    return list(by_tooth.values())


def predict_standard_boxes(args, model, cfg, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image {image_path}")

    predictions = predict(model, cfg, image)
    instances = predictions["instances"] if "instances" in predictions else None
    if instances is None:
        return []

    filtered = filter_instances(
        instances.to("cpu"),
        args.score_threshold,
        args.nms_threshold,
        args.max_detections,
        False,
    )
    filtered = suppress_duplicate_boxes(
        filtered,
        args.duplicate_iou_threshold,
        args.duplicate_center_threshold,
        args.duplicate_containment_threshold,
        args.duplicate_very_high_iou_threshold,
    )
    standards = standardize_model_boxes(filtered)
    height, width = image.shape[:2]
    for item in standards:
        item["bbox_pct_xywh"] = xyxy_to_pct_xywh(item["bbox_xyxy"], width, height)
    return standards


def count_significant_standard_hits(query_box, standard_boxes, std_cover_threshold):
    hits = 0
    for standard in standard_boxes:
        std_area = box_area(standard["bbox_xyxy"])
        inter = intersection_area(query_box, standard["bbox_xyxy"])
        std_cover = inter / max(1.0, std_area)
        if std_cover >= std_cover_threshold:
            hits += 1
    return hits


def find_best_standard_match(query_box, standard_boxes):
    query_area = box_area(query_box)
    if query_area <= 0:
        return None

    best = None
    for standard in standard_boxes:
        std_box = standard["bbox_xyxy"]
        std_area = box_area(std_box)
        inter = intersection_area(query_box, std_box)
        if inter <= 0:
            continue
        query_cover = inter / max(1.0, query_area)
        std_cover = inter / max(1.0, std_area)
        iou = box_iou(query_box, std_box)
        score = 0.55 * query_cover + 0.30 * iou + 0.15 * std_cover
        candidate = {
            "tooth_id": standard["tooth_id"],
            "class_id": standard["class_id"],
            "standard_score": standard["score"],
            "bbox_xyxy": standard["bbox_xyxy"],
            "bbox_pct_xywh": standard["bbox_pct_xywh"],
            "match_score": score,
            "query_cover": query_cover,
            "std_cover": std_cover,
            "iou": iou,
        }
        if best is None or candidate["match_score"] > best["match_score"]:
            best = candidate
    return best


def ignore_reason(doctor_box, standard_boxes, args):
    x_pct, y_pct, w_pct, h_pct = doctor_box.bbox_pct_xywh
    if w_pct > args.max_box_width_percent:
        return "box_too_wide"
    if h_pct > args.max_box_height_percent:
        return "box_too_tall"
    hit_count = count_significant_standard_hits(
        doctor_box.bbox_xyxy,
        standard_boxes,
        args.multi_hit_std_cover_threshold,
    )
    if hit_count > 1:
        return "multi_tooth_box"
    return None


def build_source_record(doctor_box, source_type, match, region_xyxy=None):
    record = {
        "annotator": doctor_box.doctor_id,
        "source_file": doctor_box.source_file,
        "label_name": doctor_box.label_name,
        "label_id": doctor_box.label_id,
        "source_type": source_type,
        "original_bbox_xyxy": [round(float(v), 2) for v in doctor_box.bbox_xyxy],
        "original_bbox_pct_xywh": doctor_box.bbox_pct_xywh,
        "match_score": round(float(match["match_score"]), 6),
        "query_cover": round(float(match["query_cover"]), 6),
        "std_cover": round(float(match["std_cover"]), 6),
        "iou": round(float(match["iou"]), 6),
    }
    if region_xyxy is not None:
        record["match_region_xyxy"] = [round(float(v), 2) for v in region_xyxy]
    return record


def get_or_create_image_entry(image_entry, standard_match):
    tooth_id = standard_match["tooth_id"]
    if tooth_id not in image_entry:
        image_entry[tooth_id] = {
            "tooth_id": tooth_id,
            "class_id": standard_match["class_id"],
            "standard_box_xyxy": [round(float(v), 2) for v in standard_match["bbox_xyxy"]],
            "standard_box_pct_xywh": [round(float(v), 4) for v in standard_match["bbox_pct_xywh"]],
            "standard_box_score": round(float(standard_match["standard_score"]), 6),
            "labels": [],
            "annotators": [],
            "source_annotations": [],
        }
    return image_entry[tooth_id]


def merge_source(image_entry, standard_match, source_records):
    target = get_or_create_image_entry(image_entry, standard_match)
    for source in source_records:
        if source["label_name"] not in target["labels"]:
            target["labels"].append(source["label_name"])
        if source["annotator"] not in target["annotators"]:
            target["annotators"].append(source["annotator"])
        target["source_annotations"].append(source)


def normalize_one_image(file_name, doctor_boxes_by_image, standard_boxes, args):
    image_results = {}
    ignored = []
    doctor_ids = sorted(doctor_boxes_by_image.keys())
    doctor_a = doctor_ids[0] if doctor_ids else "doctor_1"
    doctor_b = doctor_ids[1] if len(doctor_ids) > 1 else "doctor_2"
    boxes_a = doctor_boxes_by_image.get(doctor_a, [])
    boxes_b = doctor_boxes_by_image.get(doctor_b, [])

    valid_a = []
    valid_b = []
    for doctor_box in boxes_a + boxes_b:
        reason = ignore_reason(doctor_box, standard_boxes, args)
        if reason is not None:
            ignored.append(
                {
                    "file_name": file_name,
                    "annotator": doctor_box.doctor_id,
                    "label_name": doctor_box.label_name,
                    "bbox_xyxy": doctor_box.bbox_xyxy,
                    "bbox_pct_xywh": doctor_box.bbox_pct_xywh,
                    "reason": reason,
                }
            )
            continue
        if doctor_box.doctor_id == doctor_a:
            valid_a.append(doctor_box)
        else:
            valid_b.append(doctor_box)

    pair_candidates = []
    for idx_a, box_a in enumerate(valid_a):
        for idx_b, box_b in enumerate(valid_b):
            inter_box = intersection_box(box_a.bbox_xyxy, box_b.bbox_xyxy)
            if inter_box is None:
                continue
            match = find_best_standard_match(inter_box, standard_boxes)
            if match is None:
                continue
            if match["query_cover"] < args.pair_min_query_cover or match["match_score"] < args.pair_min_match_score:
                continue
            pair_candidates.append(
                {
                    "idx_a": idx_a,
                    "idx_b": idx_b,
                    "box_a": box_a,
                    "box_b": box_b,
                    "inter_box": inter_box,
                    "match": match,
                }
            )

    pair_candidates.sort(
        key=lambda item: (float(item["match"]["match_score"]), box_area(item["inter_box"])),
        reverse=True,
    )

    consumed_a = set()
    consumed_b = set()
    for candidate in pair_candidates:
        if candidate["idx_a"] in consumed_a or candidate["idx_b"] in consumed_b:
            continue
        merge_source(
            image_results,
            candidate["match"],
            [
                build_source_record(candidate["box_a"], "pair_intersection", candidate["match"], candidate["inter_box"]),
                build_source_record(candidate["box_b"], "pair_intersection", candidate["match"], candidate["inter_box"]),
            ],
        )
        consumed_a.add(candidate["idx_a"])
        consumed_b.add(candidate["idx_b"])

    for idx, doctor_box in enumerate(valid_a):
        if idx in consumed_a:
            continue
        match = find_best_standard_match(doctor_box.bbox_xyxy, standard_boxes)
        if match is None:
            ignored.append(
                {
                    "file_name": file_name,
                    "annotator": doctor_box.doctor_id,
                    "label_name": doctor_box.label_name,
                    "bbox_xyxy": doctor_box.bbox_xyxy,
                    "bbox_pct_xywh": doctor_box.bbox_pct_xywh,
                    "reason": "no_standard_overlap",
                }
            )
            continue
        if match["query_cover"] < args.single_min_query_cover or match["match_score"] < args.single_min_match_score:
            ignored.append(
                {
                    "file_name": file_name,
                    "annotator": doctor_box.doctor_id,
                    "label_name": doctor_box.label_name,
                    "bbox_xyxy": doctor_box.bbox_xyxy,
                    "bbox_pct_xywh": doctor_box.bbox_pct_xywh,
                    "reason": "low_single_match_quality",
                    "best_tooth_id": match["tooth_id"],
                    "match_score": round(float(match["match_score"]), 6),
                    "query_cover": round(float(match["query_cover"]), 6),
                    "iou": round(float(match["iou"]), 6),
                }
            )
            continue
        merge_source(
            image_results,
            match,
            [build_source_record(doctor_box, "single_box", match)],
        )

    for idx, doctor_box in enumerate(valid_b):
        if idx in consumed_b:
            continue
        match = find_best_standard_match(doctor_box.bbox_xyxy, standard_boxes)
        if match is None:
            ignored.append(
                {
                    "file_name": file_name,
                    "annotator": doctor_box.doctor_id,
                    "label_name": doctor_box.label_name,
                    "bbox_xyxy": doctor_box.bbox_xyxy,
                    "bbox_pct_xywh": doctor_box.bbox_pct_xywh,
                    "reason": "no_standard_overlap",
                }
            )
            continue
        if match["query_cover"] < args.single_min_query_cover or match["match_score"] < args.single_min_match_score:
            ignored.append(
                {
                    "file_name": file_name,
                    "annotator": doctor_box.doctor_id,
                    "label_name": doctor_box.label_name,
                    "bbox_xyxy": doctor_box.bbox_xyxy,
                    "bbox_pct_xywh": doctor_box.bbox_pct_xywh,
                    "reason": "low_single_match_quality",
                    "best_tooth_id": match["tooth_id"],
                    "match_score": round(float(match["match_score"]), 6),
                    "query_cover": round(float(match["query_cover"]), 6),
                    "iou": round(float(match["iou"]), 6),
                }
            )
            continue
        merge_source(
            image_results,
            match,
            [build_source_record(doctor_box, "single_box", match)],
        )

    standardized = sorted(image_results.values(), key=lambda item: int(item["tooth_id"]))
    for entry in standardized:
        entry["labels"].sort()
        entry["annotators"].sort()

    return standardized, ignored, {
        "raw_doctor_boxes": len(boxes_a) + len(boxes_b),
        "valid_doctor_boxes": len(valid_a) + len(valid_b),
        "paired_matches": len(consumed_a) + len(consumed_b),
        "standardized_teeth": len(standardized),
        "ignored_boxes": len(ignored),
    }


def main():
    args = get_parser().parse_args()
    if str(args.device).lower() == "cuda":
        # On this WSL setup, warming up CUDA before the rest of the pipeline
        # avoids a bad first-time CUDA initialization state later in model load.
        _ = torch.tensor([0.0], device="cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doctor_map = read_annotation_files(args.annotation_dir)
    doctor_boxes_by_image = build_doctor_boxes(doctor_map)

    cfg = build_cfg(args)
    model, _ = load_model(cfg)

    image_names = sorted(doctor_boxes_by_image.keys())
    normalized_images = []
    ignored_annotations = []
    per_image_summary = []
    global_counter = Counter()

    for image_name in image_names:
        image_path = Path(args.image_dir) / image_name
        standard_boxes = predict_standard_boxes(args, model, cfg, image_path)
        if not standard_boxes:
            ignored_annotations.append(
                {
                    "file_name": image_name,
                    "annotator": None,
                    "label_name": None,
                    "bbox_xyxy": None,
                    "bbox_pct_xywh": None,
                    "reason": "no_standard_boxes_predicted",
                }
            )
            per_image_summary.append(
                {
                    "file_name": image_name,
                    "standard_boxes": 0,
                    "raw_doctor_boxes": sum(len(v) for v in doctor_boxes_by_image[image_name].values()),
                    "valid_doctor_boxes": 0,
                    "paired_matches": 0,
                    "standardized_teeth": 0,
                    "ignored_boxes": 1,
                }
            )
            global_counter["images_without_standard_boxes"] += 1
            continue

        standardized, ignored, summary = normalize_one_image(
            image_name,
            doctor_boxes_by_image[image_name],
            standard_boxes,
            args,
        )
        normalized_images.append(
            {
                "file_name": image_name,
                "standard_box_count": len(standard_boxes),
                "standardized_annotations": standardized,
            }
        )
        ignored_annotations.extend(ignored)
        per_image_summary.append(
            {
                "file_name": image_name,
                "standard_boxes": len(standard_boxes),
                **summary,
            }
        )
        global_counter["images"] += 1
        global_counter["standard_boxes"] += len(standard_boxes)
        global_counter["raw_doctor_boxes"] += summary["raw_doctor_boxes"]
        global_counter["valid_doctor_boxes"] += summary["valid_doctor_boxes"]
        global_counter["standardized_teeth"] += summary["standardized_teeth"]
        global_counter["ignored_boxes"] += summary["ignored_boxes"]

    normalized_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "annotation_dir": str(Path(args.annotation_dir).resolve()),
            "image_dir": str(Path(args.image_dir).resolve()),
            "doctor_files": {
                doctor_id: payload["source_file"] for doctor_id, payload in doctor_map.items()
            },
            "pipeline_parameters": {
                "score_threshold": args.score_threshold,
                "nms_threshold": args.nms_threshold,
                "max_detections": args.max_detections,
                "duplicate_iou_threshold": args.duplicate_iou_threshold,
                "duplicate_center_threshold": args.duplicate_center_threshold,
                "duplicate_containment_threshold": args.duplicate_containment_threshold,
                "duplicate_very_high_iou_threshold": args.duplicate_very_high_iou_threshold,
                "single_min_query_cover": args.single_min_query_cover,
                "single_min_match_score": args.single_min_match_score,
                "pair_min_query_cover": args.pair_min_query_cover,
                "pair_min_match_score": args.pair_min_match_score,
                "max_box_width_percent": args.max_box_width_percent,
                "max_box_height_percent": args.max_box_height_percent,
                "multi_hit_std_cover_threshold": args.multi_hit_std_cover_threshold,
            },
        },
        "images": normalized_images,
    }

    summary_payload = {
        "global": dict(global_counter),
        "per_image": per_image_summary,
        "ignored_reason_counts": dict(Counter(item["reason"] for item in ignored_annotations)),
    }

    with open(output_dir / "normalized_annotations.json", "w", encoding="utf-8") as f:
        json.dump(normalized_payload, f, ensure_ascii=False, indent=2)
    with open(output_dir / "ignored_annotations.json", "w", encoding="utf-8") as f:
        json.dump(ignored_annotations, f, ensure_ascii=False, indent=2)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved normalized annotations to {output_dir / 'normalized_annotations.json'}")
    print(f"Saved ignored annotations to {output_dir / 'ignored_annotations.json'}")
    print(f"Saved summary to {output_dir / 'summary.json'}")
    print(json.dumps(summary_payload["global"], ensure_ascii=False, indent=2))
    print(json.dumps(summary_payload["ignored_reason_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
