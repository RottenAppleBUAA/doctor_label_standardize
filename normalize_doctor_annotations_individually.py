import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from normalize_doctor_annotations import (
    build_cfg,
    build_doctor_boxes,
    build_source_record,
    find_best_standard_match,
    ignore_reason,
    load_model,
    merge_source,
    predict_standard_boxes,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = Path("configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
DEFAULT_WEIGHTS = Path("pretrained_models/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Standardize each doctor annotation json independently with model-predicted standard boxes."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory for this run. Relative paths are resolved from the project directory.",
    )
    parser.add_argument(
        "--annotation-dir",
        default=None,
        help="Directory containing doctor annotation json files. Defaults to <data-root>/annotations.",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing source images referenced by the annotations.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <data-root>/standardized_annotations_per_doctor.",
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help="Config file path. Relative paths are resolved from the project directory.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to model weights. Relative paths are resolved from the project directory.",
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


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "doctor"


def resolve_from_project(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_annotation_files(annotation_dir: Path) -> list[Path]:
    paths = sorted(path for path in annotation_dir.glob("*.json") if path.is_file())
    if not paths:
        raise ValueError(f"No annotation json files found in {annotation_dir}")
    return paths


def doctor_map_from_path(path: Path) -> tuple[str, dict]:
    annotator_id = slugify(path.stem)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    doctor_map = {
        annotator_id: {
            "source_file": path.name,
            "files": data.get("files", []),
        }
    }
    return annotator_id, doctor_map


def normalize_one_image_for_single_doctor(file_name, doctor_boxes, standard_boxes, args):
    image_results = {}
    ignored = []
    valid_boxes = []

    for doctor_box in doctor_boxes:
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
        valid_boxes.append(doctor_box)

    for doctor_box in valid_boxes:
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
        "raw_doctor_boxes": len(doctor_boxes),
        "valid_doctor_boxes": len(valid_boxes),
        "standardized_teeth": len(standardized),
        "ignored_boxes": len(ignored),
    }


def build_pipeline_parameters(args):
    return {
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "max_detections": args.max_detections,
        "duplicate_iou_threshold": args.duplicate_iou_threshold,
        "duplicate_center_threshold": args.duplicate_center_threshold,
        "duplicate_containment_threshold": args.duplicate_containment_threshold,
        "duplicate_very_high_iou_threshold": args.duplicate_very_high_iou_threshold,
        "single_min_query_cover": args.single_min_query_cover,
        "single_min_match_score": args.single_min_match_score,
        "max_box_width_percent": args.max_box_width_percent,
        "max_box_height_percent": args.max_box_height_percent,
        "multi_hit_std_cover_threshold": args.multi_hit_std_cover_threshold,
    }


def standard_boxes_for_image(image_name, args, model, cfg, image_dir: Path, cache: dict):
    if image_name in cache:
        return cache[image_name]

    image_path = image_dir / image_name
    if not image_path.exists():
        cache[image_name] = {
            "status": "missing_image",
            "image_path": str(image_path),
            "standard_boxes": None,
        }
        return cache[image_name]

    standard_boxes = predict_standard_boxes(args, model, cfg, image_path)
    if not standard_boxes:
        cache[image_name] = {
            "status": "no_standard_boxes_predicted",
            "image_path": str(image_path),
            "standard_boxes": [],
        }
        return cache[image_name]

    cache[image_name] = {
        "status": "ok",
        "image_path": str(image_path),
        "standard_boxes": standard_boxes,
    }
    return cache[image_name]


def unique_output_dir(output_root: Path, base_name: str, used_names: set[str]) -> Path:
    candidate = slugify(base_name)
    if candidate not in used_names:
        used_names.add(candidate)
        return output_root / candidate

    index = 2
    while f"{candidate}_{index}" in used_names:
        index += 1
    final_name = f"{candidate}_{index}"
    used_names.add(final_name)
    return output_root / final_name


def process_annotation_file(path: Path, args, model, cfg, image_cache: dict, output_dir: Path):
    annotator_id, doctor_map = doctor_map_from_path(path)
    doctor_boxes_by_image = build_doctor_boxes(doctor_map)

    normalized_images = []
    ignored_annotations = []
    per_image_summary = []
    global_counter = Counter()

    for image_name in sorted(doctor_boxes_by_image.keys()):
        cache_entry = standard_boxes_for_image(image_name, args, model, cfg, Path(args.image_dir), image_cache)
        doctor_boxes = doctor_boxes_by_image[image_name].get(annotator_id, [])

        if cache_entry["status"] == "missing_image":
            ignored_annotations.append(
                {
                    "file_name": image_name,
                    "annotator": annotator_id,
                    "label_name": None,
                    "bbox_xyxy": None,
                    "bbox_pct_xywh": None,
                    "reason": "image_not_found",
                    "image_path": cache_entry["image_path"],
                }
            )
            per_image_summary.append(
                {
                    "file_name": image_name,
                    "standard_boxes": 0,
                    "raw_doctor_boxes": len(doctor_boxes),
                    "valid_doctor_boxes": 0,
                    "standardized_teeth": 0,
                    "ignored_boxes": 1,
                }
            )
            global_counter["images_with_missing_source_image"] += 1
            continue

        if cache_entry["status"] == "no_standard_boxes_predicted":
            ignored_annotations.append(
                {
                    "file_name": image_name,
                    "annotator": annotator_id,
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
                    "raw_doctor_boxes": len(doctor_boxes),
                    "valid_doctor_boxes": 0,
                    "standardized_teeth": 0,
                    "ignored_boxes": 1,
                }
            )
            global_counter["images_without_standard_boxes"] += 1
            continue

        standardized, ignored, summary = normalize_one_image_for_single_doctor(
            image_name,
            doctor_boxes,
            cache_entry["standard_boxes"],
            args,
        )
        normalized_images.append(
            {
                "file_name": image_name,
                "standard_box_count": len(cache_entry["standard_boxes"]),
                "standardized_annotations": standardized,
            }
        )
        ignored_annotations.extend(ignored)
        per_image_summary.append(
            {
                "file_name": image_name,
                "standard_boxes": len(cache_entry["standard_boxes"]),
                **summary,
            }
        )
        global_counter["images"] += 1
        global_counter["standard_boxes"] += len(cache_entry["standard_boxes"])
        global_counter["raw_doctor_boxes"] += summary["raw_doctor_boxes"]
        global_counter["valid_doctor_boxes"] += summary["valid_doctor_boxes"]
        global_counter["standardized_teeth"] += summary["standardized_teeth"]
        global_counter["ignored_boxes"] += summary["ignored_boxes"]

    normalized_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "annotation_file": str(path.resolve()),
            "image_dir": str(Path(args.image_dir).resolve()),
            "annotator_id": annotator_id,
            "pipeline_parameters": build_pipeline_parameters(args),
        },
        "images": normalized_images,
    }

    summary_payload = {
        "global": dict(global_counter),
        "per_image": per_image_summary,
        "ignored_reason_counts": dict(Counter(item["reason"] for item in ignored_annotations)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "normalized_annotations.json").open("w", encoding="utf-8") as fh:
        json.dump(normalized_payload, fh, ensure_ascii=False, indent=2)
    with (output_dir / "ignored_annotations.json").open("w", encoding="utf-8") as fh:
        json.dump(ignored_annotations, fh, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    return {
        "annotation_file": str(path.resolve()),
        "annotator_id": annotator_id,
        "output_dir": str(output_dir.resolve()),
        "global_summary": summary_payload["global"],
        "ignored_reason_counts": summary_payload["ignored_reason_counts"],
    }


def main():
    args = get_parser().parse_args()
    if str(args.device).lower() == "cuda":
        _ = torch.tensor([0.0], device="cuda")

    args.config_file = str(resolve_from_project(args.config_file))
    args.weights = str(resolve_from_project(args.weights))

    data_root = resolve_from_project(args.data_root)
    annotation_dir = resolve_from_project(args.annotation_dir) if args.annotation_dir else (data_root / "annotations")
    image_dir = resolve_from_project(args.image_dir)
    output_root = resolve_from_project(args.output_dir) if args.output_dir else (data_root / "standardized_annotations_per_doctor")

    args.image_dir = str(image_dir)

    annotation_paths = load_annotation_files(annotation_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    model, _ = load_model(cfg)

    image_cache = {}
    used_names = set()
    manifest_entries = []
    for path in annotation_paths:
        doctor_output_dir = unique_output_dir(output_root, path.stem, used_names)
        result = process_annotation_file(path, args, model, cfg, image_cache, doctor_output_dir)
        manifest_entries.append(result)
        print(f"Saved standardized outputs for {path.name} -> {doctor_output_dir}")

    manifest = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "annotation_dir": str(annotation_dir),
            "image_dir": str(image_dir),
            "output_dir": str(output_root),
            "annotation_file_count": len(annotation_paths),
            "pipeline_parameters": build_pipeline_parameters(args),
        },
        "annotation_outputs": manifest_entries,
    }
    with (output_root / "batch_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print()
    print("Per-doctor standardization complete.")
    print(f"Annotation files processed: {len(annotation_paths)}")
    print(f"Batch summary: {output_root / 'batch_summary.json'}")
    print(f"Per-doctor outputs: {output_root}")


if __name__ == "__main__":
    main()
