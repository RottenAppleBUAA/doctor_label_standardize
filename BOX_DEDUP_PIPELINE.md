# SemiT-SAM Box Dedup Pipeline

## Purpose

This document describes the current local inference pipeline used for tooth instance segmentation and box deduplication in `demo_infer.py`.

The practical goal is:

- keep recall high
- avoid aggressive pre-filtering
- remove only boxes that are likely duplicate detections of the same tooth
- output box-focused visualization for manual review

## Pipeline Overview

The script first runs the SemiT-SAM instance segmentation model, then performs lightweight box-level post-processing.

Current flow:

1. read image
2. run model inference and get `instances`
3. optionally apply score filtering, NMS, and max-detection truncation
4. run duplicate-box suppression
5. draw boxes
6. save `predictions.json` and optional side-by-side comparison images

Important:

- this pipeline does not use standalone external SAM prompting after detection
- the segmentation result comes from the model's own instance output
- current visualization defaults to box-only output, even though the model can also output masks

## Default Rule

The current default parameters are tuned for "prefer recall, only remove obvious duplicates":

```text
score-threshold = 0.0
nms-threshold = -1
max-detections = -1
duplicate-iou-threshold = 0.70
duplicate-center-threshold = 0.18
duplicate-containment-threshold = 0.88
duplicate-very-high-iou-threshold = 0.88
```

Meaning:

- no score pre-filter by default
- no standard NMS by default
- no top-k truncation by default
- duplicate suppression is the main filtering step

## Duplicate Suppression Logic

Boxes are sorted by:

1. score descending
2. area descending

So the current rule prefers keeping the higher-score box.

For each candidate box, compare it with already kept boxes. The candidate is removed if any one of the following conditions is met:

1. `IoU >= duplicate_iou_threshold` and normalized center distance `<= duplicate_center_threshold`
2. containment ratio `>= duplicate_containment_threshold`
3. `IoU >= duplicate_very_high_iou_threshold`

Where:

- `IoU` is standard intersection over union
- normalized center distance is the center-point distance divided by the smaller box diagonal
- containment ratio is:

```text
intersection_area / min(area_a, area_b)
```

The containment rule is intentionally softer than strict full inclusion. A small box can still be treated as duplicate even if a small part extends outside the larger box.

## Why This Configuration

This configuration was chosen after multiple rounds of visual comparison on local panoramic tooth images.

Observed tradeoff:

- standard NMS or score filtering reduced duplicates, but also increased missed detections
- class-agnostic NMS was too aggressive for dense neighboring teeth
- duplicate-only suppression preserved recall better
- a softened containment rule helped remove small boxes that were almost covered by a larger box

So the current version focuses on:

- keeping difficult teeth
- only deleting boxes that are very likely repeated detections of the same instance

## Command Example

Example command using the current defaults:

```bash
conda run -n gd_env python demo_infer.py \
  --weights pretrained_models/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth \
  --input-dir "/home/tianruiliu/codespace/data_process/586份数据20260116/586张原图" \
  --output-dir demo_output_default_dedup \
  --limit 3 \
  --device cpu \
  --num-classes 32 \
  --save-comparison
```

Because the defaults are already fixed in the script, the duplicate-suppression parameters do not need to be passed explicitly unless further tuning is required.

## Output Structure

Normal output:

- rendered images in `output-dir`
- `predictions.json`

If `--save-comparison` is enabled, the script also writes:

- `raw_boxes/`: raw boxes before duplicate suppression
- `filtered_boxes/`: boxes after duplicate suppression
- `comparison/`: side-by-side comparison image

## Tuning Guidance

If duplicate boxes are still too many:

- reduce `duplicate-iou-threshold`
- reduce `duplicate-center-threshold`
- reduce `duplicate-containment-threshold`
- reduce `duplicate-very-high-iou-threshold`

If missed detections start to appear:

- increase `duplicate-iou-threshold`
- increase `duplicate-center-threshold`
- increase `duplicate-containment-threshold`
- increase `duplicate-very-high-iou-threshold`

Recommended adjustment strategy:

1. change only one parameter at a time
2. inspect `comparison/` on the same image set
3. prioritize recall first, then clean up obvious duplicates

## Relevant Files

- `demo_infer.py`: inference entry and duplicate-suppression implementation
- `predictions.json`: structured output of kept boxes
- `comparison/`: easiest place to inspect raw vs filtered results
