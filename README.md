# Doctor Label Standardize

这个仓库当前主要用于医生标注标准化。

核心入口是：

- [`run_per_doctor_annotation_standardization_pipeline.sh`](./run_per_doctor_annotation_standardization_pipeline.sh)
- [`normalize_doctor_annotations_individually.py`](./normalize_doctor_annotations_individually.py)

这条 pipeline 的目标是：

- 读取 `annotation-dir` 下所有医生标注 JSON
- 对每个 JSON 单独标准化
- 将医生标注框匹配到模型预测的标准牙位框
- 把输出结果写到当前数据目录下面

它不要求标注 JSON 数量必须是 2 个。有多少个 JSON，就分别处理多少个。

## 安装

如果要按当前仓库默认配置创建 `gd_env` 环境，可以直接运行：

```bash
cd /home/tianruiliu/codespace/data_process/SemiT-SAM
bash install_gd_env.sh
```

安装脚本会处理：

- `Python 3.10`
- `PyTorch 1.13.1`
- `torchvision 0.14.1`
- `pytorch-cuda 11.7`
- 本地 `detectron2-main`
- `mask2former` 自定义算子编译

## 路径规则

这版标准化流程不依赖固定绝对路径。

你可以传绝对路径，也可以传相对路径。相对路径统一按项目根目录解析，也就是当前 README 所在目录。

涉及这个规则的参数包括：

- `--data-root`
- `--annotation-dir`
- `--image-dir`
- `--output-dir`
- `--config-file`
- `--weights`

## 快速开始

假设你的数据结构如下：

```text
some_case/
  annotations/
    doctor_a.json
    doctor_b.json
    doctor_c.json
  images/
    001.png
    002.png
```

在项目根目录运行：

```bash
cd /home/tianruiliu/codespace/data_process/SemiT-SAM

./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --image-dir ../some_case/images
```

如果不传 `--annotation-dir`，默认读取：

```text
<data-root>/annotations
```

如果不传 `--output-dir`，默认输出到：

```text
<data-root>/standardized_annotations_per_doctor
```

## 常用用法

指定标注目录：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --annotation-dir ../some_case/my_annotations \
  --image-dir ../some_case/images
```

指定输出目录：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --annotation-dir ../some_case/annotations \
  --image-dir ../some_case/images \
  --output-dir ../some_case/my_standardized_output
```

使用 CUDA：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --image-dir ../some_case/images \
  --device cuda
```

切换 conda 环境：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --image-dir ../some_case/images \
  --conda-env my_env
```

调整标准化阈值：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh \
  --data-root ../some_case \
  --image-dir ../some_case/images \
  --single-min-query-cover 0.30 \
  --single-min-match-score 0.40 \
  --max-box-width-percent 20 \
  --max-box-height-percent 26
```

## 直接运行 Python

如果你不想通过 shell pipeline，也可以直接运行：

```bash
conda run -n gd_env python normalize_doctor_annotations_individually.py \
  --data-root ../some_case \
  --annotation-dir ../some_case/annotations \
  --image-dir ../some_case/images
```

## 输出结构

默认输出目录：

```text
<data-root>/standardized_annotations_per_doctor/
```

输出示例：

```text
standardized_annotations_per_doctor/
  doctor_a/
    normalized_annotations.json
    ignored_annotations.json
    summary.json
  doctor_b/
    normalized_annotations.json
    ignored_annotations.json
    summary.json
  doctor_c/
    normalized_annotations.json
    ignored_annotations.json
    summary.json
  batch_summary.json
```

其中：

- `normalized_annotations.json`：当前医生 JSON 的标准化结果
- `ignored_annotations.json`：未进入最终结果的框及原因
- `summary.json`：当前医生 JSON 的统计汇总
- `batch_summary.json`：这次批处理的总清单

## 运行前提

运行前需要保证：

- `conda` 可用
- 环境里已安装 `torch`、`opencv-python` 等依赖
- 配置文件存在：`configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml`
- 权重文件存在：`pretrained_models/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth`
- JSON 中的 `fileName` 能在 `image-dir` 下找到对应原图

## 常见问题

`No annotation json files found ...`

- `annotation-dir` 下没有任何 `.json` 文件

`Unable to read image ...`

- 原图缺失，或者 JSON 里的 `fileName` 和实际文件名不一致

`image_not_found`

- 某张图在 `image-dir` 下找不到，对应记录会写入 `ignored_annotations.json`

`no_standard_boxes_predicted`

- 模型在某张图上没有预测出标准牙位框，对应记录会写入 `ignored_annotations.json`

CUDA 初始化失败：

- 使用 `--device cuda` 时，脚本会先做一次 CUDA 预热
- 如果仍失败，优先检查环境中的 `torch` 与 CUDA 是否匹配

## 帮助命令

查看 shell 脚本帮助：

```bash
./run_per_doctor_annotation_standardization_pipeline.sh --help
```

查看 Python 脚本帮助：

```bash
conda run -n gd_env python normalize_doctor_annotations_individually.py --help
```

## 相关文件

- [`使用说明.md`](./使用说明.md)：更完整的中文使用说明
- [`run_per_doctor_annotation_standardization_pipeline.sh`](./run_per_doctor_annotation_standardization_pipeline.sh)：标准化 pipeline 入口
- [`normalize_doctor_annotations_individually.py`](./normalize_doctor_annotations_individually.py)：逐医生单独标准化实现
- [`normalize_doctor_annotations.py`](./normalize_doctor_annotations.py)：旧的双医生标准化逻辑
- [`BOX_DEDUP_PIPELINE_ZH.md`](./BOX_DEDUP_PIPELINE_ZH.md)：标准框去重规则说明
