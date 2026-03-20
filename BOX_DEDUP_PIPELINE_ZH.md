# SemiT-SAM 框去重推理流程说明

## 目的

这份文档说明当前本地使用的牙齿实例分割与框去重推理流程，入口脚本是 `demo_infer.py`。

这条流程的目标是：

- 优先保证召回率
- 避免过强的预筛选
- 只删除很可能属于同一颗牙的重复框
- 以框为主做可视化，便于人工检查

## 整体流程

脚本会先运行 SemiT-SAM 模型得到实例分割结果，再做轻量的框级后处理。

当前执行顺序如下：

1. 读取图像
2. 运行模型推理，得到 `instances`
3. 可选地执行分数过滤、NMS、最大检测数截断
4. 执行重复框去重
5. 绘制框
6. 保存 `predictions.json`，以及可选的左右对比图

需要注意：

- 这条流程不是“外部 SAM 先分割、再单独做检测提示”的独立两阶段流程
- 分割结果来自模型自身的实例输出
- 当前默认可视化是只画框，不画 mask

## 当前默认规则

当前默认参数是按“召回优先，只去掉明显重复框”这一目标调出来的：

```text
score-threshold = 0.0
nms-threshold = -1
max-detections = -1
duplicate-iou-threshold = 0.70
duplicate-center-threshold = 0.18
duplicate-containment-threshold = 0.88
duplicate-very-high-iou-threshold = 0.88
```

这些参数的含义是：

- 默认不做分数预筛选
- 默认不做标准 NMS
- 默认不做 top-k 截断
- 主要依赖“重复框去重”这一步来清理同实例重复框

## 重复框去重逻辑

框会先按以下顺序排序：

1. 分数从高到低
2. 面积从大到小

因此当前规则是优先保留高分框。

对每个候选框，都会与已经保留下来的框逐一比较。只要满足下面任意一个条件，就会把当前候选框判定为重复框并删除：

1. `IoU >= duplicate_iou_threshold`，并且中心点归一化距离 `<= duplicate_center_threshold`
2. 包含率 `>= duplicate_containment_threshold`
3. `IoU >= duplicate_very_high_iou_threshold`

其中：

- `IoU` 是标准交并比
- 中心点归一化距离是两个框中心点距离，再除以较小框的对角线长度
- 包含率定义为：

```text
intersection_area / min(area_a, area_b)
```

这里的“包含”不是严格要求小框完全落在大框内部，而是允许小框有一小部分露在大框外面。这样可以把“基本被大框包住，但边缘稍微露出”的小重复框也筛掉。

## 为什么采用这套参数

这套参数是在本地牙片样本上多轮可视化对比后确定的。

实际观察到的现象是：

- 标准 NMS 或提高分数阈值，虽然能让图更干净，但更容易漏检
- 类无关 NMS 对牙齿这种密集相邻目标通常过强
- 只做重复框去重，能更好保住召回
- 适度放宽“包含率”判定后，可以额外去掉一些几乎被大框包住的小框

所以当前版本的重点不是“最大程度清理所有重叠”，而是：

- 尽量保住难检出的牙齿
- 只删除非常像同一实例重复检测出来的框

## 命令示例

下面是使用当前默认规则的示例命令：

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

因为默认值已经固化到脚本里，通常不需要再额外传重复框相关参数，除非后续还要继续微调。

## 输出目录结构

普通输出包括：

- `output-dir/` 下的可视化结果图
- `predictions.json`

如果启用了 `--save-comparison`，还会额外生成：

- `raw_boxes/`：原始框，可用于查看去重前结果
- `filtered_boxes/`：去重后的框
- `comparison/`：左右拼接对比图，左边是原始框，右边是去重后框

## 调参建议

如果重复框还是偏多，可以尝试：

- 调低 `duplicate-iou-threshold`
- 调低 `duplicate-center-threshold`
- 调低 `duplicate-containment-threshold`
- 调低 `duplicate-very-high-iou-threshold`

如果开始出现漏检，可以尝试：

- 调高 `duplicate-iou-threshold`
- 调高 `duplicate-center-threshold`
- 调高 `duplicate-containment-threshold`
- 调高 `duplicate-very-high-iou-threshold`

建议的调参方式是：

1. 每次只改一个参数
2. 始终在同一批图上查看 `comparison/` 对比结果
3. 先保证召回，再逐步清理明显重复框

## 相关文件

- `demo_infer.py`：推理入口和重复框去重实现
- `predictions.json`：最终保留框的结构化输出
- `comparison/`：最适合人工核对原始框与去重后效果的目录
