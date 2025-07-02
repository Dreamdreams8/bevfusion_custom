#!/bin/bash

# 设置工作目录
WORK_DIR="/data/why/bevfusion"
cd $WORK_DIR

# 配置参数
CONFIG="configs/pointpillars_lidar_only_custom.py"
CHECKPOINT="work_dirs/pointpillars_lidar_only_custom/latest.pth"
INPUT_DIR="data/custom/lidar"
OUTPUT_DIR="data/enhanced_inference_results"
BBOX_SCORE=0.3

# 运行增强版批量推理脚本
python custom_dataset/tools/enhanced_batch_inference.py \
    $CONFIG \
    --checkpoint $CHECKPOINT \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --bbox-score $BBOX_SCORE \
    --summary

# 检查结果
echo "\n检查生成的文件数量:"
echo "TXT文件数量: $(ls $OUTPUT_DIR/txt | wc -l)"
echo "PNG文件数量: $(ls $OUTPUT_DIR/viz | wc -l)"

echo "\n查看前3个TXT文件内容:"
ls $OUTPUT_DIR/txt | head -n 3 | xargs -I{} sh -c "echo '\n文件: {}'; cat $OUTPUT_DIR/txt/{}"

echo "\n查看摘要文件:"
cat $OUTPUT_DIR/inference_summary.txt

python custom_dataset/tools/demo_result_show.py custom_dataset/configs/pointpillars_lidar_only_custom.py \
    --checkpoint output/lidar_result/epoch_274.pth \
    --input-dir data/test/points \
    --output-dir data/enhanced_inference_results \
    --bbox-score 0.5 \
    --recursive \
    --summary
