#!/bin/bash

# 设置工作目录为项目根目录
cd /data/why/bevfusion

# 配置文件路径
CONFIG_FILE="custom_dataset/configs/pointpillars_lidar_only_custom.py"

# 检查点文件路径（根据实际情况修改）
CHECKPOINT="output/lidar_result/epoch_9.pth"

# 推理结果保存目录
SAVE_DIR="data/inference_results"

# 创建保存目录（如果不存在）
mkdir -p "$SAVE_DIR"

# 运行推理并保存结果
python custom_dataset/tools/save_inference_results.py $CONFIG_FILE \
    --checkpoint $CHECKPOINT \
    --bbox-score 0.3 \
    --save-dir $SAVE_DIR \
    --out-dir "viz/inference"

# 检查结果
echo "推理结果已保存到 $SAVE_DIR"
echo "文件数量:"
ls -l $SAVE_DIR | wc -l

# 显示前几个文件的内容
echo "前3个文件的内容示例:"
ls $SAVE_DIR | head -n 3 | xargs -I{} sh -c "echo '文件: {}'; head -n 3 $SAVE_DIR/{}"

# 使用预训练模型进行推理并保存结果
python custom_dataset/tools/save_inference_results.py custom_dataset/configs/pointpillars_lidar_only_custom.py --checkpoint output/lidar_result/epoch_9.pth --bbox-score 0.3 --save-dir data/inference_results
