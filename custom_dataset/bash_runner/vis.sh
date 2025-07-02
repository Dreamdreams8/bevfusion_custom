VIS_PY='custom_dataset/tools/visualize.py'
CONFIG_FILE='custom_dataset/configs/bevfusion_l_pointpillars_custom.py'
CHECK_POINT='output/lidar_result/latest.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'


CUDA_VISIBLE_DEVICES=0  python  ${VIS_PY} ${CONFIG_FILE}  --model pred --checkpoint ${CHECK_POINT} --out-dir result/visualize3  --bbox-score 0.05
# python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
python custom_dataset/tools/visualize.py 'custom_dataset/configs/bevfusion_c_l_custom.py' --checkpoint 'output/lidar_result/latest.pth'
python custom_dataset/tools/visualize.py 'custom_dataset/configs/pointpillars_lidar_only_custom.py' --checkpoint 'output/lidar_result/epoch_48.pth'
