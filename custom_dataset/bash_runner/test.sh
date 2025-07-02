TEST_PY='custom_dataset/tools/test.py'
CONFIG_FILE='custom_dataset/configs/bevfusion_l_pointpillars_custom.py'
# PTH='output/lidar_result/epoch_24.pth'
PTH='output/lidar_result/latest.pth'
EVAL='map'

CUDA_VISIBLE_DEVICES=0 python ${TEST_PY} ${CONFIG_FILE}  ${PTH} --eval bbox
# torchpack dist-run -np 1 python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
torchpack dist-run -np 1 python 'custom_dataset/tools/test.py' 'custom_dataset/configs/pointpillars_lidar_only_custom.py' 'output/lidar_result/epoch_24.pth' --eval bbox
