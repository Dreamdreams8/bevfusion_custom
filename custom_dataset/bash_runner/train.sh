DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='custom_dataset/tools/train.py'
# CONFIG_FILE='custom_dataset/configs/bevfusion_c_l_custom.py'
CONFIG_FILE='custom_dataset/configs/bevfusion_l_pointpillars_custom.py'
# WORK_DIR="runs/${DATE}/"
WORK_DIR='output/lidar_result/'
CUDA_VISIBLE_DEVICES=0 python ${TRAIN_PY}  ${CONFIG_FILE}  --run-dir  ${WORK_DIR}     
# torchpack dist-run -np 1 python ${TRAIN_PY} ${CONFIG_FILE} --run-dir ${WORK_DIR}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TRAIN_PY} ${CONFIG_FILE}
# torchpack dist-run -np 1 python 'custom_dataset/tools/train.py' 'custom_dataset/configs/bevfusion_l_pointpillars_custom.py' --run-dir  'output/lidar_result/'
torchpack dist-run -np 1 python 'custom_dataset/tools/train.py' 'custom_dataset/configs/bevfusion_c_l_custom.py' --run-dir  'output/lidar_result/'
torchpack dist-run -np 4 python 'custom_dataset/tools/train.py' 'custom_dataset/configs/pointpillars_lidar_only_custom.py' --run-dir  'output/lidar_result/'
