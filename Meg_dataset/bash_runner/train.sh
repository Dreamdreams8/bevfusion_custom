DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='projects/Meg_dataset/tools/train.py'
CONFIG_FILE='projects/Meg_dataset/configs/bevfusion_c_l_meg.py'
WORK_DIR="runs/${DATE}/"

torchpack dist-run -np 1 python ${TRAIN_PY} ${CONFIG_FILE} --run-dir ${WORK_DIR}
# torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TRAIN_PY} ${CONFIG_FILE}