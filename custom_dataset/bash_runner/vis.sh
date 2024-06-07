VIS_PY='custom_dataset/tools/visualize.py'
CONFIG_FILE='custom_dataset/configs/bevfusion_l_pointpillars_meg.py'
CHECK_POINT='pretrained/2023-06-13_08-46-56/epoch_24.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}
# python ${DEBUG_PY} ${VIS_PY} ${CONFIG_FILE} --checkpoint ${CHECK_POINT}