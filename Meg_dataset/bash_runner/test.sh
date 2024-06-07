TEST_PY='projects/Meg_dataset/tools/test.py'
CONFIG_FILE='projects/Meg_dataset/configs/bevfusion_c_l_meg.py'
PTH='pretrained/2023-06-13_08-46-56/epoch_24.pth'
EVAL='map'

# torchpack dist-run -np 1 python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}
torchpack dist-run -np 1 python -m debugpy --listen 8531 --wait-for-client ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval ${EVAL}