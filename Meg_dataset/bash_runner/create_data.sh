ROOT_PATH_PROJ='/home/bevfusion/'
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/meg_data'
echo ${ROOT_PATH_DATASET}
python Meg_dataset/tools/create_data.py meg --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag meg