ROOT_PATH_PROJ='/home/bevfusion/'
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/custom_data'
echo ${ROOT_PATH_DATASET}
python custom_dataset/tools/create_data.py custom --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag custom