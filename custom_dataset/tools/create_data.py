# projects/tools/create_data.py
import argparse
import data_converter.custom_converter as custom_converter
from data_converter.create_gt_database import create_groundtruth_database


def custom_data_prep(root_path, info_prefix, dataset_name, out_dir):
    custom_converter.create_custom_infos(root_path, info_prefix)
    create_groundtruth_database(dataset_name,
                                root_path,
                                info_prefix,
                                f"{out_dir}/{info_prefix}_infos_train.pkl")


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="MyCustomDataset", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="/",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="/",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="custom")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "custom":

        custom_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="MyCustomDataset",
            out_dir=args.out_dir
        )
# python custom_dataset/tools/create_data.py custom --root-path data/20240617-720 --out-dir data/20240617-720 --extra-tag custom 
        