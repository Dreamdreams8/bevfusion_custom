import argparse
import data_convert.meg_converter as meg_converter
from data_convert.create_gt_database import create_groundtruth_database


def meg_data_prep(root_path, info_prefix, dataset_name, out_dir):
    meg_converter.create_meg_infos(root_path, info_prefix)
    create_groundtruth_database(dataset_name,
                                root_path,
                                info_prefix,
                                f"{out_dir}/{info_prefix}_infos_train.pkl")


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="MegDataset", help="name of the dataset")
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
parser.add_argument("--extra-tag", type=str, default="meg")


args = parser.parse_args()

if __name__ == "__main__":

    if args.dataset == "meg":

        meg_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_name="MegDataset",
            out_dir=args.out_dir
        )