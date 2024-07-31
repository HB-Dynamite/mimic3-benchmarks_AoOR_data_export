# %%
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models import common_utils
from mimic3models.metrics import print_metrics_binary
from mimic3models.in_hospital_mortality.utils import save_results


import os
import numpy as np
import pandas as pd
import argparse
import json


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(
        ret["X"], ret["header"], period, features
    )
    X["Mortality"] = ret["Mortality"]
    X["LOS"] = ret["LOS"]
    X["Age"] = ret["Age"]
    X["Sex"] = ret["Sex"]
    print(ret["Eth"][0])
    X["Eth"] = [eth.split("\n")[0] for eth in ret["Eth"]]

    return (X, ret["name"])


def main():
    print("main")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--C", type=float, default=1.0, help="inverse of L1 / L2 regularization"
    )
    parser.add_argument("--l1", dest="l2", action="store_false")
    parser.add_argument("--l2", dest="l2", action="store_true")
    parser.set_defaults(l2=True)
    parser.add_argument(
        "--period",
        type=str,
        default="all",
        help="specifies which period extract features from",
        choices=[
            "first4days",
            "first8days",
            "last12hours",
            "first25percent",
            "first50percent",
            "all",
        ],
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        help="specifies what features to extract",
        choices=["all", "len", "all_but_len"],
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data of in-hospital mortality task",
        default=os.path.join(
            os.path.dirname(__file__), "../../../data/in-hospital-mortality/"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory relative which all output files are stored",
        default=".",
    )
    args = parser.parse_args()
    print(args)

    train_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, "train"),
        listfile=os.path.join(args.data, "train_listfile.csv"),
        period_length=24.0,
    )

    val_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, "train"),
        listfile=os.path.join(args.data, "val_listfile.csv"),
        period_length=24.0,
    )

    test_reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(args.data, "test"),
        listfile=os.path.join(args.data, "test_listfile.csv"),
        period_length=24.0,
    )

    print("Reading data and extracting features ...")
    # read_and_extract removes some highly implausible values according to plausible_values.json
    (df_val, val_names) = read_and_extract_features(
        val_reader, args.period, args.features
    )
    print(df_val.shape)
    (df_test, test_names) = read_and_extract_features(
        test_reader, args.period, args.features
    )
    print(df_test.shape)
    (df_train, train_names) = read_and_extract_features(
        train_reader, args.period, args.features
    )
    print(df_train.shape)

    def drop_columns_with_substrings(df, substrings):
        # Identify columns to drop
        columns_to_drop = [
            col for col in df.columns if any(sub in col for sub in substrings)
        ]
        # Drop the columns
        df_dropped = df.drop(columns=columns_to_drop)
        return df_dropped

    df = pd.concat(
        [
            df_val,
            df_train,
            df_test,
        ]
    )
    print(df.shape)
    df = drop_columns_with_substrings(df, ["GCSE", "GCSM", "GCSV", "Hours"])
    print(df.shape)
    df.to_csv("df_mimic_final.csv", index=False)


if __name__ == "__main__":
    main()
