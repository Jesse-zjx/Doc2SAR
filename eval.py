import os
import argparse
import pandas as pd

from utils.metrics import affinity_match
from utils.reformat import reformat_pd


def main(test_file, pred_files, test_path, pred_path):
    max_recall = 0.
    base_name = test_file.rsplit('_', 1)[0]
    match_files = [f for f in pred_files if base_name in f]
    
    if len(match_files) == 0:
        return max_recall, None
    match_file = match_files[0]

    target_df = pd.read_csv(os.path.join(test_path, test_file), dtype=str).dropna(how='all')
    for pred_file in match_files:
        try:

            pred_df = pd.read_csv(os.path.join(pred_path, pred_file), dtype=str).dropna(how='all')

            target_extract_df = reformat_pd(target_df)
            pred_extract_df = reformat_pd(pred_df)

            compare_fields = list(target_extract_df.columns)
            metrics = affinity_match(target_extract_df, pred_extract_df, compare_fields=compare_fields)
            recall_value = metrics["recall_value"]
            if recall_value >= max_recall:
                max_recall = recall_value
                match_file = pred_file
        except Exception as e:
            print(e)
            print()
    return max_recall, match_file


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, type=str)
    parser.add_argument('--test', required=True, type=str)
    args = parser.parse_args()

    total_count = 0
    total_recall = 0.0


    test_files = os.listdir(args.test)
    pred_files = os.listdir(args.pred)

    for file in test_files:
        try:
            recall, pred_file = main(file, pred_files, args.test, args.pred)
            if pred_file is None:
                continue
            total_recall += recall
            total_count += 1
        except Exception as e:
            pass

    print(total_recall / total_count)
    print(total_count)
