import sys
import os
from typing import List

from sklearn.model_selection import KFold

def save_to_file(directory: str, filename:str, items: List[str]):
    os.makedirs(directory,exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w') as fw:
        for item in items:
            print(item, file=fw)

if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        lines = fp.readlines()
        num_folds = int(sys.argv[2])
        splits = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        for fold_index, (train, val) in enumerate(splits.split(lines)):
            train_lines = [lines[i].strip() for i in train]
            val_lines = [lines[i].strip() for i in val]
            save_to_file(f"{num_folds}fold/fold-{fold_index}", "train.txt", train_lines)
            save_to_file(f"{num_folds}fold/fold-{fold_index}", "val.txt", val_lines)



