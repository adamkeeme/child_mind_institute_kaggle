import pandas as pd
import os

def data_load():

    file_with_names = 'datasets/datasets.txt'
    data_folder = 'datasets/imputed_or_complete/divided'
    datasets = {}

    if not os.path.exists(file_with_names):
        raise FileNotFoundError(f"file {file_with_names} doesn't exist")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"folder {data_folder} doesn't exist")

    with open(file_with_names, 'r') as file:
        names = file.read().splitlines()

    for name in names:
        X_train_path = os.path.join(data_folder, f"{name}_X_train.csv")
        X_test_path = os.path.join(data_folder, f"{name}_X_test.csv")
        y_train_path = os.path.join(data_folder, f"{name}_y_train.csv")
        y_test_path = os.path.join(data_folder, f"{name}_y_test.csv")

        if not all(map(os.path.exists, [X_train_path, X_test_path, y_train_path, y_test_path])):
            print(f" {name} doesn't exist")
            continue

        # 读取数据
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

        # 保存数据到字典中
        datasets[name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        print(f"datasets {name} loading")

    return datasets

