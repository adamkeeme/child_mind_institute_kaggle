from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from read_data import *

def KNN(dataset):
    scaler = StandardScaler()
    dataset["X_train"] = scaler.fit_transform(dataset["X_train"])
    dataset["X_test"] = scaler.transform(dataset["X_test"])

    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(dataset["X_train"], dataset["y_train"].values.ravel())

    y_pred = knn.predict(dataset["X_test"])

    accuracy = accuracy_score(dataset["y_test"], y_pred)
    print(f"Accuracy of KNN with k={k}: {accuracy * 100:.2f}%")

def find_best_k(dataset, k_values):
    scaler = StandardScaler()
    dataset["X_train"] = scaler.fit_transform(dataset["X_train"])
    dataset["X_test"] = scaler.transform(dataset["X_test"])

    best_k = None
    best_accuracy = 0

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(dataset["X_train"], dataset["y_train"].values.ravel())

        y_pred = knn.predict(dataset["X_test"])
        accuracy = accuracy_score(dataset["y_test"], y_pred)
        print(f"Accuracy of KNN with k={k}: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"Best k is {best_k} with accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    datasets = data_load()
    for dataset in datasets.keys():
        print(dataset)
        find_best_k(datasets[dataset])