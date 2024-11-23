from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from read_data import *

def preprocess_pipeline(dataset):
    if "id" in dataset["X_train"].columns:
        dataset["X_train"] = dataset["X_train"].drop(columns=["id"])
        dataset["X_test"] = dataset["X_test"].drop(columns=["id"])

    categorical_features = dataset["X_train"].select_dtypes(include=['object']).columns
    numeric_features = dataset["X_train"].select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    return preprocessor

def evaluate_knn(dataset, k):
    preprocessor = preprocess_pipeline(dataset)

    knn = KNeighborsClassifier(n_neighbors=k)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('knn', knn)])

    pipeline.fit(dataset["X_train"], dataset["y_train"].values.ravel())
    y_pred = pipeline.predict(dataset["X_test"])

    accuracy = accuracy_score(dataset["y_test"], y_pred)
    return accuracy


def find_best_k(dataset, k_values):
    best_k = None
    best_accuracy = 0

    for k in k_values:
        accuracy = evaluate_knn(dataset, k)
        # print(f"Accuracy of KNN with k={k}: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    print(f"Best k is {best_k} with accuracy: {best_accuracy * 100:.2f}%")




def dt(dataset):
    preprocessor = preprocess_pipeline(dataset)

    dt_classifier = DecisionTreeClassifier(random_state=42)


    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('dt_classifier', dt_classifier)])

    pipeline.fit(dataset["X_train"], dataset["y_train"].values.ravel())
    y_pred = pipeline.predict(dataset["X_test"])

    accuracy = accuracy_score(dataset["y_test"], y_pred)
    print(f"Accuracy of dt: {accuracy * 100:.2f}%")

    return accuracy



if __name__ == "__main__":
    datasets = data_load()
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for dataset_name, dataset in datasets.items():
        print(dataset_name)
        find_best_k(dataset, k_values)
        # dt(dataset)
