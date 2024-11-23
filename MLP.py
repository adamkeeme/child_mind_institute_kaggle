from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from read_data import *

def mlp_train(dataset):
    if "id" in dataset["X_train"].columns:
        dataset["X_train"] = dataset["X_train"].drop(columns=["id"])
        dataset["X_test"] = dataset["X_test"].drop(columns=["id"])

    categorical_features = dataset["X_train"].select_dtypes(include=['object']).columns
    numerical_features = dataset["X_train"].select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('mlp', mlp)])

    pipeline.fit(dataset["X_train"], dataset["y_train"].values.ravel())

    y_pred = pipeline.predict(dataset["X_test"])

    accuracy = accuracy_score(dataset["y_test"], y_pred)
    print(f"Accuracy of MLP: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    datasets = data_load()
    for dataset_name, dataset in datasets.items():
        print(dataset_name)
        mlp_train(dataset)
