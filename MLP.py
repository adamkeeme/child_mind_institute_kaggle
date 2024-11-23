from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def mlp_train(dataset):
    scaler = StandardScaler()
    dataset["X_train"] = scaler.fit_transform(dataset["X_train"])
    dataset["X_test"] = scaler.transform(dataset["X_test"])

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(dataset["X_train"], dataset["y_train"].values.ravel())

    y_pred = mlp.predict(dataset["X_test"])

    accuracy = accuracy_score(dataset["y_test"], y_pred)
    print(f"Accuracy of MLP: {accuracy * 100:.2f}%")
