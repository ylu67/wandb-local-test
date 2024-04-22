import wandb

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error

wandb.init(project="tdspds-iris-wandb2", name="experiment-2")

X, y = load_iris(return_X_y=True)
label_names = ["Setosa", "Versicolour", "Virginica"]

# Log your model configs to Weights & Biases
params = {"C": 0.1, "random_state": 42}
wandb.config = params

model = LogisticRegression(**params).fit(X, y)
y_pred = model.predict(X)
y_probas = model.predict_proba(X)

wandb.log({
    "accuracy": accuracy_score(y, y_pred),
    "mean_squared_error": mean_squared_error(y, y_pred)
})

wandb.sklearn.plot_roc(y, y_probas, labels=label_names)
wandb.sklearn.plot_precision_recall(y, y_probas, labels=label_names)

wandb.sklearn.plot_confusion_matrix(y, y_pred, labels=label_names)

# Save your model
with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)

# Log your model as a versioned file to Weights & Biases Artifact
artifact = wandb.Artifact(f"iris-logistic-regression-model", type="model")
artifact.add_file("logistic_regression.pkl")
wandb.log_artifact(artifact)

wandb.finish()
