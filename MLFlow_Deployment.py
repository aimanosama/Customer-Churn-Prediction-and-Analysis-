import mlflow.sklearn
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix)
import tempfile
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)


X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

models = {
    "best_model": "models/best.joblib",
    "XGBoost": "models/XGBoost.joblib",
    "Decision Tree": "models/Decision Tree.joblib",
    "SVC": "models/SVC.joblib",
    "KNN": "models/KNN.joblib",
    "GaussianNB": "models/GaussianNB.joblib",
    "Logistic Regression": "models/Logistic Regression.joblib"
}

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Customer Churn Prediction")

for model_name, model_path in models.items():
    print(f"🚀 Running experiment for {model_name}...")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        continue


    with mlflow.start_run(run_name=model_name):

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, f"{model_name}_cm.png")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)

        plt.close()
        mlflow.sklearn.log_model(model, model_name)

        print(f"✅ {model_name} experiment logged successfully!\n")

print("🎯 All experiments completed successfully.")