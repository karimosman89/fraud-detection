# src/model.py
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_model():
    # Load the preprocessed data
    preprocessed_data_path = os.path.join('data', 'preprocessed_data.pkl')
    X_train, X_test, y_train, y_test = pd.read_pickle(preprocessed_data_path)

    # Train XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Train LightGBM model
    lgb_model = LGBMClassifier()
    lgb_model.fit(X_train, y_train)

    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_lgb = lgb_model.predict(X_test)

    # Evaluation
    print("XGBoost Model Evaluation:")
    print(classification_report(y_test, y_pred_xgb))
    print(confusion_matrix(y_test, y_pred_xgb))

    print("LightGBM Model Evaluation:")
    print(classification_report(y_test, y_pred_lgb))
    print(confusion_matrix(y_test, y_pred_lgb))

if __name__ == "__main__":
    train_model()

