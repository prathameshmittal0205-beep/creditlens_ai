import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd

def train_credit_model(X, y):
    """
    Trains an XGBoost classifier for loan default prediction.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define and train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, metrics, X_train, X_test, y_train, y_test

def get_feature_importance(model, feature_names):
    """
    Extracts feature importance from XGBoost model.
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df

def generate_recommendation(probability):
    """
    Generates approval recommendation based on default probability.
    probability is P(Default).
    """
    if probability < 0.20:
        return 'Approve'
    elif probability < 0.50:
        return 'Review'
    else:
        return 'Reject'
