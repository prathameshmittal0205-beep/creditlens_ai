import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score
import pandas as pd
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_credit_model(X, y):
    """
    Trains an XGBoost classifier for loan default prediction.
    Applies SMOTE on training data only. Uses joblib for model persistence.
    """
    model_path = "models/saved_xgb.pkl"
    
    # Check if there's enough samples and classes to split and SMOTE
    if len(X) < 20:
        print("Warning: Dataset too small for reliable credit scoring. Minimum 20 samples required.")
        return None, {}, None, None, None, None
        
    if y.nunique() < 2:
        print("Warning: Target variable y must contain at least two classes (0 and 1) for XGBoost training.")
        return None, {}, None, None, None, None
        
    class_counts = y.value_counts()
    if class_counts.min() < 5:
        print(f"Warning: Dataset too small for reliable credit scoring. At least 5 samples per class required (found {class_counts.min()}).")
        return None, {}, None, None, None, None

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
    
    # Apply SMOTE to training set only
    if y_train.nunique() > 1 and len(y_train) > 10:
        try:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except ValueError:
            # Fallback if not enough samples for nearest neighbors
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train
        
    model.fit(X_train_res, y_train_res)
    
    # Save model securely
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        print(f"Warning: Failed to persist XGBoost model: {e}")
    
    # Evaluate
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
    except Exception as e:
        print(f"Warning: Credit scoring evaluation metrics incomplete: {e}")
        y_prob = y_pred
        roc = 0.5
        fpr, tpr = [0, 1], [0, 1]
        acc = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0
        prec, rec = 0, 0
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]) if len(y_test) > 0 else [[0,0],[0,0]]
    
    metrics = {
        'roc_auc': roc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else fpr,
        'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else tpr,
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
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
