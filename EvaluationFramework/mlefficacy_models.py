import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Classification imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Regression imports
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score



# Function to clean the data
def clean_data(df):
    # Strip whitespace from all string columns
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()

    # Drop rows with missing values
    df = df.dropna()


    return df
    
# Function to preprocess features for ML models without test-train split
def preprocess_features(X_train, X_test, scaler=None, encoder=None):
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if scaler is None:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[numeric_cols])
    else:
        X_train_num = scaler.transform(X_train[numeric_cols])
    X_test_num = scaler.transform(X_test[numeric_cols])

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    else:
        X_train_cat = encoder.transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    cols = numeric_cols + encoder.get_feature_names_out(categorical_cols).tolist()

    X_train_preprocessed = pd.DataFrame(
        np.hstack([X_train_num, X_train_cat]),
        columns=cols,
        index=X_train.index
    )
    X_test_preprocessed = pd.DataFrame(
        np.hstack([X_test_num, X_test_cat]),
        columns=cols,
        index=X_test.index
    )
    return X_train_preprocessed, X_test_preprocessed, scaler, encoder



# MODELS
# Train a random forest classifier
def eval_random_forest(X_train, y_train, X_test, y_test):
    # Create a Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the model to the training data
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    return accuracy, f1
    
# Train a SVM classifier
from sklearn.svm import SVC

def eval_svm(X_train, y_train, X_test, y_test):
    # Create a SVM Classifier
    svm = SVC(kernel='linear', random_state=42)
    # Fit the model to the training data
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    return accuracy, f1

# Train KNN classifier
def eval_knn(X_train, y_train, X_test, y_test):
    # Create a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    # Fit the model to the training data
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    return accuracy, f1

# Train Logistic Regression
def eval_lr(X_train, y_train, X_test, y_test):
    # Create a Logistic Regression model
    lr = LogisticRegression(random_state=42)
    # Fit the model to the training data
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    return accuracy, f1

# Train XGBoost Classifier
def eval_xgb(X_train, y_train, X_test, y_test):
    # Create an XGBoost Classifier
    xgb = XGBClassifier(random_state=42)
    # Fit the model to the training data
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    return accuracy, f1


## REGRESSION
# Train Linear Regression
def eval_linear_regression(X_train, y_train, X_test, y_test):
    # Create a Linear Regression model
    lr = LinearRegression()
    # Fit the model to the training data
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

# Train XGBoost Regressor
def eval_xgb_regressor(X_train, y_train, X_test, y_test):
    # Create an XGBoost Regressor
    xgb = XGBRegressor(random_state=42)
    # Fit the model to the training data
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2