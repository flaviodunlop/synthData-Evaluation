import mlefficacy_models as mle
from sklearn.model_selection import KFold
import numpy as np

# Function for ML-Efficacy Classification Evaluation
def mlefficacy_classification(real_data, synth_data, target_column, n_splits, random_state):
    # prepare data structures for ML metrics
    accuracy_real_dict = {}
    accuracy_real_dict['RF'] = []
    accuracy_real_dict['SVM'] = []
    accuracy_real_dict['KNN'] = []
    accuracy_real_dict['LR'] = []
    accuracy_real_dict['XGB'] = []

    accuracy_synth_dict = {}
    accuracy_synth_dict['RF'] = []
    accuracy_synth_dict['SVM'] = []
    accuracy_synth_dict['KNN'] = []
    accuracy_synth_dict['LR'] = []
    accuracy_synth_dict['XGB'] = []

    f1_real_dict = {}
    f1_real_dict['RF'] = []
    f1_real_dict['SVM'] = []
    f1_real_dict['KNN'] = []
    f1_real_dict['LR'] = []
    f1_real_dict['XGB'] = []

    f1_synth_dict = {}
    f1_synth_dict['RF'] = []
    f1_synth_dict['SVM'] = []
    f1_synth_dict['KNN'] = []
    f1_synth_dict['LR'] = []
    f1_synth_dict['XGB'] = []

    # Clean the data
    df_real_cleaned = mle.clean_data(real_data)
    df_synth_cleaned = mle.clean_data(synth_data)

    # Split Features and Targets
    X_real = df_real_cleaned.drop(columns=[target_column])
    y_real = df_real_cleaned[target_column]

    X_synth = df_synth_cleaned.drop(columns=[target_column])
    y_synth = df_synth_cleaned[target_column]

    # KFold Setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Loop through each fold
    for train_index, test_index in kf.split(X_real):
        # Real Data
        X_train_real, X_test_real = X_real.iloc[train_index], X_real.iloc[test_index]
        y_train_real, y_test_real = y_real.iloc[train_index], y_real.iloc[test_index]

        # Synth Data (identical split for both datasets)
        X_train_synth, X_test_synth = X_synth.iloc[train_index], X_synth.iloc[test_index]
        y_train_synth, y_test_synth = y_synth.iloc[train_index], y_synth.iloc[test_index]

        # Preprocessing real data (fit scaler and encoder)
        X_train_preprocessed_real, X_test_preprocessed_real, scaler, encoder = mle.preprocess_features(X_train_real, X_test_real)
        # Preprocessing synthetic data (reuse scaler and encoder)
        X_train_preprocessed_synth, X_test_preprocessed_synth, _, _ = mle.preprocess_features(X_train_synth, X_test_real, scaler=scaler, encoder=encoder)


        # Train and evaluate the model
        # Random Forest - Real Data
        accuracy_real, f1_real = mle.eval_random_forest(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real)
        accuracy_real_dict['RF'].append(accuracy_real)
        f1_real_dict['RF'].append(f1_real)
        # Random Forest - Synthetic Data
        accuracy_synth, f1_synth = mle.eval_random_forest(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real)
        accuracy_synth_dict['RF'].append(accuracy_synth)
        f1_synth_dict['RF'].append(f1_synth)

        # SVM - Real Data
        accuracy_synth, f1_synth = mle.eval_svm(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real)
        accuracy_real_dict['SVM'].append(accuracy_synth)
        f1_real_dict['SVM'].append(f1_synth)
        # SVM - Synthetic Data
        accuracy_synth, f1_synth = mle.eval_svm(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real)
        accuracy_synth_dict['SVM'].append(accuracy_synth)
        f1_synth_dict['SVM'].append(f1_synth)

        # KNN - Real Data
        accuracy_real, f1_real = mle.eval_knn(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real)
        accuracy_real_dict['KNN'].append(accuracy_real)
        f1_real_dict['KNN'].append(f1_real)
        # KNN - Synthetic Data
        accuracy_synth, f1_synth = mle.eval_knn(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real)
        accuracy_synth_dict['KNN'].append(accuracy_synth)
        f1_synth_dict['KNN'].append(f1_synth)

        # Logistic Regression - Real Data
        accuracy_real, f1_real = mle.eval_lr(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real)
        accuracy_real_dict['LR'].append(accuracy_real)
        f1_real_dict['LR'].append(f1_real)
        # Logistic Regression - Synthetic Data
        accuracy_synth, f1_synth = mle.eval_lr(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real)
        accuracy_synth_dict['LR'].append(accuracy_synth)
        f1_synth_dict['LR'].append(f1_synth)

        # XGBoost - Real Data
        accuracy_real, f1_real = mle.eval_xgb(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real)
        accuracy_real_dict['XGB'].append(accuracy_real)
        f1_real_dict['XGB'].append(f1_real)
        # XGBoost - Synthetic Data
        accuracy_synth, f1_synth = mle.eval_xgb(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real)
        accuracy_synth_dict['XGB'].append(accuracy_synth)
        f1_synth_dict['XGB'].append(f1_synth)

    return accuracy_real_dict, accuracy_synth_dict, f1_real_dict, f1_synth_dict

# Function for ML-Efficacy Regression Evaluation with K-Fold
def mlefficacy_regression(real_data, synth_data, target_column, n_splits, random_state):
    r2_real_dict = {'LR': [], 'XGB': []}
    r2_synth_dict = {'LR': [], 'XGB': []}

    # Clean the data
    df_real_cleaned = mle.clean_data(real_data)
    df_synth_cleaned = mle.clean_data(synth_data)

    # Split Features and Targets
    X_real = df_real_cleaned.drop(columns=[target_column])
    y_real = df_real_cleaned[target_column]

    X_synth = df_synth_cleaned.drop(columns=[target_column])
    y_synth = df_synth_cleaned[target_column]

    # KFold Setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Loop through each fold
    for train_index, test_index in kf.split(X_real):
        # Real Data
        X_train_real, X_test_real = X_real.iloc[train_index], X_real.iloc[test_index]
        y_train_real, y_test_real = y_real.iloc[train_index], y_real.iloc[test_index]

        # Synth Data (identical split for both datasets)
        X_train_synth, X_test_synth = X_synth.iloc[train_index], X_synth.iloc[test_index]
        y_train_synth, y_test_synth = y_synth.iloc[train_index], y_synth.iloc[test_index]

        # Preprocessing real data (fit scaler and encoder)
        X_train_preprocessed_real, X_test_preprocessed_real, scaler, encoder = mle.preprocess_features(X_train_real, X_test_real)
        # Preprocessing synthetic data (reuse scaler and encoder)
        X_train_preprocessed_synth, X_test_preprocessed_synth, _, _ = mle.preprocess_features(X_train_synth, X_test_real, scaler=scaler, encoder=encoder)

        # Evaluation
        r2_real_dict['LR'].append(mle.eval_linear_regression(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real))
        r2_synth_dict['LR'].append(mle.eval_linear_regression(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real))

        r2_real_dict['XGB'].append(mle.eval_xgb_regressor(X_train_preprocessed_real, y_train_real, X_test_preprocessed_real, y_test_real))
        r2_synth_dict['XGB'].append(mle.eval_xgb_regressor(X_train_preprocessed_synth, y_train_synth, X_test_preprocessed_real, y_test_real))

    return r2_real_dict, r2_synth_dict


# Evaluation of synthetic data using ML-Efficacy metrics
def run_mlefficacy_evaluation(real_data, synth_data, ml_task, target_column, cv, cv_seeds):
    if 'Classification' in ml_task:
        accuracy_real_dict, accuracy_synth_dict, f1_real_dict, f1_synth_dict = mlefficacy_classification(real_data, synth_data, target_column, n_splits=cv, random_state=cv_seeds[0])

        # ML Efficacy - Difference Accuracy
        mean_accuracy_diff = {
            model: np.mean(np.abs(np.array(accuracy_synth_dict[model]) - np.array(accuracy_real_dict[model])))
            for model in accuracy_real_dict
        }
        result_mean_accuracy_diff = np.mean(list(mean_accuracy_diff.values()))
        # Final inverse Score
        mlefficacy_acc = round((1 - result_mean_accuracy_diff) * 100, 2)

        # ML Efficacy - Difference F1
        mean_f1_diff = {
            model: np.mean(np.abs(np.array(f1_synth_dict[model]) - np.array(f1_real_dict[model])))
            for model in f1_real_dict
        }
        result_mean_f1_diff = np.mean(list(mean_f1_diff.values()))
        # Final inverse Score
        mlefficacy_f1 = round((1 - result_mean_f1_diff) * 100, 2)

        return mlefficacy_acc, mlefficacy_f1

    else: # for regression tasks
        r2_real_dict, r2_synth_dict = mlefficacy_regression(real_data, synth_data, target_column, n_splits=cv, random_state=cv_seeds[0])
        
        # ML Efficacy - Difference R2
        r2_threshold = -2
        mean_r2_diff = {}
        invalid_r2 = False
        valid_count = 0
        
        for model in r2_real_dict:
            r2_real = np.array(r2_real_dict[model])
            r2_synth = np.array(r2_synth_dict[model])

            # masking invalid R2 values
            # True if both r2_real and r2_synth are above the threshold
            valid_mask = np.array((r2_real >= r2_threshold) & (r2_synth >= r2_threshold))
           
            # count valid scores
            valid_count += np.sum(valid_mask)

            if not np.all(valid_mask):
                invalid_r2 = True
                invalid_indices = np.where(~valid_mask)[0]
                print(f"WARNING: Model '{model}': Ignoring folds with indices {invalid_indices} due to exceeded R²-threshold")

            # calculate mean R2 difference only for valid entries
            if np.any(valid_mask):
                diff = np.abs(r2_synth[valid_mask] - r2_real[valid_mask])
                mean_r2_diff[model] = np.mean(diff)
            else: # if all entries are invalid
                print(f"WARNING: Model '{model}': All folds have R² values below the threshold of {r2_threshold}. Returning None.")

        # filter out NaN values from mean_r2_diff
        valid_model_diffs = [v for v in mean_r2_diff.values() if not np.isnan(v)] 


        # calculate the final inverse score if there are valid model differences
        if valid_model_diffs:
            result_mean_r2_diff = np.mean(valid_model_diffs)
            mlefficacy_r2 = round((1 - result_mean_r2_diff) * 100, 2)
            
            # clip the score to a minimum of 0
            if mlefficacy_r2 < 0:
                mlefficacy_r2 = 0
                mlefficacyClipped = True
                print(f'WARNING: ML-Efficacy R² score is negative ({mlefficacy_r2}). Setting it to 0.')
            else:
                mlefficacyClipped = False

        else:
            print('ERORR: No valid R² values in any model. Returning None.')
            mlefficacy_r2 = None
            mlefficacyClipped = False

        return invalid_r2, mlefficacy_r2, valid_count, mlefficacyClipped