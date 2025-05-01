import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


def xgb_classifier(X_train, y_train, param_grid, num_class=None, random_state=7):
    objective = 'multi:softmax'
    xgb_classifier = XGBClassifier(objective=objective, num_class=num_class, random_state=random_state)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    return grid_search.best_estimator_


def plot_feature_importance(X, y, classifier, num_class):
    import matplotlib.pyplot as plt
    y_pred_proba = classifier.predict_proba(X)
    
    if num_class == 2:
        auc = roc_auc_score(y, y_pred_proba[:, 1])
        print(f"AUC: {auc}")
    else:
        class_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average=None)
        print("Class-wise AUC:", class_auc)
    
    feature_importance = classifier.feature_importances_
    print("Feature Importance:", feature_importance)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in XGBoost Classifier')
    plt.show()



if __name__ == '__main__':
    # training of metadata model and combined model as an example

    # ========================
    # DKD Metadata Model
    # ========================
    
    # Load DKD data
    train_dkd = pd.read_csv('data/dkd-train.csv')
    test_dkd = pd.read_csv('data/dkd-test.csv')
    
    # Metadata features
    dkd_metadata_features = ['Age','Gender','BMI','smoke','DiabetesDuration',
                             'HbA1c','SBP','DBP','DR']
    
    # Parameter grid
    param_grid_dkd_meta = {
        'learning_rate': [0.1, 0.15, 0.2, 0.25],
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [25, 30, 35, 40]
    }
    
    # Train-test split
    X = train_dkd[dkd_metadata_features].values
    y = train_dkd['DKD'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    
    # Train and evaluate
    dkd_metadata_model = xgb_classifier(X_train, y_train, param_grid_dkd_meta, 
                                      num_class=4, random_state=7)
    plot_feature_importance(X_test, y_test, dkd_metadata_model, num_class=4)

    # ========================
    # DKD Combined Model 
    # ========================
    
    # Combined features, prob_i correspond to image-derived probabilities from DKD classifier
    dkd_combined_features = ['Age','Gender','BMI','smoke','DiabetesDuration',
                             'HbA1c','SBP','DBP','DR','prob_0','prob_1','prob_2','prob_3']

    # Parameter grid
    param_grid_dkd_comb = {
    'learning_rate': [0.1, 0.15, 0.2, 0.25],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [14, 16, 18, 20, 22]
    }
    
    # Train-test split
    X = train_dkd[dkd_combined_features].values
    y = train_dkd['DKD'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    
    # Train and evaluate
    dkd_combined_model = xgb_classifier(X_train, y_train, param_grid_dkd_comb, 
                                      num_class=4, random_state=7)
    plot_feature_importance(X_test, y_test, dkd_combined_model, num_class=4)


    # ========================
    # DN Metadata Model
    # ========================
    
    # Load DN data
    train_dn = pd.read_csv('data/dn-train.csv')
    test_dn = pd.read_csv('data/dn-test.csv')
    
    # Metadata features
    dn_metadata_features = ['Age','Gender','BMI','smoke','DiabetesDuration',
                             'HbA1c','SBP','DBP','DR_grade','tc','hematuria','proteinuria']
    
    # Parameter grid
    param_grid_dn_meta = {
        'learning_rate': [0.1, 0.15, 0.2, 0.25],
        'max_depth': [2, 3, 4, 5, 6],
        'n_estimators': [15, 18, 20, 22, 25]
    }
    
    # Train-test split
    X = train_dn[dn_metadata_features].values
    y = train_dn['DN'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    
    # Train and evaluate
    dn_metadata_model = xgb_classifier(X_train, y_train, param_grid_dn_meta, 
                                      num_class=2, random_state=7)
    plot_feature_importance(X_test, y_test, dn_metadata_model, num_class=2)

    # ========================
    # DN Combined Model 
    # ========================
    
    # Combined features, prob_i correspond to image-derived probabilities from DN classifier
    dn_combined_features = ['Age','Gender','BMI','smoke','DiabetesDuration','HbA1c',
                             'SBP','DBP','DR_grade','tc','hematuria','proteinuria','prob_0','prob_1']

    # Parameter grid
    param_grid_dn_comb = {
    'learning_rate': [0.1, 0.15, 0.2, 0.25],
    'max_depth': [2, 3, 4, 5, 6],
    'n_estimators': [14, 16, 18, 20, 22]
    }
    
    # Train-test split
    X = train_dn[dn_combined_features].values
    y = train_dn['DN'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    
    # Train and evaluate
    dn_combined_model = xgb_classifier(X_train, y_train, param_grid_dn_comb, 
                                      num_class=2, random_state=7)
    plot_feature_importance(X_test, y_test, dn_combined_model, num_class=2)






    

