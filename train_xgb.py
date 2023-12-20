import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score


def xgb_classifier(X_train,y_train,param_grid,num_class=4,random_state=7):
    xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=num_class, random_state=random_state)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    best_classifier = grid_search.best_estimator_
    return best_classifier


def plot_feature_importance(X,y,classifier):
    import matplotlib.pyplot as plt
    y_pred_proba = classifier.predict_proba(X)
    class_auc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average=None)
    print(class_auc)
    feature_importance = classifier.feature_importances_
    print("Feature Importance:", feature_importance)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xticks(range(len(feature_importance)), X.columns)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in XGBoost Classifier')
    plt.show()


if __name__ == '__main__':
    # training of metadata model of DKD as an example
    train = pd.read_csv('data/dkd-train.csv')
    test = pd.read_csv('data/dkd-test.csv')
    X = train[['Age','Gender','BMI','smoke','DiabetesDuration','HbA1c','SBP','DBP','DR']].values
    y = train['DKD'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


    # param_grad in our tasks
    param_grid_dkd = {
        'learning_rate': [0.1, 0.15, 0.2, 0.25],
        'max_depth': [3, 4, 5, 6],
        'n_estimators': [25, 30, 35, 40]
    }

    param_grid_dn = {
        'learning_rate': [0.10, 0.15, 0.20, 0.25],
        'max_depth': [6, 8, 10, 12],
        'n_estimators': [14, 16, 18, 20]
    }

    dkd_metadata_model = xgb_classifier(X_train, y_train, param_grid_dkd, num_class=4, random_state=7)
    plot_feature_importance(X_test, y_test, dkd_metadata_model)
