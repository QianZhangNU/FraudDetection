from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

def performance_visul(xgb_model, X_train, X_valid, X_test, y_train, y_valid, y_test, threshold):
    y_scores_train = xgb_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_scores_train > threshold).astype(int)
    y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_train_proba)
    logloss_train = log_loss(y_train, y_scores_train)
    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)

    # Validation set
    y_scores_valid = xgb_model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_scores_valid > threshold).astype(int)
    y_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]
    roc_auc_valid = roc_auc_score(y_valid, y_valid_proba)
    logloss_valid = log_loss(y_valid, y_scores_valid)
    precision_valid = precision_score(y_valid, y_valid_pred)
    recall_valid = recall_score(y_valid, y_valid_pred)

    #Testing set
    y_scores_test = xgb_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_scores_test > threshold).astype(int)
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_test_proba)
    logloss_test = log_loss(y_test, y_scores_test)
    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)

    # Plotting
    labels = ['AUC', 'Log Loss', 'Precision', 'Recall']
    train_metrics = [roc_auc_train, logloss_train, precision_train, recall_train]
    valid_metrics = [roc_auc_valid, logloss_valid, precision_valid, recall_valid]
    test_metrics  = [roc_auc_test, logloss_test, precision_test, recall_test] 

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    rects1 = ax.bar(x - width/2, train_metrics, width, label='Train')
    rects2 = ax.bar(x + width/2, valid_metrics, width, label='Validation')
    rects3 = ax.bar(x + width*1.5, test_metrics, width, label='Test')

    ax.set_ylabel('Scores')
    ax.set_title('Training, Validation and Testing Metrics from XGB')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    rects=rects1
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')       
    rects=rects2
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')  
    rects=rects3
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')     
    #add_annotations(rects1)
    #add_annotations(rects2)
    #add_annotations(rects3)
    plt.show()
    
def Threshold_visual(model, X_valid, y_valid):
    threshold_range = np.arange(0, 1.1, 0.1)
    Precision = []
    Recall = []
    F1_score = []
    y_valid_proba = model.predict_proba(X_valid)

    min_difference = 100
    optimal_threshold = 2
    optimal_precision = 2
    optimal_recall = 2
    optimal_F1_score = 2

    for threshold in threshold_range:
        y_valid_pred = (y_valid_proba[:, 1] > threshold).astype(int) 
        precision = precision_score(y_valid, y_valid_pred)
        recall = recall_score(y_valid, y_valid_pred)
        f1 = f1_score(y_valid, y_valid_pred)
    
        metric_distance =  abs(precision - recall) + abs(precision - f1) + abs(recall - f1)
        if metric_distance < min_difference and f1 > 0.1:
            min_difference = metric_distance
            optimal_threshold = threshold
            optimal_precision = round(precision,2)
            optimal_recall = round(recall,2)
            optimal_F1_score = round(f1,2)
        
        Precision.append(precision)
        Recall.append(recall)
        F1_score.append(f1)
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Precision, Recall, F1 score across thresholds
    sns.lineplot(ax=axes[0], x=threshold_range, y=Precision, label='Precision', color='r')
    sns.lineplot(ax=axes[0], x=threshold_range, y=Recall, label='Recall', color='b')
    sns.lineplot(ax=axes[0], x=threshold_range, y=F1_score, label='F1 Score', color='g')

    axes[0].legend(loc='lower left')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Metric Score')
    axes[0].set_title('Metric Scores Across Thresholds')

    # ROC Curve
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_valid_proba[:, 1])
    sns.lineplot(ax=axes[1], x=fpr, y=tpr, color='b')

    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate or Recall')

    # Precision-Recall Curve
    y_scores = model.predict_proba(X_valid)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_valid, y_scores)
    average_precision = average_precision_score(y_valid, y_scores)
    sns.lineplot(ax=axes[2], x=recall, y=precision, label=f'Average_Precision = {average_precision:.2f}', color='r')

    axes[2].set_title('Precision-Recall Curve')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')

    plt.tight_layout()
    plt.show()
    
def Feature_Importance_Plot(xgrg_model,X_train):
    importances = xgrg_model[0].feature_importances_

    #feature_names = ""
    feature_names = list(X_train.columns)

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Plot the bar chart
    plt.figure(figsize=(25, 5))
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance in Decision Tree Regression")
    plt.show()