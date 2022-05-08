import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.special import softmax


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape == target.shape
        return torch.sum(pred == target) / target.size(0)


def compute_metrics(title, writer_valid, datasets, results, orig_dataset, epoch):
    ## Accuracy
    accuracy_dict = dict()
    accuracies_sr = []
    accuracies_orig = []
    for dataset in datasets:
        y_pred = np.array(results[dataset]["y_pred"])
        y_true = np.array(results[dataset]["y_true"])

        accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
        accuracy_dict[dataset] = accuracy
        if orig_dataset == dataset:
            accuracies_orig.append(accuracy)
        else:
            accuracies_sr.append(accuracy)

        writer_valid.add_scalar(f'{title}-Accuracy/{dataset}', accuracy, global_step=epoch)
    accuracy_dict['Average'] = (np.mean(accuracies_sr) + np.mean(accuracies_orig)) / 2.
    writer_valid.add_scalar(f'{title}-Accuracy/Average', accuracy_dict['Average'], global_step=epoch)

    ## Roc-Auc-Score
    roc_auc_scores = []
    roc_auc_scores_dict = dict()
    for dataset in datasets:
        if orig_dataset == dataset:
            continue
        y_pred = np.concatenate((np.array(results[dataset]["y_pred"]),
                                 np.array(results[orig_dataset]["y_pred"])))
        y_true = np.concatenate((np.array(results[dataset]["y_true"]),
                                 np.array(results[orig_dataset]["y_true"])))

        auc_score = roc_auc_score(y_true, softmax(y_pred, axis=-1)[:, -1])
        roc_auc_scores.append(auc_score)
        roc_auc_scores_dict[dataset] = auc_score
        writer_valid.add_scalar(f'{title}-Roc-Auc-score/{dataset}', auc_score, global_step=epoch)
    roc_auc_scores_dict['Average'] = np.mean(roc_auc_scores)
    writer_valid.add_scalar(f'{title}-Roc-Auc-score/Average', np.mean(roc_auc_scores), global_step=epoch)

    ## F-1 measure
    f1_measures = []
    f1_measures_dict = dict()
    for dataset in datasets:
        if orig_dataset == dataset:
            continue
        y_pred = np.concatenate((np.array(results[dataset]["y_pred"]),
                                 np.array(results[orig_dataset]["y_pred"])))
        y_true = np.concatenate((np.array(results[dataset]["y_true"]),
                                 np.array(results[orig_dataset]["y_true"])))

        f1_score_ = f1_score(y_true, np.argmax(y_pred, axis=1))
        f1_measures.append(f1_score_)
        f1_measures_dict[dataset] = f1_score_
        writer_valid.add_scalar(f'{title}-F1-measure/{dataset}', f1_score_, global_step=epoch)
    f1_measures_dict['Average'] = np.mean(f1_measures)
    writer_valid.add_scalar(f'{title}-F1-measure/Average', f1_measures_dict['Average'], global_step=epoch)
    return accuracy_dict, roc_auc_scores_dict, f1_measures_dict
