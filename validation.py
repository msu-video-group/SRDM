import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json

from utils import visualize_images
from metric import compute_metrics, accuracy


def validate(model, datasetValidVideoOriginal, datasetValidVideoCompressed, device, writer_valid, epoch, config):
    model.eval()
    with torch.no_grad():
        all_results = dict()
        orig_dataset = "GT"

        # Validation on original videos ================================================================================
        title = "OriginalVideo"
        results = run_validation_circle(model, datasetValidVideoOriginal, device, writer_valid, epoch)
        accuracy_dict, roc_auc_scores_dict, f1_measures_dict = compute_metrics(title, writer_valid,
                                                                               datasetValidVideoOriginal,
                                                                               results,
                                                                               orig_dataset, epoch)
        all_results[title] = {
            'results': results,
            'metrics': {
                'Accuracy': accuracy_dict,
                'Roc-Auc-score': roc_auc_scores_dict,
                'F1-measure': f1_measures_dict
            }
        }

        # Validation on compressed videos ==============================================================================
        title = "CompressedVideo"
        results = run_validation_circle(model, datasetValidVideoCompressed, device, writer_valid, epoch)
        accuracy_dict, roc_auc_scores_dict, f1_measures_dict = compute_metrics(title, writer_valid,
                                                                               datasetValidVideoCompressed,
                                                                               results, orig_dataset, epoch)
        all_results[title] = {
            'results': results,
            'metrics': {
                'Accuracy': accuracy_dict,
                'Roc-Auc-score': roc_auc_scores_dict,
                'F1-measure': f1_measures_dict
            }
        }

        file_name = f"{config.LOG.METRICS}/{epoch}.json"
        with open(file_name, "w") as fout:
            json.dump(all_results, fout)


def run_validation_circle(model, datasets, device, writer_val, epoch):
    results = dict()
    with torch.no_grad():
        for dataset in datasets:
            T = tqdm(enumerate(datasets[dataset]), desc=f'Process {dataset}')

            epoch_loss = []
            epoch_accuracy = []

            y_true, y_pred = [], []
            for i, (inputs, labels) in T:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = F.cross_entropy(outputs, labels)

                # statistics
                y_pred += outputs.detach().cpu().numpy().tolist()
                y_true += labels.cpu().detach().numpy().flatten().tolist()

                epoch_loss.append(loss.cpu().detach().item())
                epoch_accuracy.append(accuracy(outputs, labels).detach().cpu().item())

                T.set_postfix_str(f"loss: {np.mean(epoch_loss):.5f}, accuracy: {np.mean(epoch_accuracy):.5f}",
                                  refresh=False)

                if i == 0:
                    writer_val.add_image('Images/val', visualize_images(inputs, outputs, labels),
                                         global_step=epoch)
            results[dataset] = {"y_true": y_true, "y_pred": y_pred}
    return results
