from ..classifier import *
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# colors for printed examples
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


class ClassificatorEvaluation:
  """
    Class to evaluate a classification model with different metrics and visualize the corresponding plots.
  """
  def __init__(self, model: LightningHiddenStateSAPLMA, test_loader: DataLoader):
    self.model = model
    self.test_loader = test_loader

  def compute_preds(self, threshold: float = 0.5):
    """
      Method to get predictions from the model, together with the corresponding labels, according to some threshold for true predictions.
    """
    self.model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
      for inputs, labels, _ in self.test_loader:
        outputs = self.model(inputs)
        preds = (outputs >= threshold).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels
  
  def print_sampled_examples(self, threshold: float = 0.5):
    """
      Method to get some random examples from the text loader, with the corresponding label and model prediction.
    """
    colors = [RED, GREEN]
    text = ["False", "True"]
    dataset_size = len(self.test_loader.dataset) 
    num_samples = 10 
    idxs = random.sample(range(dataset_size), num_samples)
    sampled_subset = Subset(self.test_loader.dataset, idxs)

    for input, label, _ in sampled_subset:
        pred = (self.model(input) >= threshold).int()
        pred_label = text[pred]
        pred_color = colors[pred]
        input_color = colors[label]
        print(f"Input: {input_color}{input}{RESET}, Prediction: {pred_color}{pred_label}{RESET}")

  def compute_statistics(self, threshold: float = 0.5, print_statistics=True):
    """
      Method to compute the accuracy, precision, recall and F1 score of the model with respect to each label.
    """
    all_preds, all_labels = self.compute_preds(threshold)
    report = classification_report(all_labels, all_preds, output_dict=True)
    if print_statistics: print(classification_report(all_labels, all_preds))
    return report

  def plot_confusion_matrix(self, threshold: float = 0.5):
    """
      Method to plot the confusion matrix of the model.
    """
    all_preds, all_labels = self.compute_preds(threshold)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Ground truth')
    plt.show()
    
  def plot_roc_curve(self, threshold: float = 0.5):
    """
      Method to plot the ROC curve of the model.
    """
    all_preds, all_labels = self.compute_preds(threshold)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random guess')  # Diagonal random guess line

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()