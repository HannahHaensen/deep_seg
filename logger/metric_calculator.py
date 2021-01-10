from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from logger.tensorboard_logging import TensorboardLogger


class MetricCalculator:
    """Running Confusion Matrix class that enables computation of confusion matrix
       on the go and has methods to compute such accuracy metrics as Mean Intersection over
       Union MIOU.

       Attributes
       ----------
       labels : list[int]
           List that contains int values that represent classes.
       overall_confusion_matrix : sklean.confusion_matrix object
           Container of the sum of all confusion matrices. Used to compute MIOU at the end.
       ignore_label : int
           A label representing parts that should be ignored during
           computation of metrics

       """

    def __init__(self, labels, ignore_label=255):
        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None

    def update_matrix(self, ground_truth: Any, prediction: Any,
                      tag: str, writer: TensorboardLogger):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
            An array with predictions
            :param tag:
            :param writer:
            :param prediction: array, shape = [n_samples]
            :param ground_truth: array, shape = [n_samples]
        """

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            return

        current_confusion_matrix = confusion_matrix(y_true=np.array(ground_truth, dtype=np.float32),
                                                    y_pred=np.array(prediction, dtype=np.float32),
                                                    labels=self.labels)

        if self.overall_confusion_matrix is not None:

            self.overall_confusion_matrix += current_confusion_matrix
        else:

            self.overall_confusion_matrix = current_confusion_matrix
        writer.log_confusion_matrix(tag, self.overall_confusion_matrix)

    def compute_current_mean_intersection_over_union(self, writer):

        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)

        writer.log_scalar('mIoU', mean_intersection_over_union)
        return mean_intersection_over_union

    def calculate_accuracy(self, output, target, writer: TensorboardLogger):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        writer.log_scalar('accuracy', accuracy)

