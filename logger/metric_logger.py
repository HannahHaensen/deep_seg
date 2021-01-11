from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from logger.tensorboard_logger import TensorboardLogger


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
        self.labels = list(labels.keys())
        self.display_labels = list(labels.values())
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

        writer.log_confusion_matrix(tag, self.overall_confusion_matrix, self.display_labels)
        self.compute_current_mean_intersection_over_union(writer=writer)

    def compute_current_mean_intersection_over_union(self, writer):

        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(np.nan_to_num(intersection_over_union))

        writer.log_scalar('mIoU', mean_intersection_over_union)
        return mean_intersection_over_union

    def iou(self, pred, target, n_classes=21):
        ious = []
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1).squeeze(1)
        pred = pred.view(-1)
        target = target.view(-1)

        for cls in range(n_classes):
            pred_idx = (pred == cls)
            target_idx = (target == cls)
            intersection = (pred_idx[target_idx]).long().sum().item()  # Cast to long to prevent overflows

            union = pred_idx.long().sum().item() + target_idx.long().sum().item() - intersection

            if union == 0:
                ious.append(float(0))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
        # print(np.array(ious))
        return np.array(ious)

    def calculate_accuracy(self, output, target):
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy

    def calculate_metrics_for_epoch(self, writer, loss, num_classes, output, target):
        m_io_u_list = self.iou(output, target, num_classes)
        info = {
            'overall/loss': loss,
            'overall/mIoU': np.array(m_io_u_list).mean(),
            'overall/accuracy': self.calculate_accuracy(output, target)  # Compute accuracy
        }
        # Using enumerate()
        for i, val in enumerate(m_io_u_list):
            info[self.display_labels[i] + '/IoU'] = val

        writer.log_scalars(info)
        writer.set_global_step()
        return m_io_u_list
