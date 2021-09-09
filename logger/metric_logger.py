from itertools import product
from math import floor, ceil
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

from dataset.voc_seg_dataset import get_color_for_classes
from logger.training_logger import TrainingLogger


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
        self.best_prediction_train = 0
        self.best_prediction_eval = 0
        self.color_palette = get_color_for_classes()

    def update_matrix(self, ground_truth: Any, prediction: Any,
                      tag: str, writer: TrainingLogger):
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
        """
        calculate interseciton over union
        :param pred:
        :param target:
        :param n_classes:
        :return:
        """
        i_o_u = []
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
                i_o_u.append(float(0))  # If there is no ground truth, do not include in evaluation
            else:
                i_o_u.append(float(intersection) / float(max(union, 1)))
        return np.array(i_o_u)

    def calculate_accuracy_per_class(self, outputs, targets, num_classes: int = 21):
        """
        TODO
        :param outputs:
        :param targets:
        :param num_classes:
        :return:
        """
        acc = []
        _, preds = torch.max(outputs.data, 1)
        for c in range(num_classes):
            acc.append(((preds == targets).all() * (targets == 0)).all().float() / (max(targets == c).sum(), 1))
        return acc

    def calculate_accuracy(self, output, target):
        """
        caclulate overall accuracy for one step or epoch
        :param output:
        :param target:
        :return:
        """
        _, argmax = torch.max(output, 1)
        accuracy = (target == argmax.squeeze()).float().mean()
        return accuracy

    def calculate_classification_metrics_for_epoch(self, writer: TrainingLogger,
                                                   loss: float,
                                                   outputs,
                                                   labels,
                                                   image,
                                                   is_train: bool = False) -> [float]:
        """
        calculates metrics for classification task
        :param writer:
        :param loss:
        :param outputs:
        :param labels:
        :param image:
        :param is_train:
        :return:
        """
        _, predictions = torch.max(outputs, dim=1)
        correct = torch.sum(predictions == labels).item()
        total = labels.size(0)
        acc = correct/total
        info = {
            'epoch/loss': loss,
            'epoch/accuracy': acc # Compute accuracy
        }
        if is_train:
            if self.best_prediction_train < acc:
                self.best_prediction_train = acc
                # writer.log_image('train/best_prediction', image)
        else:
            if self.best_prediction_eval < acc:
                self.best_prediction_eval = acc
                # writer.log_image('eval/best_prediction', image)

        writer.log_scalars(info)
        writer.set_global_step()

        return correct, total, predictions


    def calculate_seg_metrics_for_epoch(self, writer: TrainingLogger,
                                        loss: float, output, target,
                                        image,
                                        is_train: bool = False,
                                        num_classes: int = 21) -> [float]:
        """
        calculates the metrics for a semantic segmentation task
        Loss, mIoU, IoU per class, generall ACC
        :param writer:
        :param loss:
        :param output:
        :param target:
        :param image:
        :param is_train: if is train log best image
        :param num_classes:
        :return:
        """
        m_io_u_per_class = self.iou(output, target, num_classes)
        m_i_o_u = np.array(m_io_u_per_class).mean()

        # m_acc_per_class = self.calculate_accuracy_per_class(outputs=output, targets=target, num_classes=num_classes)
        info = {
            'overall/loss': loss,
            'overall/mIoU': m_i_o_u,
            'overall/accuracy': self.calculate_accuracy(output, target)  # Compute accuracy
        }
        # Using enumerate()
        for i in range(num_classes):
            info[self.display_labels[i] + '/IoU'] = m_io_u_per_class[i]
            # info[self.display_labels[i] + '/Accuracy'] = m_acc_per_class[i]

        _, preds = torch.max(output.data, 1)
        if is_train:
            if self.best_prediction_train < m_i_o_u:
                self.best_prediction_train = m_i_o_u
                self.log_images_to_board(image, preds, target, writer, 'train')
        else:
            if self.best_prediction_eval < m_i_o_u:
                self.best_prediction_eval = m_i_o_u
                self.log_images_to_board(image, preds, target, writer, 'eval')

        writer.log_scalars(info)
        writer.set_global_step()

        return m_io_u_per_class

    def log_images_to_board(self, image, preds, target, writer: TrainingLogger, tag: str = ''):
        rgb_target = self.create_rgb_target(target)
        rgb_pred = self.create_rgb_target(preds)
        writer.log_image(tag + '/best_prediction', rgb_pred)
        writer.log_image(tag + '/best_target', rgb_target)
        writer.log_image(tag + '/best_image', image.numpy())

    def create_rgb_target(self, target):
        """
        conversion from normalized tensor
        :param target:
        :return:
        """
        target = target[-1, :, :]
        h, w = target.shape
        rgb_target = np.random.randint(255, size=(h, w, 3), dtype=np.uint8)
        for pos in product(range(h), range(w)):
            pixel = target[pos[0]][pos[1]]
            rgb_target[pos[0]][pos[1]] = self.color_palette[pixel.item()]
        rgb_target = np.transpose(rgb_target, (2, 0, 1))
        return rgb_target

    def intersection_over_union(boxA, boxB):
        """
        TODO for object detection
        :param boxB:
        :return:
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return (interArea / float(boxAArea + boxBArea - interArea))

    def convert_bbox(old_size, new_size, x_min, y_min, x_max, y_max):
        """
        TODO for object detection
        """
        x_min_new = floor((x_min / old_size[0]) * new_size[0])
        y_min_new = floor((y_min / old_size[1]) * new_size[1])
        x_max_new = ceil((x_max / old_size[0]) * new_size[0])
        y_max_new = ceil((y_max / old_size[1]) * new_size[1])

        return [x_min_new, y_min_new, x_max_new, y_max_new]
