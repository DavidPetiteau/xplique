from xplique.attributions.base import BlackBoxExplainer
import tensorflow as tf
from typing import Iterable, Tuple, Union, Optional
import numpy as np
import abc


class IIouCalculator:

    @abc.abstractmethod
    def intersect(self, o1: tf.Tensor, o2: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched objects
        An object could be box, segmentation mask...
        :param o1: object1
        :param o2: object2
        :return: a score between [0,1] corresponding to the intersection of the 2 objects.
        """
        raise NotImplementedError()


class SegmentationIouCalculator(IIouCalculator):

    def intersect(self, mask1: tf.Tensor, mask2: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched segmentation mask.
        The segmentation is a boolean mask on the whole image
        :param mask1: mask of segmentation 1
        :param mask2: mask of segmentation 2
        :return: iou score
        """
        axis = np.arange(1, len(tf.shape(mask1)))
        inter_area = tf.reduce_sum(tf.cast(tf.logical_and(mask1, mask2), dtype=tf.float32), axis=axis)
        union_area = tf.reduce_sum(tf.cast(tf.logical_or(mask1, mask2), dtype=tf.float32), axis=axis)

        return inter_area / tf.maximum(union_area, 1.0)


class BoxIouCalculator(IIouCalculator):
    EPSILON = tf.constant(1e-4)

    def intersect(self, boxA: tf.Tensor, boxB: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched bounding boxes.
        The bounding box is defined by (x1, y1, x2, y2)
        :param boxA: bounding box 1
        :param boxB: bounding box 2
        :return: iou score
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = tf.maximum(boxA[..., 0], boxB[..., 0])
        yA = tf.maximum(boxA[..., 1], boxB[..., 1])
        xB = tf.minimum(boxA[..., 2], boxB[..., 2])
        yB = tf.minimum(boxA[..., 3], boxB[..., 3])
        # compute the area of intersection rectangle
        interArea = tf.maximum(0, xB - xA) * tf.maximum(0, yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[..., 2] - boxA[..., 0]) * (boxA[..., 3] - boxA[..., 1])
        boxBArea = (boxB[..., 2] - boxB[..., 0]) * (boxB[..., 3] - boxB[..., 1])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea + BoxIouCalculator.EPSILON)
        # return the intersection over union value
        return iou

class IObjectFormater:

    def format_objects(self, predictions) -> Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Format the model prediction of a given image to have the prediction of the following format:
        objects, proba_detection, one_hots_classifications
        :param predictions: prediction of the model of a given image
        :return: list of tuple: "object,proba,classification"
        """
        raise NotImplementedError()


class ImageObjectDetectorScoreCalculator:

    def __init__(self, object_formater: IObjectFormater, iou_calculator: IIouCalculator):
        self.object_formater = object_formater
        self.iou_calculator = iou_calculator

    def score(self, model, x, object_ref) -> tf.Tensor:
        """
        Compute the matching score between prediction and a given object
        :param model: the model used for the object detection
        :param x: the batched image
        :param object_ref: the object target to compare with the prediction of the model
        :return: For each image, the matching score between the object of reference and the prediction of the model
        """
        objects = model(x)
        score_values = []
        for o, o_ref in zip(objects, object_ref):
            if o is None:
                score_values.append(tf.constant(0.0, dtype=x.dtype))
            else:
                current_boxes, proba_detection, classification = self.object_formater.format_objects(o)

                if len(tf.shape(o_ref)) == 1:
                    o_ref = tf.expand_dims(o_ref, axis=0)

                o_ref = self.object_formater.format_objects(o_ref)

                scores = []
                size = tf.shape(current_boxes)[0]
                for boxes_ref, proba_ref, class_ref in zip(*o_ref):
                    boxes_ref = tf.repeat(tf.expand_dims(boxes_ref, axis=0), repeats=size, axis=0)
                    #proba_ref = tf.repeat(tf.expand_dims(proba_ref,axis=0), repeats=size, axis=0)
                    class_ref = tf.repeat(tf.expand_dims(class_ref, axis=0), repeats=size, axis=0)

                    iou = self.iou_calculator.intersect(boxes_ref, current_boxes)
                    classification_similarity = tf.reduce_sum(class_ref * classification, axis=1) / (
                            tf.norm(classification, axis=1) * tf.norm(class_ref, axis=1))

                    current_score = iou * tf.squeeze(proba_detection, axis=1) * classification_similarity
                    current_score = tf.reduce_max(current_score)
                    scores.append(current_score)

                score_value = tf.reduce_max(tf.stack(scores))
                score_values.append(score_value)

        score_values = tf.stack(score_values)

        return score_values


class ImageObjectDetectorExplainer(BlackBoxExplainer):
    def __init__(self, explainer: BlackBoxExplainer, object_detector_formater: IObjectFormater,
                 iou_calculator: IIouCalculator):
        """
        :param explainer: the black box explainer used to explain the object detector model
        :param object_detector_formater: the formater of the object detector model used to format the prediction of
                                         the right format
        :param iou_calculator: the iou calculator used to compare two objects.
        """
        super().__init__(explainer.model, explainer.batch_size)
        self.explainer = explainer
        self.score_calculator = ImageObjectDetectorScoreCalculator(object_detector_formater, iou_calculator)
        self.explainer.inference_function = self.score_calculator.score

    def explain(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
        if len(tf.shape(targets)) == 1:
            targets = tf.expand_dims(targets, axis=0)

        return self.explainer.explain(inputs, targets)


class BoundingBoxesExplainer(ImageObjectDetectorExplainer, IObjectFormater):
    """
    For a given black box explainer, this class allows to find explications of an object detector model
    The object model detector shall return a list (length of the size of the batch) containing a tensor of 2 dimensions.
    The first dimension of the tensor is the number of bounding boxes found in the image
    The second dimension is: [x1_boxe, y1_boxe, x2_boxe, y2_boxe, probability_detection, ones_hot_classif_result]

    This work is a generalisation of the following article at any kind of black box explainer and also can be used for
    other kind of object detector (like segmentation)
    Ref. Petsiuk & al., Black-box Explanation of Object Detectors via Saliency Maps (2021).
    https://arxiv.org/pdf/2006.03204.pdf
    """

    def __init__(self, explainer: BlackBoxExplainer):
        super().__init__(explainer, self, BoxIouCalculator())

    def format_objects(self, predictions) -> Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        boxes, proba_detection, one_hots_classifications = tf.split(predictions,
                                                                    [4, 1, tf.shape(predictions[0])[0] - 5], axis=1)
        return boxes, proba_detection, one_hots_classifications
