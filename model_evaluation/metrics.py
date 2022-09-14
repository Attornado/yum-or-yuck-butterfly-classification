import tensorflow as tf
from typing import final


def f1_score(y_true, y_pred):
    precision_obj = tf.keras.metrics.Precision()
    recall_obj = tf.keras.metrics.Recall()

    precision_obj.update_state()
    return 2 * precision * recall / (precision + recall)

