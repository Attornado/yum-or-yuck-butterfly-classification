from typing import Optional
import tensorflow.experimental.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def decode_predictions(preds: np.ndarray, name_list: list[str], top: int = 1):
    """
    It takes the predictions from the model, and returns the top predictions for each image.

    :param preds: the predictions from the model
    :param name_list: a list of class names
    :param top: the number of results to return for each prediction, defaults to 1 (optional)
    :return: A list of tuples containing the top predictions.
    """
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple([name_list[i]]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)

    return results


def plot_confusion_matrix(cm, cmap=None, title=None,
                          y_labels=None, x_labels=None,
                          binary_flip=True,
                          error_1_text='',
                          error_2_text='',
                          show_annotations=True,
                          quad_line=True,
                          size=(10, 8), savepath: Optional[str] = None):
    """
   It takes a confusion matrix and plots it with a color map, text, and optional annotations.

   :param cm: The confusion matrix to be plotted
   :param cmap: The color map to use
   :param title: The title of the plot
   :param y_labels: The labels for the y-axis
   :param x_labels: The labels for the x-axis
   :param binary_flip: If True, the matrix will be flipped so that the True Positive is in the top-left corner,
    defaults to True (optional)
   :param error_1_text: Text to display in the Type I Error box
   :param error_2_text: The text to display in the False Negative quadrant
   :param show_annotations: True/False, defaults to True (optional)
   :param quad_line: True/False or width of true-positives, defaults to True (optional)
   :param size: The size of the figure.
   :param savepath: optional path to save the figure of the confusion matrix.
   """

    np.experimental_enable_numpy_behavior()
    is_binary = len(cm.ravel()) == 4

    # Binary matrices will unravel in this order:
    # tn, fp, fn, tp = cm.ravel()
    # Binary labels should be in False,True or 0,1 order

    tn_c = (0, 0)  # True Negative top-left corner
    fp_c = (0, 1)
    fn_c = (1, 0)
    tp_c = (1, 1)

    _cm = cm

    if is_binary and binary_flip:
        _cm = np.rot90(np.rot90(cm)).numpy()
        x_labels = list(reversed(x_labels)) if x_labels else None
        y_labels = list(reversed(y_labels)) if y_labels else None
        tn_c = (1, 1)  # True Negative bottom-right corner
        fp_c = (1, 0)
        fn_c = (0, 1)
        tp_c = (0, 0)

    plt.style.use("default")

    if y_labels and not x_labels:
        x_labels = y_labels

    if x_labels and not y_labels:
        y_labels = x_labels

    labels = y_labels  # either None or both labels set

    # Set the optional ax argument so we can change the figsize()
    fig, ax = plt.subplots(figsize=size)

    if title:
        plt.suptitle(title, color="darkgreen", fontsize=20, y=1.1)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=_cm,
        display_labels=labels,
    )

    if not cmap:
        # Contract the color space
        cmap = plt.get_cmap('RdGy')
        min_v = .15
        max_v = .9
        _cmap = cmap(np.linspace(min_v, max_v, 100))
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_v, b=max_v),
            _cmap)
    else:
        cmap = plt.get_cmap(cmap)

    cm_plot = disp.plot(
        ax=ax,
        cmap=cmap,  # 'seismic_r'
        xticks_rotation=37,
        include_values=False,
    )

    ax.set_title("Confusion Matrix", fontsize=18)

    ax.set_ylabel('Actual', fontsize=17)
    ax.set_xlabel('Prediction', fontsize=17)

    if not x_labels:
        x_labels = ax.xaxis.get_ticklabels()

    ax.xaxis.set_ticklabels(x_labels, ha='right')

    ax.tick_params(axis='both', which='major', labelsize=14, labelcolor='dimgrey')

    #
    # Display our own content in the grid
    #

    tm = np.empty_like(_cm, dtype=object).numpy()  # Text Matrix

    # print text with appropriate color depending on background
    cmap_ls = cmap(np.linspace(0, 1.0, (_cm.max() - _cm.min())))
    lum_white = 0.9277833117792471
    lum_black = 0.0

    for j in range(tm.shape[0]):
        for i in range(tm.shape[1]):

            # Determine the highest contrast text color
            r, g, b, a = tuple(cmap_ls[_cm[j, i] - _cm.min() - 1].tolist())

            lum_r = r / 12.92 if r <= 0.03928 else (r + 0.055) / 1.055 ** 2.4
            lum_g = g / 12.92 if g <= 0.03928 else (g + 0.055) / 1.055 ** 2.4
            lum_b = b / 12.92 if b <= 0.03928 else (b + 0.055) / 1.055 ** 2.4

            lum = 0.2126 * lum_r + 0.7152 * lum_g + 0.0722 * lum_b

            contrast_ratio_light = ((max(lum, lum_white) + 0.05) / (min(lum, lum_white) + 0.05))
            contrast_ratio_dark = ((max(lum, lum_black) + 0.05) / (min(lum, lum_black) + 0.05))

            tm_color = 'k' if contrast_ratio_dark > contrast_ratio_light else 'w'

            # Determine the number formatting
            tm_fmt = format(_cm[j, i], ".2g")
            if _cm.dtype.kind != "f":
                text_d = format(_cm[j, i], "d")
                if len(text_d) < len(tm_fmt):
                    tm_fmt = text_d

            tm[j, i] = {'value': tm_fmt, 'color': tm_color}
            ax.text(y=j, x=i, s=tm_fmt, ha="center", va="center", color=tm_color, fontsize=17)

    if is_binary and show_annotations:
        tm[tn_c]['quadrant'] = '(True Negative)'
        tm[fp_c]['quadrant'] = '(False Positive)'
        tm[fn_c]['quadrant'] = '(False Negative)'
        tm[tp_c]['quadrant'] = '(True Positive)'
        tm[fn_c]['description'] = 'Type II Error\n' + error_2_text
        tm[fp_c]['description'] = 'Type I Error\n' + error_1_text

        for j in range(tm.shape[0]):
            for i in range(tm.shape[1]):
                if tm[j, i].get('quadrant'):
                    ax.text(s=f"{tm[j, i]['quadrant']}",
                            y=j, x=i,
                            position=(i, j + 0.25),
                            color=tm[j, i]['color'],
                            fontsize='16',
                            verticalalignment='bottom',
                            horizontalalignment='center',
                            )
                if tm[j, i].get('description'):
                    ax.text(s=f"{tm[j, i]['description']}",
                            y=j, x=i,
                            position=(i, j + 0.3),
                            color=tm[j, i]['color'],
                            fontsize='12',
                            fontstyle='italic',
                            verticalalignment='top',
                            horizontalalignment='center',
                            )

    if quad_line:
        # Check to see we have to calculate line position and we have right shape
        if isinstance(quad_line, bool):
            if len(cm) == len(cm[0]) and len(cm) % 2 == 0:
                quad_line = len(cm) // 2
            else:
                quad_line = 0

        if quad_line > 0:
            ax.axhline(y=quad_line - 0.5, xmin=0.0, xmax=1.0, color='k')
            ax.axvline(x=quad_line - 0.5, ymin=0.0, ymax=1.0, color='k')

    if savepath is not None:
        plt.savefig(f"{savepath}")

    plt.show()
