import os
from typing import final
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, confusion_matrix, classification_report
from model_evaluation.constants import EVALUATION_BATCH_SIZE, BETA_SCORE, AVERAGE_TYPE_BETA_SCORE
from model_evaluation.utils import decode_predictions, plot_confusion_matrix
from training.constants import FITTED_YOYNET_DIR, PLOT_DIR
from preprocessing.constants import EVALUATION_PATH_CLEANED, YUMS, YUCKS, CLASS_NAMES
from preprocessing.utils import decode_image, decode_label, decode_image_id


_VERSION_LOAD: final = 0.4
_EPOCHS_LOAD: final = 100
_MODEL_NAME: final = f"yoynet_{_EPOCHS_LOAD}_epochs_v{_VERSION_LOAD}"
_MODEL_PATH: final = os.path.join(FITTED_YOYNET_DIR, _MODEL_NAME)


def main():
    # Load the dataset and the model
    eval_ds_prebatch = tf.data.experimental.load(EVALUATION_PATH_CLEANED)
    eval_ds = eval_ds_prebatch.batch(EVALUATION_BATCH_SIZE)
    model = tf.keras.models.load_model(_MODEL_PATH)

    # Make predicition and format them into a prediction data frame
    preds = model.predict(eval_ds, verbose=1)
    predicted_label_confidence = decode_predictions(preds, CLASS_NAMES)
    eval_ds_decoded = [
        (decode_image(m), decode_label(l.numpy()), decode_image_id(image_id)) for (m, l, image_id) in eval_ds.unbatch()
    ]

    # Zip the tuples, manually
    predictions_df = pd.DataFrame(
        [(eval_ds_decoded[i][2],  # id
          eval_ds_decoded[i][0],  # X
          eval_ds_decoded[i][1],  # y
          eval_ds_decoded[i][1] in YUMS,  # yum
          predicted_label_confidence[i][0][0],  # y_pred
          predicted_label_confidence[i][0][0] in YUMS,  # yum_pred
          predicted_label_confidence[i][0][1]  # y_conf
          ) for i in range(len(eval_ds_decoded))],
        columns=['id', 'X', 'y', 'yum', 'y_pred', 'yum_pred', 'y_conf']
    )

    # Check out the classification metrics (accuracy, precision, recall, F1-score, ...)
    fbeta = fbeta_score(
        predictions_df[['y']],
        predictions_df[['y_pred']],
        beta=BETA_SCORE,
        average=AVERAGE_TYPE_BETA_SCORE
    )
    print(f"F{BETA_SCORE}-score: {fbeta}")

    butterfly_cr = classification_report(predictions_df[['y']], predictions_df[['y_pred']], digits=5, output_dict=False)
    print("Butterfly Classification Report: \n\n")
    print(butterfly_cr)

    # Create classification and yum/yuck confusion matrices and plot them
    butterfly_cm = confusion_matrix(predictions_df[['y']], predictions_df[['y_pred']])
    quad_line_butterfly_cm = confusion_matrix(predictions_df[['y']], predictions_df[['y_pred']], labels=YUMS+YUCKS)
    yum_cm = confusion_matrix(predictions_df[['yum']], predictions_df[['yum_pred']])

    plot_confusion_matrix(
        butterfly_cm,
        title="Butterfly Classification",
        cmap='Spectral_r',
        y_labels=CLASS_NAMES,
        size=(13, 10),
        quad_line=False
    )

    plot_confusion_matrix(
        yum_cm,
        title="Yum or Yuck Binary Classification",
        y_labels=['Yuck', 'Yum'],
        x_labels=['No Eat', 'Ate'],
        error_1_text='"Ate yucky butterfly"',
        error_2_text='"Missed yummy butterfly"',
        savepath=os.path.join(PLOT_DIR, f"{_MODEL_NAME}_yum_yuck_confusion_matrix.svg")
    )

    plot_confusion_matrix(
        quad_line_butterfly_cm,
        title="Butterfly Classification",
        cmap='Spectral_r',
        y_labels=CLASS_NAMES,
        size=(13, 10),
        quad_line=True,
        savepath=os.path.join(PLOT_DIR, f"{_MODEL_NAME}_confusion_matrix.svg")
    )


if __name__ == "__main__":
    main()
