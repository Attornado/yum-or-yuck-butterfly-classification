import os
from typing import final
import pandas as pd
import tensorflow as tf
from model_evaluation.utils import decode_predictions
from training.constants import FITTED_YOYNET_DIR
from preprocessing.constants import TEST_PATH_CLEANED, YUMS, CLASS_NAMES
from preprocessing.utils import decode_image, decode_image_id
from predictions.constants import PREDICTIONS_DIR, PREDICTIONS_BATCH_SIZE


_VERSION_LOAD: final = 3.2
_EPOCHS_LOAD: final = 300
_MODEL_NAME: final = f"yoynet_{_EPOCHS_LOAD}_epochs_v{_VERSION_LOAD}"
_MODEL_PATH: final = os.path.join(FITTED_YOYNET_DIR, _MODEL_NAME)
_SUBMIT_PATH: final = os.path.join(PREDICTIONS_DIR, f"{_MODEL_NAME}_predictions.csv")


def main():
    # Load the dataset and the model
    test_ds_prebatch = tf.data.experimental.load(TEST_PATH_CLEANED)
    test_ds = test_ds_prebatch.batch(PREDICTIONS_BATCH_SIZE)
    model = tf.keras.models.load_model(_MODEL_PATH)

    # Make predictions and format them into a prediction data frame
    preds = model.predict(test_ds, verbose=1)
    predicted_label_confidence = decode_predictions(preds, CLASS_NAMES)

    # Combine the actual and predicted data together into a single list
    test_ds_decoded = [(decode_image(m), decode_image_id(im_id)) for (m, im_id) in test_ds.unbatch()]

    # Zip the tuples manually
    predictions_df = pd.DataFrame(
        [(test_ds_decoded[i][1],  # id
          test_ds_decoded[i][0],  # X
          predicted_label_confidence[i][0][0],  # y_pred
          predicted_label_confidence[i][0][0] in YUMS,  # yum_pred
          predicted_label_confidence[i][0][1],  # y_conf
          ) for i in range(len(test_ds_decoded))],
        columns=['id', 'X', 'y_pred', 'yum_pred', 'y_conf'])

    # Print head to see dataframe structure
    print(predictions_df.head())

    # Change the dataframe to match the expected submission CSV
    submit_df = predictions_df[['id', 'y_pred']]
    submit_df.columns = ['image', 'name']

    # Save submit csv
    submit_df.to_csv(_SUBMIT_PATH, header=True, index=False)
    print(f'Submit csv file "{_SUBMIT_PATH}" saved')
    print(submit_df)


if __name__ == "__main__":
    main()
