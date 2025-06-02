from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from clearml import Task, Logger

class CustomMetricsCallback(Callback):
    def __init__(self, species_map, validation_data=None):
        super(CustomMetricsCallback, self).__init__()
        self.species_map = species_map
        self.validation_data = validation_data  # Store validation data for metrics
        self.logger = Task.current_task().get_logger()

    def on_epoch_end(self, epoch, logs=None):
        # Log accuracy and loss for training and validation
        self.logger.report_scalar("Accuracy", "Train", iteration=epoch, value=logs["accuracy"])
        self.logger.report_scalar("Loss", "Train", iteration=epoch, value=logs["loss"])
        self.logger.report_scalar("Accuracy", "Validation", iteration=epoch, value=logs["val_accuracy"])
        self.logger.report_scalar("Loss", "Validation", iteration=epoch, value=logs["val_loss"])

        # Log confusion matrix and per-species accuracy for validation
        if self.validation_data:
            val_data, val_labels = self.validation_data
            val_labels = np.argmax(val_labels, axis=1)  # Convert one-hot labels to integers
            predictions = np.argmax(self.model.predict(val_data), axis=1)

            cm = confusion_matrix(val_labels, predictions)
            per_species_accuracy = cm.diagonal() / cm.sum(axis=1)

            # Log per-species accuracy
            for idx, species in enumerate(self.species_map.keys()):
                self.logger.report_scalar(
                    title="Per-Species Accuracy",
                    series=species,
                    value=per_species_accuracy[idx],
                    iteration=epoch
                )

            # Log confusion matrix
            self.logger.report_confusion_matrix(
                title="Confusion Matrix",
                series="Validation",
                iteration=epoch,
                matrix=cm.tolist(),
                xlabels=list(self.species_map.keys()),
                ylabels=list(self.species_map.keys())
            )
