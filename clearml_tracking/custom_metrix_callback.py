import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import Callback
from clearml import Task

class CustomMetricsCallback(Callback):
    def __init__(self, species_map):
        super(CustomMetricsCallback, self).__init__()
        self.species_map = species_map
        self.clearml_logger = Task.current_task().get_logger()

    def on_epoch_end(self, epoch, logs=None):
        val_data = self.validation_data[0]
        val_labels = np.argmax(self.validation_data[1], axis=1)  # Convert one-hot to integer labels
        predictions = np.argmax(self.model.predict(val_data), axis=1)

        # Confusion matrix
        cm = confusion_matrix(val_labels, predictions)

        # Calculate per-species accuracy
        per_species_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Log per-species accuracy in ClearML
        for idx, species in enumerate(self.species_map.keys()):
            accuracy = per_species_accuracy[idx]
            print(f"Epoch {epoch + 1} - {species} accuracy: {accuracy}")
            self.clearml_logger.report_scalar(title='Per-Species Accuracy', series=species, value=accuracy, iteration=epoch + 1)

        # Log the confusion matrix for the current epoch
        self.clearml_logger.report_confusion_matrix(
            title='Confusion Matrix',
            matrix=cm.tolist(),
            iteration=epoch + 1,
            xlabels=list(self.species_map.keys()),
            ylabels=list(self.species_map.keys())
        )
