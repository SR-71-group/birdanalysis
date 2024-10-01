# utils.py
def print_results(history):
    """Print training results (loss and accuracy)."""
    print("Training Loss: ", history.history['loss'])
    print("Validation Loss: ", history.history['val_loss'])
    print("Training Accuracy: ", history.history['accuracy'])
    print("Validation Accuracy: ", history.history['val_accuracy'])
