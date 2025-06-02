# ======================
# IMPORTS & DEFINITIONS
# ======================
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import ASTModel, AutoFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib2 import Path
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from clearml import Task
import argparse
from clearml import Task

from datetime import datetime

# ======================
# DATASET CLASS
# ======================
class BirdSoundDataset(Dataset):
    def __init__(self, paths, class_to_idx):
        self.paths = paths
        self.class_to_idx = class_to_idx

    def __getitem__(self, idx):
        data, label = torch.load(self.paths[idx])
        return data, torch.tensor(self.class_to_idx[label])

    def __len__(self):
        return len(self.paths)

# ======================
# HELPER FUNCTIONS
# ======================
def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None, None
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for input_values, labels in loader:
        if input_values is None:
            continue
        input_values, labels = input_values.to(device), labels.to(device)
        outputs = model(input_values)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds

# ======================
# MODEL CLASS
# ======================
class ASTClassifier(nn.Module):
    def __init__(self, num_classes, model_name, cache_dir):
        super().__init__()
        self.base = ASTModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.base.config.hidden_size, num_classes)

    def forward(self, input_values):
        outputs = self.base(input_values=input_values)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(self.dropout(pooled_output))




parser = argparse.ArgumentParser(description="AST Model Trainer")
parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help="Optimizer to use")
args = parser.parse_args()

# ======================
# MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # ClearML initialization

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    #logger = task.get_logger()

    # Configuration ❗ training is based on precomputed_ast dir which needed to be created or downloaded❗
    # metadata file is created based on new dataset from precomputed_ast folder,
    # so that excludes faulty files for which the other script failed to create feature vectors
    # and files which were not selected by k-means algorithms during undersampling step

    base_dir = Path(__file__).resolve().parent
    CSV_PATH = base_dir / "data/metadata/ast_dataset.csv"
    FEATURE_DIR = base_dir / "precomputed_ast"
    CHECKPOINT_PATH = base_dir / f"checkpoints/ast-new{timestamp}.pt"
    # sample rate and length were chosen according to requirements of ast model which is corresponding to audio files of 10s
    # sample rate of 16000 does not cover all frequencies which were mentioned in dataset description
    SAMPLE_RATE = 16000
    MAX_LENGTH = 1024
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CUDA available: {torch.cuda.is_available()}")

    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    CACHE_DIR = base_dir / "hf_cache"
    AutoFeatureExtractor.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))

    # Load labels and compute class weights
    df = pd.read_csv(CSV_PATH)
    classes = sorted(df['species_code'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    labels = [class_to_idx[l] for l in df['species_code']]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Split dataset paths
    def stratified_tensor_split(tensor_dir, class_to_idx, test_size=0.2, random_state=42):
        paths = list(Path(tensor_dir).glob("*.pt"))
        lbls = []
        for path in paths:
            _, lbl = torch.load(path)
            lbls.append(class_to_idx[lbl])
        train_p, test_p = train_test_split(paths, test_size=test_size, stratify=lbls, random_state=random_state)
        return train_p, test_p

    train_paths, test_paths = stratified_tensor_split(FEATURE_DIR, class_to_idx)
    print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

    train_dataset = BirdSoundDataset(train_paths, class_to_idx)
    test_dataset = BirdSoundDataset(test_paths, class_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=safe_collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=safe_collate)

    # Build model
    model = ASTClassifier(len(classes), MODEL_NAME, str(CACHE_DIR)).to(DEVICE)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for input_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            if input_values is None:
                continue
            input_values, labels = input_values.to(DEVICE), labels.to(DEVICE)
            outputs = model(input_values)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        y_true, y_pred = evaluate(model, test_loader, DEVICE)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Final evaluation & reporting
    y_true, y_pred = evaluate(model, test_loader, DEVICE)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # Confusion matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (AST)")
    plt.tight_layout()
    
    cm_path = base_dir / f"confusion_matrix_ast_{timestamp}.png"
    plt.savefig(cm_path)