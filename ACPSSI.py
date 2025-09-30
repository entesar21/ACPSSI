import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, recall_score, precision_score
)
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training data
sequence_data_train = np.load('T5XL_alternate_train.npy')
sequence_labels_train = pd.read_csv('dataset/deepgram/internal_alternate.csv')['label']
smiles_data_train = pd.read_csv('internal_alternate_smiles.csv')

# Load testing data
sequence_data_test = np.load('T5XL_alternate_test.npy')
sequence_labels_test = pd.read_csv('dataset/deepgram/validation_alternate.csv')['label']
smiles_data_test = pd.read_csv('validation_alternate_smiles.csv')

# Check for matching lengths
assert sequence_data_train.shape[0] == len(sequence_labels_train) == len(smiles_data_train), "Train data mismatch"
assert sequence_data_test.shape[0] == len(sequence_labels_test) == len(smiles_data_test), "Test data mismatch"

# ChemBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chemberta = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
for param in chemberta.parameters():
    param.requires_grad = False
for name, param in chemberta.named_parameters():
    if 'encoder.layer.10' in name or 'encoder.layer.11' in name:
        param.requires_grad = True

# Dataset class
class ChemDataset(Dataset):
    def __init__(self, smiles, sequences, labels):
        self.smiles = smiles
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoded = tokenizer(self.smiles[idx], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded, sequence, label

# Model class
class MultiModalClassifier(nn.Module):
    def __init__(self, seq_input_dim, smiles_hidden_dim=768, lstm_hidden_dim=512, lstm_layers=2, dropout=0.3, num_classes=2):
        super(MultiModalClassifier, self).__init__()
        self.bilstm = nn.LSTM(seq_input_dim, lstm_hidden_dim, num_layers=lstm_layers,
                              bidirectional=True, dropout=dropout, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim * 2, nhead=8, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.smiles_fc = nn.Linear(smiles_hidden_dim, lstm_hidden_dim * 2)
        self.fc1 = nn.Linear(lstm_hidden_dim * 4, lstm_hidden_dim)
        self.fc2 = nn.Linear(lstm_hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, seq_input, smiles_input):
        x, _ = self.bilstm(seq_input)
        x = self.transformer(x)
        x_seq = x[:, -1, :]
        x_smiles = self.smiles_fc(smiles_input)
        x = torch.cat((x_seq, x_smiles), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for data, seqs, labels in loader:
        optimizer.zero_grad()
        smiles_inputs = {key: val.squeeze(1).to(device) for key, val in data.items()}
        smiles_features = chemberta(**smiles_inputs).pooler_output
        outputs = model(seqs.to(device), smiles_features)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, seqs, labels in loader:
            smiles_inputs = {key: val.squeeze(1).to(device) for key, val in data.items()}
            smiles_features = chemberta(**smiles_inputs).pooler_output
            outputs = model(seqs.to(device), smiles_features)
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return total_loss / len(loader), acc, mcc, specificity, sensitivity

# Hyperparameters to test
epochs_list = [10]
dropouts = [0.2]
learning_rates = [1e-5]

# Input dim
input_dim = sequence_data_train.shape[2]
batch_size = 32
num_classes = 2

# Store results
results = []

# Hyperparameter tuning loop
for epochs in epochs_list:
    for dropout in dropouts:
        for lr in learning_rates:
            print(f"\nTesting config: Epochs={epochs}, Dropout={dropout}, LR={lr}")
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_val_acc = 0
            best_model_state = None

            for fold, (train_idx, val_idx) in enumerate(kf.split(sequence_data_train, sequence_labels_train)):
                train_dataset = ChemDataset(
                    smiles_data_train.iloc[train_idx]['smiles'].tolist(),
                    sequence_data_train[train_idx],
                    sequence_labels_train.iloc[train_idx].tolist())
                val_dataset = ChemDataset(
                    smiles_data_train.iloc[val_idx]['smiles'].tolist(),
                    sequence_data_train[val_idx],
                    sequence_labels_train.iloc[val_idx].tolist())

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                model = MultiModalClassifier(input_dim, dropout=dropout).to(device)
                optimizer = torch.optim.AdamW(list(model.parameters()) + list(filter(lambda p: p.requires_grad, chemberta.parameters())), lr=lr)
                criterion = nn.CrossEntropyLoss()

                for epoch in range(epochs):
                    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                    print(f"  Fold {fold+1} | Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f}")

                val_loss, val_acc, val_mcc, val_spec, val_sens = evaluate(model, val_loader, criterion)
                print(f"  Fold {fold+1} | Val Acc: {val_acc:.4f} | MCC: {val_mcc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()

            # Test set evaluation
            model.load_state_dict(best_model_state)
            test_dataset = ChemDataset(smiles_data_test['smiles'].tolist(), sequence_data_test, sequence_labels_test.tolist())
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            test_loss, test_acc, test_mcc, test_spec, test_sens = evaluate(model, test_loader, criterion)
            print(f"Test Results for this config — Acc: {test_acc:.4f}, MCC: {test_mcc:.4f}")

            results.append({
                "epochs": epochs,
                "dropout": dropout,
                "lr": lr,
                "test_acc": test_acc,
                "test_mcc": test_mcc,
                "test_spec": test_spec,
                "test_sens": test_sens
            })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("\nAll results saved to hyperparameter_tuning_results.csv")

# Summary of best config
best_row = results_df.loc[results_df['test_acc'].idxmax()]
print("\nBest Configuration:")
print(best_row)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(len(results_df)), results_df['test_acc'], tick_label=[f"{r['epochs']}-{r['dropout']}-{r['lr']}" for _, r in results_df.iterrows()])
plt.xticks(rotation=45)
plt.title("Test Accuracy")
plt.subplot(1, 2, 2)
plt.bar(range(len(results_df)), results_df['test_mcc'], tick_label=[f"{r['epochs']}-{r['dropout']}-{r['lr']}" for _, r in results_df.iterrows()])
plt.xticks(rotation=45)
plt.title("Test MCC")
plt.tight_layout()
plt.savefig("hyperparameter_tuning_charts.png")
plt.show()
