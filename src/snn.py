import torch
import torch.nn as nn
import snntorch as snn
import torch.optim as optim
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

X_train = np.load('../data/processed/npy/X_train.npy')
X_test = np.load('../data/processed/npy/X_test.npy')
X_val = np.load('../data/processed/npy/X_val.npy')
y_train = np.load('../data/processed/npy/y_train.npy')
y_test = np.load('../data/processed/npy/y_test.npy')
y_val = np.load('../data/processed/npy/y_val.npy')

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

class SNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(beta=0.9)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1, mem1 = self.lif1(self.fc1(x), mem1)
        spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
        out = self.fc3(spk2)
        return out

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = SNN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
train_acc, val_acc = [], []
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        train_preds = torch.argmax(model(X_train), dim=1)
        val_preds = torch.argmax(model(X_val), dim=1)
        train_accuracy = accuracy_score(y_train.argmax(dim=1), train_preds)
        val_accuracy = accuracy_score(y_val.argmax(dim=1), val_preds)

    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

end_time = time.time()
with torch.no_grad():
    test_preds = torch.argmax(model(X_test), dim=1)
    y_test_labels = y_test.argmax(dim=1)
    test_accuracy = accuracy_score(y_test_labels, test_preds)
    precision = precision_score(y_test_labels, test_preds, average='weighted')
    recall = recall_score(y_test_labels, test_preds, average='weighted')
    f1 = f1_score(y_test_labels, test_preds, average='weighted')
    cm = confusion_matrix(y_test_labels, test_preds)
    class_report = classification_report(y_test_labels, test_preds)

print(f'Final Test Accuracy: {test_accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", class_report)
print(f'Run Time: {end_time - start_time:.2f} seconds')

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), train_acc, label='Train Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()