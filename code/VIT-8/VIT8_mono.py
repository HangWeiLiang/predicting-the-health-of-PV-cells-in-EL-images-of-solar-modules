# Import the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize
from einops.layers.torch import Rearrange
from einops import rearrange
from tqdm import tqdm
from sklearn.metrics import classification_report
from data_pre import poly_X_train, poly_y_train, mono_X_test, mono_y_test
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

X_train_tensor = torch.tensor(poly_X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(poly_y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(mono_X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(mono_y_test, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class ImageTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, channels=1, dropout_rate=0.1, num_layers=12, num_heads=12):
        super(ImageTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        assert patch_dim % num_heads == 0, "patch_dim must be divisible by num_heads"

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, patch_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=patch_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(patch_dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x + self.pos_embedding[:, :(x.size(1))])
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

model = ImageTransformer(
    image_size=64,
    patch_size=8,
    num_classes=4,
    channels=1,
    dropout_rate=0.2,
    num_layers=6,
    num_heads=8
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_loop = tqdm(train_loader, position=0, leave=True)
    for data, target in train_loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        train_loop.set_description(f'Epoch {epoch+1}/{num_epochs}')
        train_loop.set_postfix(loss=loss.item())

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct / total)

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    valid_losses.append(running_loss / len(test_loader))
    valid_accuracies.append(correct / total)

    scheduler.step()

classification_rep = classification_report(all_targets, all_predictions, zero_division=1)
print("Classification Report:")
print(classification_rep)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(valid_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

torch.save(model.state_dict(), 'transformer_model.pth')

from sklearn.metrics import confusion_matrix, f1_score

conf_matrix = confusion_matrix(y_test_tensor.cpu().numpy(), all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

f1 = f1_score(y_test_tensor.cpu().numpy(), all_predictions, average='macro')
print("F1 Score (Macro):")
print(f1)
