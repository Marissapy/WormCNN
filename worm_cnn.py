import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Custom Dataset class for Worm images
class WormDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load images and assign labels based on directory names (age group)
        for day_dir in os.listdir(root_dir):
            day = int(day_dir.replace('day', ''))
            label = 0 if day <= 29 else 1  # Age groups: Days 8-29 (young), 30-54 (old)
            full_dir = os.path.join(root_dir, day_dir)
            for img_file in os.listdir(full_dir):
                img_path = os.path.join(full_dir, img_file)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Load image as grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Neural network class defining WormCNN architecture
class WormCNN(nn.Module):
    def __init__(self):
        super(WormCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.residual_block1 = self.make_layer(64, 128)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.residual_block1(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Transformations applied to images
transform = transforms.Compose([
    transforms.Resize((600, 30)),  # Resize images
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
])

# Data loaders for training and testing datasets
train_dataset = WormDataset(root_dir='train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

test_dataset = WormDataset(root_dir='test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# Setting up the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WormCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=10e-6)

# Training loop
for epoch in range(256):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.round()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/256], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    # Evaluation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            predicted = outputs.round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
