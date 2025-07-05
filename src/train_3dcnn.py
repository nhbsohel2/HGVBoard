
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class GestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        for label, gesture in enumerate(self.classes):
            gesture_path = os.path.join(root_dir, gesture)
            for file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, file)
                self.samples.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.imread(self.samples[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        if self.transform:
            img = self.transform(img)
        return torch.tensor(img), self.labels[idx]

class SimpleGestureCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleGestureCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.cnn(x)

dataset = GestureDataset('./dataset/demo_gestures')
loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGestureCNN(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), './model/gesture_model.pth')
