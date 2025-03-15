import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

from model import UNet

def load_data(imagePath, labelPath):
    # Load images and labels
    images = np.load(imagePath)
    labels = np.load(labelPath)

    # Chuyen sang shape: 160 x 320 x 3
    new_images, new_labels = [], []
    for i in range(len(images)):
        cropped_image = images[i]
        cropped_image = cropped_image[10:170, :, :]

        cropped_label = labels[i]
        cropped_label = cropped_label[10:170, :]

        new_images.append(cropped_image)
        new_labels.append(cropped_label)

    images = np.array(new_images, dtype='uint8')
    labels = np.array(new_labels, dtype='uint8')

    return images, labels

def preprocessing(images, labels, n_classes):
    # 1. Mở rộng thêm chiều của nhãn
    labels = np.expand_dims(labels, axis=-1)
    print(labels.shape)

    # 2. Thay đổi kiểu sang float32 và Chuẩn hoá dữ liệu về miền [0, 1]
    images = images.astype(np.float32)
    images /= 255

    # 3. Train Test Split
    X_train, X_val, Y_train, Y_val = train_test_split(images, labels,
                                                      test_size=0.2, 
                                                      random_state=42)
    
    # One hot encoded output mask 
    Y_train = np.eye(n_classes)[Y_train]
    Y_val = np.eye(n_classes)[Y_val]

    Y_train = Y_train.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))
    Y_val = Y_val.reshape((Y_val.shape[0], Y_val.shape[1], Y_val.shape[2], n_classes))

    print('Training data shape is: {}'.format(Y_train.shape))
    print("Validating data shape is: {}".format(Y_val.shape))

    return X_train, Y_train, X_val, Y_val

def create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train).permute(0, 3, 1, 2), torch.tensor(Y_train).permute(0, 3, 1, 2))
    val_dataset = TensorDataset(torch.tensor(X_val).permute(0, 3, 1, 2), torch.tensor(Y_val).permute(0, 3, 1, 2))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model

if __name__ == '__main__':
    # Load data
    imagePath = 'data/image_180_320.npy'
    labelPath = 'data/label_180_320.npy'

    images, labels = load_data(imagePath, labelPath)

    X_train, Y_train, X_val, Y_val = preprocessing(images, labels, n_classes=3)

    # Tinh chỉnh siêu tham số
    batch_size = 16
    epochs = 50
    learning_rate = 0.001

    # Load model
    model = UNet(in_channels=3, out_channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = create_dataloaders(X_train, Y_train, X_val, Y_val, batch_size)

    device = "cuda"  # "cuda", "cpu" or "mps"
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)