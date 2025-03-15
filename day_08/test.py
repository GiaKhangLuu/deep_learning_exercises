import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from train import preprocessing, load_data
from model import UNet

if __name__ == '__main__':
    # Load model từ file
    model_file_path = 'best_model.pth'
    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

    # Load data
    imagePath = 'data/image_180_320.npy'
    labelPath = 'data/label_180_320.npy'

    images, labels = load_data(imagePath, labelPath)

    X_train, Y_train, X_val, Y_val = preprocessing(images, labels, n_classes=3)

    # Lấy ảnh từ index
    index = 100
    image = X_val[index]
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    # Dự đoán nhãn
    with torch.no_grad():
        pred = model(image_tensor)
        pred = pred.squeeze(0).permute(1, 2, 0).numpy()

    # Chuyển dự đoán về mask
    train_id_to_color = np.array([(0, 255, 0),
                                  (255, 255, 0),
                                  (0, 0, 0)])

    mask_pred = np.argmax(pred, axis=2)
    mask_pred = train_id_to_color[mask_pred]

    mask_pred = np.array(mask_pred, dtype='uint8')
    combo_image = cv2.addWeighted(image, 0.8, mask_pred, 1, 1)
    
    _ = plt.figure()
    plt.imshow(combo_image)
    plt.savefig('output.png')
    plt.close()