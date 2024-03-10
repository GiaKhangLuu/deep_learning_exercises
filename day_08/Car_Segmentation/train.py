import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from model import U_NET

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
    # Ví dụ: (3430, 160, 320) --> (3430, 160, 320, 1)
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
    train_masks = to_categorical(Y_train, num_classes=n_classes)
    Y_train = train_masks.reshape((Y_train.shape[0], Y_train.shape[1],
                                   Y_train.shape[2], 3))
    print('Training data shape is: {}'.format(Y_train.shape))

    val_masks = to_categorical(Y_val, num_classes=n_classes)
    Y_val = val_masks.reshape((Y_val.shape[0], Y_val.shape[1], 
                               Y_val.shape[2], 3))
    print("Validaing data shape is: {}".format(Y_val.shape))

    return X_train, Y_train, X_val, Y_val

# Chuyển dữ liệu về format tf.data
def tf_data(X_train, Y_train, X_val, Y_val, batch_size):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, 
                                                   Y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val,
                                                 Y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

if __name__ == '__main__':
    # Load data
    imagePath = 'data/image_180_320.npy'
    labelPath = 'data/label_180_320.npy'

    images, labels = load_data(imagePath, labelPath)

    X_train, Y_train, X_val, Y_val = preprocessing(images, labels, n_classes=3)

    # Tinh chỉnh siêu tham số
    batch_size = 16
    epochs = 50

    # Load model
    model = U_NET(width=320, height=160, channels=3, n_classes=3)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Callback
    es = EarlyStopping(monitor='val_loss', patience=5,
                       restore_best_weights=True)
    
    train_ds, val_ds = tf_data(X_train, Y_train, X_val, Y_val, batch_size)

    H = model.fit(train_ds, callbacks=[es],
                  epochs=epochs, verbose=1,
                  validation_data=val_ds)
    
    # Save model
    path = 'best_model.h5'
    model.save(path)

    # In đường cong học tập
    print(H.history.keys())
    plt.figure()
    plt.plot(H.history['accuracy'], label="train_acc")
    plt.plot(H.history['val_accuracy'], label="val_acc")
    plt.plot(H.history['loss'], label = "train_loss")
    plt.plot(H.history['val_loss'], label = "val_loss")
    plt.title('Training Loss and Accuracy')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch #')
    plt.legend()
    plt.show()