import data
import model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Kích thước dữ liệu
    height, width, channel, nclasses = (224,224,3,3)

    # Load data
    images, labels, bboxes = data.get_data(224, 224)

    train_set, val_set = data.preprocess_input(images, labels, bboxes)
    images_train, labels_train, bboxes_train = train_set
    images_valid, labels_valid, bboxes_valid = val_set

    # Load model
    model = model.get_model(height, width, channel, nclasses)

    images_train = np.stack(images_train)
    labels_train = np.stack(labels_train)
    bboxes_train = np.stack(bboxes_train)

    images_valid = np.stack(images_valid)
    labels_valid = np.stack(labels_valid)
    bboxes_valid = np.stack(bboxes_valid)

    # Huấn luyện model
    H = model.fit(x=images_train, y=[labels_train, bboxes_train],
                  validation_data=(images_valid,[labels_valid,bboxes_valid]), 
                  batch_size=32, epochs=10, verbose=1)
    # Lưu model
    model.save("model.keras")

    print(H.history.keys())

    # Vẽ biểu đồ
    plt.figure()
    plt.plot(H.history['class_label_loss'], label="class_loss")
    plt.plot(H.history['bounding_box_loss'], label="box_loss")
    plt.plot(H.history['class_label_accuracy'], label="class_acc")
    plt.plot(H.history['bounding_box_accuracy'], label="box_acc")

    plt.plot(H.history['val_class_label_loss'], label="val_class_loss")
    plt.plot(H.history['val_bounding_box_loss'], label="val_box_loss")
    plt.plot(H.history['val_class_label_accuracy'], label="val_class_acc")
    plt.plot(H.history['val_bounding_box_accuracy'], label="val_box_acc")

    plt.title('Training Loss and Accuracy')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epoch #')
    plt.legend()

    plt.savefig('./plot_learning_curve.jpg')

    plt.close()