import numpy as np
from pathlib import Path
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def get_model(height, width, channel, nclasses):
    # Load VGG16
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, channel))

    # Đóng băng các tầng
    vgg.trainable = False

    # Nhánh label
    output = vgg.output
    flatten1 = Flatten(name='flatten1')(output)
    x_cls = Dense(256, activation='relu')(flatten1)
    x_cls = Dense(256, activation='relu')(x_cls)
    label = Dense(nclasses, activation="softmax", name="class_label")(x_cls)

    # Nhánh bounding box
    output = vgg.output
    flatten2 = Flatten(name='flatten2')(output)
    x_box = Dense(128, activation='relu')(flatten2)
    x_box = Dense(64, activation='relu')(x_box)
    bbox = Dense(4, activation="sigmoid", name="bounding_box")(x_box)

    # Xây dựng model
    model = Model(inputs=vgg.input, outputs=[label, bbox])  # Combine both branches

    # Định nghĩa losses cho các hàm losses
    losses = {
        "class_label": "categorical_crossentropy",
        "bounding_box": "mean_squared_error",
    }

    # Định nghĩa losses cho các hàm lossWeights
    lossWeights = {
        "class_label": 1.0,
        "bounding_box": 1.0
    }

    # Khởi tạo optimizer
    opt = Adam(lr=1e-4)
    model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

    return model

if __name__ == "__main__":
    height, width, channel, nclasses = (224,224,3,3)
    model = get_model(height, width, channel, nclasses)
    print(model.summary())