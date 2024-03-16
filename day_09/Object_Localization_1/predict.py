import numpy as np
from tensorflow.keras.saving import load_model
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import img_to_array, load_img
import data

def predict():
    # load model
    model = load_model('./model.keras')

    # load ảnh
    image_path = './dataset/dataset/airplane/image_0004.jpg'
    # Load lại ảnh với hàm load_img với kích thước height, width
    target_size = (224, 224)
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image /= 255.
    image = np.expand_dims(image, 0)

    # predict
    (labelPreds, boxPreds) = model(image)
    box_preds = boxPreds.numpy()[0]
    label_preds = labelPreds.numpy()[0]

    (x1, y1, x2, y2) = box_preds

    # show ảnh và kết quả dự đoán
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape

    x1 = int(x1 * img_w)
    y1 = int(y1 * img_h)
    x2 = int(x2 * img_w)
    y2 = int(y2 * img_h)

    # Vẽ box
    image = cv2.rectangle(image, (x1, y1), 
                          (x2, y2), (0, 255, 0), 2)

    # Ghi lớp dự đoán
    class_names = ['airplane', 'face', 'motorcycle']
    predicted_class = class_names[np.argmax(label_preds)]
    org = (x1, y1 + 10)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    lineType = cv2.LINE_AA
    image = cv2.putText(image, 
                      predicted_class, 
                      org, 
                      fontFace, 
                      fontScale, 
                      color, 
                      thickness, 
                      lineType)
    
    plt.imshow(image)

    plt.savefig('result.jpg')
 



if __name__ == "__main__":
    predict()