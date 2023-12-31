import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

# Ex 09
def read_image(img_path):
    img = cv2.imread(img_path)
    return img

def get_image_dimension(img):
    img_dim = img.shape
    return img_dim

def convert_to_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def resize_img(img, new_size):
    img_resize = cv2.resize(img, new_size, cv2.INTER_LINEAR)
    return img_resize

def show_img(img, window_name='image'):
    """
    cv2.imshow helps to convert image to RGB to show 
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

# Ex 11
def show_video(video_path):
    video_cap = cv2.VideoCapture(video_path)
    if video_cap.isOpened() == False:
        print('Không thể mở video/camera')
    while True:
        ret, frame = video_cap.read()

        # Check if the video has ended
        if frame is None:
            #print("End of video.")
            #break
            # If the video has ended, rewind to the beginning
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("video", frame)

        if not ret:
            print("Không thể nhận frame. Thoát ...")
            break

        if cv2.waitKey(1) == ord('q'):
            break
    
    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # --------------Ex 09----------------
    """
    img_path = './asset/lenna_dump.png'

    # Color image
    img = read_image(img_path)
    show_img(img, 'color image')
    img_color_shape = get_image_dimension(img)
    print('Color Shape = ', img_color_shape)

    # Gray scale image
    img_gray = convert_to_gray_scale(img)
    show_img(img_gray, 'gray scale image')
    img_gray_shape = get_image_dimension(img_gray)
    print('Gray Shape = ', img_gray_shape) 

    # Resize
    new_shape = (200, 100)  # new_shape = (new_width, new_height)
    new_size_img = resize_img(img, new_shape)
    show_img(new_size_img, 'new size img')
    new_size_img_shape = get_image_dimension(new_size_img)
    print('Resize image shape = ', new_size_img_shape)
    """

    # -------------Ex 10-----------------
    #img_path = './asset/lenna_dump.png'
    #img = cv2.imread(img_path)

    #cv2.imshow('original image', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #new_width = 300
    #old_height, old_width, _ = img.shape
    #img_scale = old_width / old_height
    #new_height = int(new_width // img_scale)
    #new_shape = (new_width, new_height)
    #new_img = cv2.resize(img, new_shape)

    #cv2.imshow('resize image', new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # -------------Ex 11-----------------
    video_path = '/Users/giakhang/dev/teaching/huflit/deep_learning/day_01/asset/dump_video.mp4'
    show_video(video_path)