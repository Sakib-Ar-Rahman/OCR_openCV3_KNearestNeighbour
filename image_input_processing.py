import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

# ---------------------------- TAKING PICTURES ------------------------------

def take_camera_picture():
    # Camera 0 is the integrated web cam on my mac
    camera_port = 0
     
     
    # Now we can initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port.
    cam = cv2.VideoCapture(camera_port)
    cv2.namedWindow("OCR Image Capture")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("OCR Image Capture", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame.png".format(img_counter)
            # Change to Greyscale
            img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_grey, (5,5), 0)
            (thresh, img_bw) = cv2.threshold(img_blur, 70, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(img_name, img_bw)
            print("{} written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()


# ---------------------------- SCALING PICTURES ------------------------------

def scalePicture():
    print ("Opening Preview")
    preview_img = cv2.imread("opencv_frame.png")
    cv2.imshow("Preview", preview_img)

    # Pause until escape key is hit
    k = cv2.waitKey(0)
    if k%256 == 27:
    	cv2.destroyAllWindows()

    # Reshaping image
    # r = 100.0 / preview_img.shape[1]
    # dim = (100, int(preview_img.shape[0] * r))
    dim = (20, 20)

    # perform the actual resizing of the image and show it
    resized = cv2.resize(preview_img, dim, interpolation = cv2.INTER_AREA)
    print("showing resized image")
    cv2.imshow("resized", resized)
    cv2.imwrite("resized_image.png", resized)
    cv2.waitKey(0)


def main():
    take_camera_picture()
    scalePicture()

if __name__ == "__main__":
    main()