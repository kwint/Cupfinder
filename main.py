import cv2
import numpy as np


def nothing(x):
    pass


def init():
    # Create windows and sliders
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

    cv2.namedWindow('slider', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('slider', 640, 0)
    cv2.resizeWindow('slider', 560, 400)
    cv2.createTrackbar('B', 'slider', 0, 255, nothing)
    cv2.createTrackbar('G', 'slider', 0, 255, nothing)
    cv2.createTrackbar('R', 'slider', 0, 255, nothing)

    cv2.createTrackbar('B1', 'slider', 0, 255, nothing)
    cv2.createTrackbar('G1', 'slider', 0, 255, nothing)
    cv2.createTrackbar('R1', 'slider', 0, 255, nothing)
    cv2.createTrackbar('color', 'slider', 0, 5, nothing)
    cv2.createTrackbar('image', 'slider', 1, 3, nothing)
    cv2.setTrackbarMin('image', 'slider', 1)


def filter_image(img, lower_mask, upper_mask):
    # set sliders to start values
    kernel = np.ones((10, 10), np.uint8)
    cv2.setTrackbarPos('B', 'slider', lower_mask[0])
    cv2.setTrackbarPos('G', 'slider', lower_mask[1])
    cv2.setTrackbarPos('R', 'slider', lower_mask[2])
    cv2.setTrackbarPos('B1', 'slider', upper_mask[0])
    cv2.setTrackbarPos('G1', 'slider', upper_mask[1])
    cv2.setTrackbarPos('R1', 'slider', upper_mask[2])

    # wait a bit to update
    cv2.waitKey(5)

    # Read slider positions
    b = cv2.getTrackbarPos('B', 'slider')
    g = cv2.getTrackbarPos('G', 'slider')
    r = cv2.getTrackbarPos('R', 'slider')
    b1 = cv2.getTrackbarPos('B1', 'slider')
    g1 = cv2.getTrackbarPos('G1', 'slider')
    r1 = cv2.getTrackbarPos('R1', 'slider')

    # Build mask array from sliders
    lower_unit = np.array([b, g, r])
    upper_unit = np.array([b1, g1, r1])

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Filter colors
    mask = cv2.inRange(hsv, lower_unit, upper_unit)
    res = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Make binary image
    ret, thres = cv2.threshold(gray, 20, 255, 0)

    # Close some holes
    thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)

    # Return binary image and slider data, so program remebers their position
    return thres, b, g, r, b1, g1, r1


# Declaration of variables
# Arrays that store values for color filtering
#   None        Green           Blue            Pink              Orange          Yellow
lower_mask = np.array(
    [[0, 0, 0], [59, 95, 101], [0, 99, 0], [119, 90, 107], [107, 143, 141], [83, 131, 131]])
upper_mask = np.array(
    [[0, 0, 0], [79, 255, 255], [28, 255, 222], [149, 225, 218], [120, 255, 230], [94, 255, 233]])

# Build windows and trackbars
init()

# Mainloop
while True:
    # Get input form user, so we know in what image we need to look and which color
    color_code = cv2.getTrackbarPos("color", "slider")
    image_number = cv2.getTrackbarPos("image", "slider")

    # Read image
    img = cv2.imread("cup" + str(image_number) + ".jpg")

    # Filter image. Returns image as thres and returns lower_mask and upper_mask
    # because it could be changed by the user
    thres, b, g, r, b1, g1, r1 = filter_image(img, lower_mask[color_code], upper_mask[color_code])
    lower_mask[color_code] = [b, g, r]
    upper_mask[color_code] = [b1, g1, r1]

    # Detect contours
    im2, contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Reset cupcounter
    cupcounter = 0

    # Filter countours with a lower Area then 12000. Then draw a rectangle around remaining contours
    for cnt in range(0, len(contours)):
        if cv2.contourArea(contours[cnt]) > 12000:
            rect = cv2.minAreaRect(contours[cnt])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cupcounter += 1

    # Print the number of cups on te image
    img = cv2.putText(img, str(cupcounter), (80, 80), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0))
    cv2.imshow("Image", img)

    # Pressing esc exits the program 27 for windows, 1048603 for linux
    if cv2.waitKey(1) in {1048603, 27}:
        cv2.destroyAllWindows()
        break
