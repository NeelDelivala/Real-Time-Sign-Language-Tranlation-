import numpy as np
import cv2
import matplotlib.pyplot as plt

def handsegment(frame):
    if frame is None:
        print("Error: Could not load the image. Check the file path.")
        return None

    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply mask to original image
    output = cv2.bitwise_and(frame, frame, mask=mask)

    return output

if __name__ == '__main__':
    frame = cv2.imread("image.png")
    segmented_hand = handsegment(frame)

    if segmented_hand is not None:
        plt.imshow(cv2.cvtColor(segmented_hand, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Improved Segmented Hand")
        plt.show()

        cv2.imwrite("improved_segmented_hand.jpg", segmented_hand)
        print("Segmented image saved as 'improved_segmented_hand.jpg'")
