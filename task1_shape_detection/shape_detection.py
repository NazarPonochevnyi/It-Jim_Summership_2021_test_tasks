"""
Software that detects each of the yellow shapes on the video frames and
classifies the shapes into classes: circle, rectangle, triangle.

USAGE: python3 shape_detection.py <video path> <output video path>

"""

import sys
import cv2
import imutils
import numpy as np
from tqdm import tqdm


BOX_COLORS = {
    "triangle": (255, 0, 0),
    "rectangle": (0, 255, 0),
    "circle" : (0, 0, 255)
}


def get_contours(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Gets edge and yellow contours from an image.
    Parameters
    ----------
    image: np.ndarray
        Target image.
    Returns
    -------
    edges_filled: np.ndarray
        Detected edges in an image as a boolean 2D map.
    yellow_contours: np.ndarray
        Detected yellow contours in an image.

    """

    # get edges
    edges = cv2.Canny(image, 10, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges_filled = np.zeros_like(edges_thresh)
    edges_contours = cv2.findContours(edges_thresh.copy(), cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    edges_contours = imutils.grab_contours(edges_contours)
    for cont in edges_contours:
        cv2.drawContours(edges_filled, [cont], 0, 255, -1)

    # select yellow color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_lower = np.array([30, 10, 10])
    yellow_upper = np.array([90, 255, 255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_output = cv2.bitwise_and(image, image, mask=mask_yellow)
    gray = cv2.cvtColor(yellow_output, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    yellow_contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = imutils.grab_contours(yellow_contours)
    return edges_filled, yellow_contours


def detect(tcontour: np.ndarray) -> str:
    """
    Detects shape by a contour.
    Parameters
    ----------
    tcontour: np.ndarray
        Target contour.
    Returns
    -------
    shape: str
        Detected shape of a contour.

    """
    shape = "unidentified"
    peri = cv2.arcLength(tcontour, True)
    approx = cv2.approxPolyDP(tcontour, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        shape = "rectangle"
    else:
        shape = "circle"
    return shape


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'USAGE: python3 {sys.argv[0]} <video path> <output video path>')
        sys.exit()
    VIDEO_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

    # open input and output videos
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    frame_count = int(cap.get(7))
    out = cv2.VideoWriter(OUTPUT_PATH, -1, fps, (width, height))
    if not cap.isOpened() or not out.isOpened():
        raise RuntimeError("video file cannot be opened")
    print(f'\nVideo "{VIDEO_PATH}" opened for processing\n')

    for i in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if ret is True:

            # preprocess frame
            resized = imutils.resize(frame, width=300)
            ratio = frame.shape[0] / float(resized.shape[0])
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            edges_map, contours = get_contours(blurred)
            for contour in contours:
                if cv2.contourArea(contour) > 50:

                    # get contour's center
                    M = cv2.moments(contour)
                    rel_cX = int(M["m10"] / M["m00"])
                    rel_cY = int(M["m01"] / M["m00"])
                    if edges_map[rel_cY, rel_cX]:
                        SHAPE = detect(contour)

                        # get exact coordinates
                        contour = contour.astype("float")
                        contour *= ratio
                        cX = int((M["m10"] / M["m00"]) * ratio)
                        cY = int((M["m01"] / M["m00"]) * ratio)
                        contour = contour.astype("int")

                        # draw contour and shape name
                        cv2.drawContours(frame, [contour], -1, BOX_COLORS[SHAPE], 2)
                        cv2.putText(frame, SHAPE, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    print(f'\nVideo "{OUTPUT_PATH}" successfully saved')
