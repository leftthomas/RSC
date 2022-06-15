import cv2
import numpy as np
from PIL import Image


def show_flow_hsv(flow, show_style=1):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)
    if show_style == 1:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    elif show_style == 2:
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


flow_x = np.array(Image.open('result/flow_x/651.jpg'))
flow_y = np.array(Image.open('result/flow_y/651.jpg'))
flow = ((np.asarray(np.stack([flow_x, flow_y], axis=-1), dtype=np.float32)) - 128.0) / (256.0 / 40)
flow = show_flow_hsv(flow)
cv2.imwrite('result/651.jpg', flow)
