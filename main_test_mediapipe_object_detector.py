import argparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def pre_process(image_path):
    image = cv2.imread(image_path)

    kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    kernel_sharpen_4 = cv2.filter2D(image, -1, kernel_sharpen_4)

    return kernel_sharpen_4


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model file to load")
    parser.add_argument("image_path", help="Image file to test")
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                           score_threshold=0.5)
    detector = vision.ObjectDetector.create_from_options(options)

    # cv_mat = cv2.imread(image_path)
    cv_mat = pre_process(image_path)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    detection_result = detector.detect(image)
    print('detections: {}'.format(detection_result))
    image_copy = np.copy(image.numpy_view())
    for detection in detection_result.detections:
        cv2.rectangle(image_copy,
                      (detection.bounding_box.origin_x, detection.bounding_box.origin_y, detection.bounding_box.width,
                       detection.bounding_box.height),
                      (0, 255, 0), 2)
        print('detected categories: ', detection.categories)
        cv2.putText(image_copy, detection.categories[0].category_name,
                    (detection.bounding_box.origin_x, detection.bounding_box.origin_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('detected', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
