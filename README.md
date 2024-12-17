# python-LicensePlateDetector
This project aim to train and provide machine learning license plate detector models

`main_train_mediapipe_object_detector.py` script used to train object detector using mediapipe.

`main_test_mediapipe_object_detector.py` script used to test object detector using mediapipe, also include pre-processing logic.

# Pre-processing (MediaPipe)
Current pre-processing logic only sharpen image before model inference which is enough to raise inference accuracy
```
def pre_process(image_path):
    image = cv2.imread(image_path)

    kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 8, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 8.0
    kernel_sharpen_4 = cv2.filter2D(image, -1, kernel_sharpen_4)

    return kernel_sharpen_4
```

# Models
1. `output/model/mediapipe/id_detector.tflite` this model was trained based on [indonesian license plate dataset](https://www.kaggle.com/datasets/imamdigmi/indonesian-plate-number) with epoch `100`, batch-size `8`, learning-rate `0.3` 

## Support this project
Consider donation to support this project
<table>
  <tr>
    <td><a href="https://trakteer.id/rh-id">https://trakteer.id/rh-id</a></td>
  </tr>
</table>
