import argparse
import os
import pathlib

import tensorflow as tf

assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="Directory to train image")
    parser.add_argument("test_dir", help="Directory to test image")
    parser.add_argument("dataset_type", choices=['coco', 'pascal_voc']
                        , help="Dataset type? (COCO or Pascal VOC)")
    parser.add_argument("--train-learning-rate", type=float, default=0.3,
                        help="Learning rate for the model")
    parser.add_argument("--train-batch-size", type=int, default=8,
                        help="Learning rate for the model")
    parser.add_argument("--train-epochs", type=int, default=30,
                        help="Learning rate for the model")
    parser.add_argument("--out-dir", default=os.path.join("output", "model", "mediapipe"),
                        help="Output directory for the model")
    parser.add_argument("--out-model-name", default="detector.tflite",
                        help="Output directory for the model")
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.out_dir

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset_type is 'coco':
        train_data = object_detector.Dataset.from_coco_folder(train_dir)
        test_data = object_detector.Dataset.from_coco_folder(test_dir)
    else:
        train_data = object_detector.Dataset.from_pascal_voc_folder(train_dir)
        test_data = object_detector.Dataset.from_pascal_voc_folder(test_dir)

    print("train_data size: ", train_data.size)
    print("test_data size: ", test_data.size)

    spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG
    hparams = object_detector.HParams(export_dir=output_dir, learning_rate=args.train_learning_rate
                                      , batch_size=args.train_batch_size, epochs=args.train_epochs)
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams
    )

    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=test_data,
        options=options)
    model.export_model(args.out_model_name)

    loss, coco_metrics = model.evaluate(test_data, batch_size=4)
    print(f"Validation loss: {loss}")
    print(f"Validation coco metrics: {coco_metrics}")
