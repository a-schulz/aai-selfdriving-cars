---
comments: true
description: Learn how to use Ultralytics YOLO for object tracking in video streams. Guides to use different trackers and customise tracker configurations.
keywords: Ultralytics, YOLO, object tracking, video streams, BoT-SORT, ByteTrack, Python guide, CLI guide
---

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png">

Object tracking is a task that involves identifying the location and class of objects, then assigning a unique ID to
that detection in video streams.

The output of tracker is the same as detection with an added object ID.

## Available Trackers

The following tracking algorithms have been implemented and can be enabled by passing `tracker=tracker_type.yaml`

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - `botsort.yaml`
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - `bytetrack.yaml`

The default tracker is BoT-SORT.

## Tracking

Use a trained YOLOv8n/YOLOv8n-seg model to run tracker on video streams.

!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load an official detection model
        model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
        model = YOLO('path/to/best.pt')  # load a custom model

        # Track with the model
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True)
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml")
        ```
    === "CLI"

        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc"  # official detection model
        yolo track model=yolov8n-seg.pt source=...   # official segmentation model
        yolo track model=path/to/best.pt source=...  # custom model
        yolo track model=path/to/best.pt  tracker="bytetrack.yaml" # bytetrack tracker

        ```

As in the above usage, we support both the detection and segmentation models for tracking and the only thing you need to
do is loading the corresponding (detection or segmentation) model.

## Configuration

### Tracking

Tracking shares the configuration with predict, i.e `conf`, `iou`, `show`. More configurations please refer
to [predict page](https://docs.ultralytics.com/modes/predict/).
!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", conf=0.3, iou=0.5, show=True)
        ```
    === "CLI"

        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" conf=0.3, iou=0.5 show

        ```

### Tracker

We also support using a modified tracker config file, just copy a config file i.e `custom_tracker.yaml`
from [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers) and modify
any configurations(expect the `tracker_type`) you need to.
!!! example ""

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", tracker='custom_tracker.yaml')
        ```
    === "CLI"

        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc" tracker='custom_tracker.yaml'
        ```

Please refer to [ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)
page
