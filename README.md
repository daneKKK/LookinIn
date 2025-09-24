# Installation

1. create conda environment:

```
conda create -n lookin python=3.10

conda activate lookin
```

2. install required libraries:

```pip install -r requirements.txt```

3. download the mediapipe model:

```wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O face_landmarker.task```

3. run tracker

```python tracker.py```