# Multi-Method Object Detection Engine

## Overview
A comprehensive Python-based object detection and camera capture library using OpenCV, Dlib, and MediaPipe.

## Features
- Real-time face and body detection
- Multiple detection methods
- Camera capture and frame handling
- Configurable detection parameters

## Modules
- Detection Engine
- Camera Module
- Main Detection Script

## Requirements
- OpenCV (`cv2`)
- Dlib
- MediaPipe
- Python 3.x

## Installation
```bash
pip3 install opencv-python dlib mediapipe
```

## Usage Examples
### Integrated Detection Script
```python
# Runs real-time detection from camera
python main_detection.py
```

### Detailed Usage
```python
from detection_engine import DetectionEngine
from main_cam import MainCam

# Initialize components
detection_engine = DetectionEngine()
cam = MainCam()

# Capture and detect in real-time
frame = cam.captureFrame()
faces = detection_engine.detectFaceLocations(frame, method=0)
bodies = detection_engine.detectBodyLocations(frame, method=0)
```

## Main Detection Script Features
- Real-time camera feed
- Face and body detection
- Visualization of detected objects
- Continuous frame processing
- Quit option with 'q' key

## Detection Methods
- Face Detection: Haar Cascade (method 0)
- Body Detection: Haar Cascade (method 0)

## Keyboard Controls
- 'q': Quit the detection feed

## Limitations
- Depends on camera availability
- Detection accuracy varies by method and environment

## License
MIT License

## Contributing
Contributions welcome! Please submit pull requests or open issues.