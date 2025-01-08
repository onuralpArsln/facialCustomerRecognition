import cv2
import time
import os
import sys
from urllib import request

def download_models():
    """Download the required model files if they don't exist."""
    base_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
    model_urls = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": base_url + "res10_300x300_ssd_iter_140000.caffemodel"
    }

    for filename, url in model_urls.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                sys.exit(1)

def main():
    # Download model files if they don't exist
    download_models()

    # Load DNN model files
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = "deploy.prototxt"
    
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
    except cv2.error as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)

    # Calculate FPS using a rolling average
    fps_history = []
    fps_window = 30  # Number of frames to average

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Resize frame
        frame = cv2.resize(frame, (320, 240))
        h, w = frame.shape[:2]

        # Face detection with DNN
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Draw detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                x = max(0, min(x, w))
                y = max(0, min(y, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence*100:.1f}%", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Calculate and smooth FPS
        fps = 1 / (time.time() - start_time)
        fps_history.append(fps)
        if len(fps_history) > fps_window:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("DNN Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()