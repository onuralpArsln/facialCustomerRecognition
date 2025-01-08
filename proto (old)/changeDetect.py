import cv2
import numpy as np

class ChangedPixelsDetector:
    def __init__(self, camera_index=0, blur_kernel_size=(5, 5), threshold_value=30, dilation_kernel_size=(3, 3), dilation_iterations=1):
        # Initialize parameters
        self.camera_index = camera_index
        self.blur_kernel_size = blur_kernel_size
        self.threshold_value = threshold_value
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_iterations = dilation_iterations

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video capture.")
        
        # Read the first frame to initialize background
        ret, self.prev_frame = self.cap.read()
        if not ret:
            raise ValueError("Error: Could not read from camera.")
        
        # Apply Gaussian blur to the first frame to suppress noise
        self.prev_frame = cv2.GaussianBlur(self.prev_frame, self.blur_kernel_size, 0)

    def detect_changed_pixels(self):
        while True:
            # Capture the current frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from camera.")
                break

            # Apply Gaussian blur to suppress noise
            blurred_frame = cv2.GaussianBlur(frame, self.blur_kernel_size, 0)

            # Calculate absolute difference between the blurred current and previous frames
            diff = cv2.absdiff(self.prev_frame, blurred_frame)

            # Convert the difference to grayscale and threshold it
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_thresh = cv2.threshold(diff_gray, self.threshold_value, 255, cv2.THRESH_BINARY)

            # Denoise the binary mask using morphological operations
            kernel = np.ones(self.dilation_kernel_size, np.uint8)
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)

            # Apply dilation to include surrounding pixels (3-pixel range effect)
            dilated_diff = cv2.dilate(diff_thresh, kernel, iterations=self.dilation_iterations)

            # Apply the dilated mask to the original frame to isolate changed regions with the area effect
            changed_pixels = cv2.bitwise_and(frame, frame, mask=dilated_diff)

            # Display the result
            cv2.imshow("Changed Pixels with Area Effect", changed_pixels)

            # Update the previous frame
            self.prev_frame = blurred_frame

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

# Example of using the class
if __name__ == "__main__":
    # Instantiate the class with custom parameters (dilation size and iterations are adjustable)
    detector = ChangedPixelsDetector(camera_index=0, blur_kernel_size=(5, 5), threshold_value=30, dilation_kernel_size=(1, 1), dilation_iterations=2)
    detector.detect_changed_pixels()
