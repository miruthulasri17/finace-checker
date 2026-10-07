import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Global variables for speech coordination
engine_lock = threading.Lock()

def speak(text):
    """Function to speak text in a separate thread without blocking the main loop."""
    def run_speech():
        with engine_lock:
            engine.say(text)
            engine.runAndWait()
    
    # Run in a daemon thread so it doesn't prevent the script from exiting
    thread = threading.Thread(target=run_speech, daemon=True)
    thread.start()

def run_detector():
    # Load the trained model
    # If you haven't trained yet, you can use 'yolov8n.pt' for testing
    # Once trained, replace with 'runs/detect/indian_currency_detector/weights/best.pt'
    try:
        model = YOLO('runs/detect/indian_currency_detector/weights/best.pt')
    except:
        print("Trained model not found. Using base YOLOv8n for demonstration.")
        model = YOLO('yolov8n.pt')

    # Constants for TTS cooldown and detection logic
    COOLDOWN_SECONDS = 3
    last_speak_time = 0
    no_detection_start_time = time.time()
    NO_DETECTION_THRESHOLD = 5 # Say "No currency found" after 5 seconds of empty frames
    last_no_detection_speak = 0

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Indian Currency Detector...")
    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference
        results = model(frame, conf=0.5, verbose=False)
        
        detections = results[0].boxes
        currency_detected = False
        current_time = time.time()

        # Process detections
        for box in detections:
            # Get class ID and label
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Filter: Only detect currency (assuming classes are defined correctly in data.yaml)
            # If using base yolov8n, we might see many things; in custom model, it should only see currency.
            
            currency_detected = True
            no_detection_start_time = current_time # Reset empty counter

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # TTS logic with cooldown
            if current_time - last_speak_time > COOLDOWN_SECONDS:
                speak(f"Detected {label}")
                last_speak_time = current_time

        # "No currency found" logic
        if not currency_detected:
            if current_time - no_detection_start_time > NO_DETECTION_THRESHOLD:
                if current_time - last_no_detection_speak > COOLDOWN_SECONDS * 2:
                    speak("No currency found")
                    last_no_detection_speak = current_time
                    # Keep resetting the start time so it doesn't repeat every frame
                    no_detection_start_time = current_time 

        # Display the frame
        cv2.imshow("Indian Currency Detector", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detector()
