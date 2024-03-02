import cv2
import mediapipe as mp
import numpy as np
import ctypes
import math
import time

# Function to move the mouse cursor
def move_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)

# Function to simulate a left mouse click
def left_click():
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # Mouse left button down
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # Mouse left button up

# Function to simulate a right mouse click
def right_click():
    ctypes.windll.user32.mouse_event(8, 0, 0, 0, 0)  # Mouse right button down
    ctypes.windll.user32.mouse_event(16, 0, 0, 0, 0)  # Mouse right button up

# Initialize MediaPipe for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam for hand detection
cap = cv2.VideoCapture(0)

# Set camera resolution and frame rate
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)

# Threshold distance between index and middle fingers for click action
click_distance_threshold = 50  # Adjust this value as needed

# Variables for tap detection
last_click_time = 0
double_click_threshold = 0.5  # seconds

# Main loop
while cap.isOpened():  # Check if the webcam is open
    try:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue  # Skip the rest of the loop if frame is None

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(frame_rgb)

        # Post-process the result
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_x, index_y = None, None
                middle_x, middle_y = None, None
                # Extracting landmark coordinates
                for i, lm in enumerate(hand_landmarks.landmark):
                    if i == 8:  # Index finger tip
                        index_x, index_y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    elif i == 12:  # Middle finger tip
                        middle_x, middle_y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

                # Check if both index and middle fingers are detected
                if index_x is not None and index_y is not None and middle_x is not None and middle_y is not None:
                    # Calculate the distance between index and middle fingers
                    distance = math.sqrt((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2)
                    # Move the cursor to the index finger tip position
                    move_cursor(index_x, index_y)
                    # Draw circles at the index and middle finger tip positions
                    cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (middle_x, middle_y), 8, (0, 255, 0), cv2.FILLED)
                    # Check if the distance is below the threshold for clicking
                    if distance < click_distance_threshold:
                        # Check if it's a single tap or double tap
                        current_time = time.time()
                        if current_time - last_click_time < double_click_threshold:
                            right_click()  # Double tap, perform right-click
                        else:
                            left_click()  # Single tap, perform left-click
                        last_click_time = current_time

        # Show the final output
        cv2.imshow("Hand Gesture Mouse Control", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
