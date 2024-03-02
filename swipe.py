import cv2
import mediapipe as mp
import numpy as np
import ctypes
import math
import win32api
import win32con

# Function to move the mouse cursor
def move_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)

# Function to perform page up action
def page_up():
    win32api.keybd_event(win32con.VK_PRIOR, 0, 0, 0)
    win32api.keybd_event(win32con.VK_PRIOR, 0, win32con.KEYEVENTF_KEYUP, 0)

# Function to perform page down action
def page_down():
    win32api.keybd_event(win32con.VK_NEXT, 0, 0, 0)
    win32api.keybd_event(win32con.VK_NEXT, 0, win32con.KEYEVENTF_KEYUP, 0)

# Initialize MediaPipe for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam for hand detection
cap = cv2.VideoCapture(0)

# Set camera resolution and frame rate
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)

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
                index_tip_up = False
                index_tip_down = False
                # Extracting landmark coordinates
                for idx, lm in enumerate(hand_landmarks.landmark):
                    if idx == 8:  # Index finger tip
                        if lm.y * frame.shape[0] < hand_landmarks.landmark[0].y * frame.shape[0]:
                            index_tip_up = True
                        else:
                            index_tip_down = True

                if index_tip_up:
                    page_up()
                elif index_tip_down:
                    page_down()

        # Show the final output
        cv2.imshow("Hand Gesture Page Scroll", frame)

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
