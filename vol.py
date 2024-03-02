import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam for Hand Detection
cap = cv2.VideoCapture(0)

# Set camera resolution and frame rate
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)

# Previous distance between thumb and index finger tips
prev_distance = None

# Set the distance threshold for hand gesture activation
gesture_distance_threshold = 50  # Adjust this value as needed

# Flag to indicate whether hand gesture is active
hand_gesture_active = False

while cap.isOpened():  # Check if the webcam is open
    try:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            continue  # Skip the rest of the loop if frame is None

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # Post-process the result
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handslms.landmark:
                    # Extracting landmark coordinates
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Get the coordinates of the thumb tip and index finger tip
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

                # Calculate the distance between thumb and index finger tips
                distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                # Check if both thumb and index tips are close together
                if distance < gesture_distance_threshold:
                    hand_gesture_active = True
                else:
                    hand_gesture_active = False

                # Adjust volume based on hand gesture activation
                if hand_gesture_active:
                    pyautogui.press('volumedown')
                else:
                    pyautogui.press('volumeup')
                    

        # Show the final output
        cv2.imshow("Hand Gesture Volume Control", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
