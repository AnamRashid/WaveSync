import cv2
import mediapipe as mp
import pyautogui
import os
import winsound

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam for Hand Detection
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()

# Constants for screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Folder to save screenshots
SAVE_FOLDER = "screenshots"

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Flag to keep track of closed palm state
closed_palm = False

# Load sound file for screenshot capture
screenshot_sound_file = "ScreenShot.wav"  # Provide the path to your sound file
if not os.path.exists(screenshot_sound_file):
    print(f"Error: Sound file '{screenshot_sound_file}' not found.")
    exit()

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

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
            # Extracting landmark coordinates
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append((lmx, lmy))

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Check if hand is in closed palm position
            if all(landmarks[i][1] < landmarks[i+1][1] for i in range(0, 4)):
                closed_palm = True
            else:
                closed_palm = False

            # If hand is in closed palm position, take and save a screenshot
            if closed_palm:
                screenshot = pyautogui.screenshot(region=(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
                screenshot_path = os.path.join(SAVE_FOLDER, f"screenshot_{len(os.listdir(SAVE_FOLDER)) + 1}.png")
                screenshot.save(screenshot_path)
                print("Screenshot saved:", screenshot_path)
                # Play sound effect
                winsound.PlaySound(screenshot_sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)

    # Show the final output
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
