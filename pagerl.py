import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam for Hand Detection
cap = cv2.VideoCapture(0)

# Constants for screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Constants for scrolling
SCROLL_SPEED = 10  # You can adjust this value as needed

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
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
            # Calculate the average y-coordinate of all landmarks
            avg_y = sum([lm.y for lm in handslms.landmark]) / len(handslms.landmark)
            # Calculate the average x-coordinate of all landmarks
            avg_x = sum([lm.x for lm in handslms.landmark]) / len(handslms.landmark)

           
            # Determine scrolling direction based on hand position
            if avg_x < 0.4:  # Hand is moving left (for page up)
                pyautogui.scroll(SCROLL_SPEED)
            elif avg_x > 0.6:  # Hand is moving right (for page down)
                pyautogui.scroll(-SCROLL_SPEED)

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # Show the final output
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()