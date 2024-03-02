import cv2
import mediapipe as mp
import pyautogui
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


DELAY_BETWEEN_TRANSITIONS = 1

last_transition_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    current_time = time.time()

    if result.multi_hand_landmarks and current_time - last_transition_time >= DELAY_BETWEEN_TRANSITIONS:
        for handslms in result.multi_hand_landmarks:
            landmarks = [(lm.x * x, lm.y * y) for lm in handslms.landmark]

            
            index_finger_tip = landmarks[8]

            
            if index_finger_tip[0] < x / 2:
                pyautogui.hotkey('left')  
                last_transition_time = current_time  

            
            elif index_finger_tip[0] > x / 2:
                pyautogui.hotkey('right')  
                last_transition_time = current_time  

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
