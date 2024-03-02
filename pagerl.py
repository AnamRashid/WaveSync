import cv2
import mediapipe as mp
import pyautogui


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


SCROLL_SPEED = 10  

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

   
    frame = cv2.flip(frame, 1)

   
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

   
    result = hands.process(framergb)

  
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            
            avg_y = sum([lm.y for lm in handslms.landmark]) / len(handslms.landmark)
            
            avg_x = sum([lm.x for lm in handslms.landmark]) / len(handslms.landmark)

           
            
            if avg_x < 0.4:  
                pyautogui.scroll(SCROLL_SPEED)
            elif avg_x > 0.6: 
                pyautogui.scroll(-SCROLL_SPEED)

            
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
