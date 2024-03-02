import cv2
import mediapipe as mp
import numpy as np
import ctypes
import math
import time


def move_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)


def left_click():
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  


def right_click():
    ctypes.windll.user32.mouse_event(8, 0, 0, 0, 0)  
    ctypes.windll.user32.mouse_event(16, 0, 0, 0, 0)  


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)


click_distance_threshold = 50  


last_click_time = time.time()
click_delay = 0.5  


while cap.isOpened():  
    try:
      
        ret, frame = cap.read()
        if not ret:
            continue  

       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       
        result = hands.process(frame_rgb)

     
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_x, index_y = None, None
                middle_x, middle_y = None, None
               
                for i, lm in enumerate(hand_landmarks.landmark):
                    if i == 8:
                        index_x, index_y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    elif i == 12: 
                        middle_x, middle_y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

                
                if index_x is not None and index_y is not None and middle_x is not None and middle_y is not None:
                   
                    distance = math.sqrt((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2)
                    
                    move_cursor(index_x, index_y)
                   
                    cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, (middle_x, middle_y), 8, (0, 255, 0), cv2.FILLED)
                   
                    if distance < click_distance_threshold:
                        
                        current_time = time.time()
                        if current_time - last_click_time < click_delay:
                            right_click()  
                        else:
                            left_click()  
                        last_click_time = current_time

        
        cv2.imshow("Hand Gesture Mouse Control", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error: {e}")


cap.release()
cv2.destroyAllWindows()