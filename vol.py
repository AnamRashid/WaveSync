import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)


prev_distance = None


gesture_distance_threshold = 50  


hand_gesture_active = False

while cap.isOpened():  
    try:
        
        ret, frame = cap.read()
        if not ret:
            continue  

        x, y, c = frame.shape

        
        frame = cv2.flip(frame, 1)

        
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        result = hands.process(framergb)

        
        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                landmarks = []
                for lm in handslms.landmark:
                    
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]

               
                distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                
                if distance < gesture_distance_threshold:
                    hand_gesture_active = True
                else:
                    hand_gesture_active = False

                
                if hand_gesture_active:
                    pyautogui.press('volumedown')
                else:
                    pyautogui.press('volumeup')
                    

        
        cv2.imshow("Hand Gesture Volume Control", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
