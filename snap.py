import cv2
import mediapipe as mp
import pyautogui
import os
import winsound


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Failed to open webcam.")
    exit()


SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


SAVE_FOLDER = "screenshots"


if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


closed_palm = False


screenshot_sound_file = "ScreenShot.wav"  
if not os.path.exists(screenshot_sound_file):
    print(f"Error: Sound file '{screenshot_sound_file}' not found.")
    exit()

while True:
   
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

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
                landmarks.append((lmx, lmy))

           
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            
            if all(landmarks[i][1] < landmarks[i+1][1] for i in range(0, 4)):
                closed_palm = True
            else:
                closed_palm = False

           
            if closed_palm:
                screenshot = pyautogui.screenshot(region=(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
                screenshot_path = os.path.join(SAVE_FOLDER, f"screenshot_{len(os.listdir(SAVE_FOLDER)) + 1}.png")
                screenshot.save(screenshot_path)
                print("Screenshot saved:", screenshot_path)
                
                winsound.PlaySound(screenshot_sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)

   
    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
