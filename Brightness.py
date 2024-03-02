import cv2
import mediapipe as mp
import numpy as np
import ctypes
import os



def change_gamma_values(lp_ramp, brightness):
  
    for i in range(256):
        i_value = i * (brightness + 128)
        if i_value > 65535:
            i_value = 65535
        lp_ramp[0][i] = lp_ramp[1][i] = lp_ramp[2][i] = i_value
    return lp_ramp




def set_brightness(brightness):
    try:
        hdc = ctypes.windll.user32.GetDC(None)
        gamma_array = ((ctypes.c_ushort * 256) * 3)()
        ctypes.windll.gdi32.GetDeviceGammaRamp(hdc, ctypes.byref(gamma_array))
        gamma_array = change_gamma_values(gamma_array, brightness)
        ctypes.windll.gdi32.SetDeviceGammaRamp(hdc, ctypes.byref(gamma_array))
        ctypes.windll.user32.ReleaseDC(None, hdc)
        print(f"Brightness set to {brightness}%")
    except Exception as e:
        print(f"Error setting brightness: {e}")


if __name__ == '__main__':
    brightness = 100  

    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    
    cap = cv2.VideoCapture(0)

    
    cap.set(3, 640)  
    cap.set(4, 480) 
    cap.set(cv2.CAP_PROP_FPS, 30)

   
    gesture_distance_threshold = 50 

    while cap.isOpened(): 
        try:
         
            ret, frame = cap.read()
            if not ret:
                continue 

            x, y, c = frame.shape

           
            frame = cv2.flip(frame, 1)

            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            result = hands.process(frame_rgb)

            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                       
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]

                   
                    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                   
                    if distance < gesture_distance_threshold:
                        
                        brightness -= 10
                        if brightness < 0:
                            brightness = 0

                       
                    else:
                      
                        brightness += 10
                        if brightness > 255:
                            brightness = 255
                       
                    set_brightness(brightness)  
                    
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            cv2.imshow("Hand Gesture Brightness Control", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

    
    cap.release()
    cv2.destroyAllWindows()
