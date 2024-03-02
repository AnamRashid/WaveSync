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
    brightness = 100  # can be any value in 0-255 (as per my system)

    # Initialize MediaPipe for hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # Initialize the webcam for hand detection
    cap = cv2.VideoCapture(0)

    # Set camera resolution and frame rate
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Set the distance threshold for hand gesture activation
    gesture_distance_threshold = 50  # Adjust this value as needed

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = hands.process(frame_rgb)

            # Post-process the result
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        # Extracting landmark coordinates
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])

                    # Get the coordinates of the thumb tip and index finger tip
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]

                    # Calculate the distance between thumb and index finger tips
                    distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                    # Check if both thumb and index tips are close together
                    if distance < gesture_distance_threshold:
                        # Increase brightness
                        brightness -= 10
                        if brightness < 0:
                            brightness = 0

                       
                    else:
                        # Decrease brightness
                        brightness += 10
                        if brightness > 255:
                            brightness = 255
                       
                    set_brightness(brightness)  # Set brightness based on hand gesture

                    # Drawing landmarks on frames
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the final output
            cv2.imshow("Hand Gesture Brightness Control", frame)

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
