import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Webcam setup
cap = cv2.VideoCapture(0)

# Set camera resolution and frame rate
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 30)

# Define sign language mapping (extend as needed)
sign_mapping = {
    "thumbs_up": "Hello everyone!",
    "peace": "Peace be with you!",
    "fist": "Rock on!",
    "five": "High five!",
    "none": "No sign detected"
}

# Initialize variables for hand tracking
current_signs = []

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        current_signs = []  # Reset current signs for this frame
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the landmarks for thumb, index, middle, and ring fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Get the y-coordinates of thumb, index, middle, and ring finger tips
            thumb_y = int(thumb_tip.y * frame.shape[0])
            index_y = int(index_tip.y * frame.shape[0])
            middle_y = int(middle_tip.y * frame.shape[0])
            ring_y = int(ring_tip.y * frame.shape[0])

            # Check the relative position of fingers to determine the sign
            if thumb_y < index_y:
                if middle_y < index_y:
                    current_signs.append("peace")
                else:
                    current_signs.append("thumbs_up")
            elif ring_y < middle_y:
                current_signs.append("five")
            else:
                current_signs.append("open hand")

    # Display the current signs and messages
    sign_message = ", ".join([sign_mapping.get(sign, 'Unknown') for sign in current_signs])
    cv2.putText(frame, f"Signs: {', '.join(current_signs)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Messages: {sign_message}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Exit the program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
