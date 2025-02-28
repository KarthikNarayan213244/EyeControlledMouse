import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Capture webcam
cam = cv2.VideoCapture(0)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Variables for click detection
prev_left_eye_state = None
prev_right_eye_state = None
click_threshold = 0.01  # Adjust this threshold for click detection
dragging = False  # Drag and drop state

# Smoothing variables for cursor movement
cursor_x, cursor_y = 0, 0
alpha = 0.2  # Smoothing factor

# Create a named window to ensure it opens properly
cv2.namedWindow("Eye Controlled Mouse", cv2.WINDOW_NORMAL)

# Small delay to let OpenCV initialize properly
time.sleep(1)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip image to avoid mirror effect
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process image with FaceMesh
    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Extract landmark points for tracking
        eye_cursor_point = landmarks[1]  # Nose bridge point for cursor movement
        left_eye = [landmarks[145], landmarks[159]]  # Upper and lower eyelid
        right_eye = [landmarks[374], landmarks[386]]  # Upper and lower eyelid

        # Convert normalized coordinates to screen coordinates
        x = int(eye_cursor_point.x * frame_w)
        y = int(eye_cursor_point.y * frame_h)

        # Smooth cursor movement using exponential moving average
        cursor_x = int(alpha * x + (1 - alpha) * cursor_x)
        cursor_y = int(alpha * y + (1 - alpha) * cursor_y)

        # Map to screen size
        screen_x = int((cursor_x / frame_w) * screen_w)
        screen_y = int((cursor_y / frame_h) * screen_h)
        
        # Move the cursor
        pyautogui.moveTo(screen_x, screen_y)

        # Draw tracking point
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Eye blink detection for clicking
        left_eye_distance = abs(left_eye[0].y - left_eye[1].y)
        right_eye_distance = abs(right_eye[0].y - right_eye[1].y)

        left_eye_closed = left_eye_distance < click_threshold
        right_eye_closed = right_eye_distance < click_threshold

        # Left eye blink → Left click
        if left_eye_closed and prev_left_eye_state is False:
            pyautogui.click()
            print("Left Click")
            time.sleep(0.2)  # Avoid multiple clicks

        # Right eye blink → Right click
        if right_eye_closed and prev_right_eye_state is False:
            pyautogui.click(button='right')
            print("Right Click")
            time.sleep(0.2)

        # Both eyes closed → Drag and Drop
        if left_eye_closed and right_eye_closed:
            if not dragging:
                pyautogui.mouseDown()
                print("Drag Start")
                dragging = True
        else:
            if dragging:
                pyautogui.mouseUp()
                print("Drag Stop")
                dragging = False

        prev_left_eye_state = left_eye_closed
        prev_right_eye_state = right_eye_closed

    # Show output
    cv2.imshow("Eye Controlled Mouse", frame)

    # Ensure window stays on top
    cv2.setWindowProperty("Eye Controlled Mouse", cv2.WND_PROP_TOPMOST, 1)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
