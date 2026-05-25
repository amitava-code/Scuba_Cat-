import cv2
import numpy as np
from src.detector import HandDetector
from src.overlay_utils import Overlay


def load_gif_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frames.append(frame)

    cap.release()
    return frames


frames = load_gif_frames("assets/cat_animation.gif")

GIF_SIZE= (200,200)

mouse_x, mouse_y = 100, 100

def mouse_move(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_move)

GIF_WIN ="GifOverlay"
cv2.namedWindow(GIF_WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(GIF_WIN, cv2.WND_PROP_TOPMOST, 1)

gif_w , gif_h =GIF_SIZE

detector = HandDetector()
overlay = Overlay(frames)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    results = detector.detect(img)
    lm_list = detector.find_positions(img, results)

    hand_detected = len(lm_list) > 0
    overlay.set_visible(hand_detected)
    overlay.next_frame()

   

    if hand_detected:
        cv2.putText(img, "HAND DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", img)

    if overlay.is_visible():
        current_frame = overlay.get_current_frame()           # BGRA
        gif_resized = cv2.resize(current_frame, GIF_SIZE)     # match your size
        gif_bgr = cv2.cvtColor(gif_resized, cv2.COLOR_BGRA2BGR)
 
        win_x = mouse_x - gif_w // 2
        win_y = mouse_y - gif_h // 2
 
        cv2.resizeWindow(GIF_WIN, gif_w, gif_h)
        cv2.moveWindow(GIF_WIN, win_x, win_y)
        cv2.imshow(GIF_WIN, gif_bgr)
    else:
        # Move off-screen and show blank to keep window alive
        cv2.moveWindow(GIF_WIN, -9999, -9999)
        blank = np.zeros((gif_h, gif_w, 3), dtype=np.uint8)
        cv2.imshow(GIF_WIN, blank)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()