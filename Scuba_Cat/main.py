import cv2
import numpy as np
import pyautogui
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


frames_cat = load_gif_frames("assets/cat_animation.gif")
frames_nick = load_gif_frames("assets/nick_animation.gif")

print(f"Cat frames: {len(frames_cat)}")
print(f"Nick frames: {len(frames_nick)}")  

GIF_SIZE= (500,400)


cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

GIF_WIN_1 ="GifOverlay"
cv2.namedWindow(GIF_WIN_1, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(GIF_WIN_1, cv2.WND_PROP_TOPMOST, 1)

GIF_WIN_2="GifOverlay2"
cv2.namedWindow(GIF_WIN_2, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(GIF_WIN_2, cv2.WND_PROP_TOPMOST, 1)

gif_w , gif_h =GIF_SIZE

overlay_cat = Overlay(frames_cat)
overlay_nick = Overlay(frames_nick)

detector = HandDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    mouse_x, mouse_y= pyautogui.position()

    img = cv2.flip(img, 1)

    results = detector.detect(img)
    lm_list = detector.find_positions(img, results)

    hand_detected = len(lm_list) > 0
    overlay_cat.set_visible(hand_detected)
    overlay_nick.set_visible(hand_detected)
    overlay_cat.next_frame()
    overlay_nick.next_frame()
 

   

    if hand_detected:
        cv2.putText(img, "HAND DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", img)

    if overlay_cat.is_visible():
        current_frame = overlay_cat.get_current_frame()           # BGRA
        gif_resized = cv2.resize(current_frame, GIF_SIZE)     # match your size
        gif_bgr = cv2.cvtColor(gif_resized, cv2.COLOR_BGRA2BGR)
 
        win_x = mouse_x - gif_w // 2
        win_y = mouse_y - gif_h // 2
 
        cv2.resizeWindow(GIF_WIN_1, gif_w, gif_h)
        cv2.moveWindow(GIF_WIN_1, win_x, win_y)
        cv2.imshow(GIF_WIN_1, gif_bgr)
    else:
        # Move off-screen and show blank to keep window alive
        cv2.moveWindow(GIF_WIN_1, -9999, -9999)
        cv2.imshow(GIF_WIN_1, np.zeros((gif_h, gif_w, 3), dtype=np.uint8))


    if overlay_nick.is_visible():
        current_frame = overlay_nick.get_current_frame()
        gif_resized = cv2.resize(current_frame, GIF_SIZE)
        gif_bgr = cv2.cvtColor(gif_resized, cv2.COLOR_BGRA2BGR)
        win_x = mouse_x + gif_w // 2
        win_y = mouse_y - gif_h // 2
        cv2.resizeWindow(GIF_WIN_2, gif_w, gif_h)
        cv2.moveWindow(GIF_WIN_2, win_x, win_y)
        cv2.imshow(GIF_WIN_2, gif_bgr)
    else:
        cv2.moveWindow(GIF_WIN_2, -9999, -9999)
        cv2.imshow(GIF_WIN_2, np.zeros((gif_h, gif_w, 3), dtype=np.uint8))

    
      

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()