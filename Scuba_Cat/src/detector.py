import cv2
import mediapipe as mp 


class HandDetector : 
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands= self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils


    def detect(self, img):
        img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results= self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return results

      

