import cv2

class Overlay:
    def __init__(self, img_path):
        self.overlay = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        self.visible= False


        if self.overlay is None:
            raise ValueError("Overlay image not found")
        

    def set_visible(self, state: bool):
        self.visible = state

    def apply(self, frame, pos=(50, 50), size=(200, 200)):
        if not self.visible:
            return frame
        
        x,y= pos
        w,h= size

        overlay_resized = cv2.resize(self.overlay, (w,h))

        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0

            for c in range(3):
                frame[y:y+h, x:x+w, c] = (
                    alpha * overlay_resized[:, :, c] +
                    (1 - alpha) * frame[y:y+h, x:x+w, c]
                )
        else:
            frame[y:y+h, x:x+w] = overlay_resized

        return frame
