import cv2

class Overlay:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0
        self.visible = False

    def set_visible(self, state: bool):
        self.visible = state

    def is_visible(self)-> bool:
        return self.visible
    
    def get_current_frame(self):
        return self.frames[self.index]

    def next_frame(self):
        self.index = (self.index + 1) % len(self.frames)

    def apply(self, frame, pos=(50, 50), size=(200, 200)):
        if not self.visible:
            return frame

        overlay_img = self.frames[self.index]

        x, y = pos
        w, h = size

        overlay_resized = cv2.resize(overlay_img, (w, h))

        fh, fw = frame.shape[:2]

        x = max(0, x)
        y = max(0, y)

        w = min(w, fw - x)
        h = min(h, fh - y)

        if w <= 0 or h <= 0:
            return frame

        overlay_resized = overlay_resized[0:h, 0:w]

        if overlay_resized.shape[2] == 4:
            overlay_rgb = overlay_resized[:, :, :3]
            alpha = overlay_resized[:, :, 3:] / 255.0

            frame[y:y+h, x:x+w] = (
                alpha * overlay_rgb +
                (1 - alpha) * frame[y:y+h, x:x+w]
            ).astype("uint8")
        else:
            frame[y:y+h, x:x+w] = overlay_resized

        return frame