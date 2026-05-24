import cv2
from src.detector import HandDetector

cap = cv2.VideoCapture(0)

detector =HandDetector()

while True:
    success, img = cap.read()
    if not success:
        break

    img=cv2.flip(img,1)

    results =detector.detect(img)

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()