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

    lm_list = detector.find_positions(img, results)


    hand_detected = len(lm_list) > 0

    if hand_detected:
        cv2.putText(img, "HAND DETECTED", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()