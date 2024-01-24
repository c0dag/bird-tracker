import cv2
from tracker import *

tracker = EuclideanDistTracker()
cap = cv2.VideoCapture(1)
object_detector = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=20)

cv2.createBackgroundSubtractorKNN()


while True:
    ret, frame = cap.read()
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        #area em qtd pixeis
        area = cv2.contourArea(cnt)
        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, h, w, id = box_id
        cv2.putText(frame, str(id), (x, y -15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 1) 
    cv2.imshow("Frame", frame)  
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()