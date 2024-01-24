import cv2
import numpy as np

kernel = np.ones((3,3))
min_area = 10
id = 0
thresh = 30

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)

def get_frame(cap):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture(1)
    scaling_factor = 0.9
    prev_frame = cv2.cvtColor(get_frame(cap), cv2.COLOR_BGR2GRAY)
    cur_frame = cv2.cvtColor(get_frame(cap), cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(get_frame(cap), cv2.COLOR_BGR2GRAY)
    prev_id = 0
    
    while True:
        frame_difference = frame_diff(prev_frame,cur_frame, next_frame)
        frame_difference = cv2.dilate(frame_difference, kernel)
        _,frame_th = cv2.threshold(frame_difference, thresh, 255, cv2.THRESH_BINARY)
        frame_th = cv2.dilate(frame_th, kernel)
        contours, _ = cv2.findContours(frame_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        next_frame_color = get_frame(cap)
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(next_frame_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
                # check if moving object is still the same as before
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                if centroid_x > 0 and centroid_y > 0:
                    if prev_id == 0:
                        id += 1
                        print("Moving object with ID {} detected".format(id))
                    else:
                        # calculate distance between centroids to check if it's the same object
                        prev_centroid_x, prev_centroid_y = prev_id[1], prev_id[2]
                        dist = np.sqrt((centroid_x - prev_centroid_x)**2 + (centroid_y - prev_centroid_y)**2)
                        if dist < 50:
                            print("Moving object with ID {} detected".format(prev_id[0]))
                        else:
                            id += 1
                            print("Moving object with ID {} detected".format(id))
                            
                    prev_id = (id, centroid_x, centroid_y)

        cv2.imshow("Camera Feed", next_frame_color)
        cv2.imshow("Object Movement", frame_difference)
        cv2.imshow("Thresholded Difference", frame_th)

        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = cv2.cvtColor(get_frame(cap), cv2.COLOR_BGR2GRAY)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
