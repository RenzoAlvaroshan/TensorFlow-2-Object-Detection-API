import cv2

cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)
#if you wanna use Web camera

# cam_ip = cv2.VideoCapture('http://username:password@IPAddress/video/mjpg.cgi')
#if you wanna use IP camera

cv2.namedWindow("window 1")
cv2.namedWindow("window 2")

img_counter = 0 #starts from 0°

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    if not ret1:
        print("failed to grab frame")
        break

    cv2.imshow("test", frame1)
    cv2.imshow("test", frame2)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name1 = "logitech_{}_0.jpeg".format(img_counter) #0 means 0°
        img_name2 = "tiza_logitech_{}_90.jpeg".format(img_counter) #90 means 90°
        
        cv2.imwrite(img_name1, frame1)
        print("{} written!".format(img_name1))

        img_counter += 5

cam1.release()
cam2.release()

cv2.destroyAllWindows()

