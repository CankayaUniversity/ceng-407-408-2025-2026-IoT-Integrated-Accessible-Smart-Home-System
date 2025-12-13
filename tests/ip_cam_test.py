import cv2

url ="http://192.x.x.x:4747/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
	print("IP cam source cannot be opened")
	exit()
while True:
	ret,frame = cap.read()
	if not ret:
		print("cant receive frame")
		break
	frame = cv2.resize(frame,(0, 0), fx=0.5, fy=0.5)
	cv2.imshow("tel cam", frame)
	if cv2.waitKey(1)&0xFF == 27: #ESC
		break
cap.release()
cv2.destroyAllWindows()

