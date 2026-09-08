import cv2

for backend, name in [
    (cv2.CAP_ANY, "ANY"),
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF")
]:
    print(f"\nTesting {name}")

    cap = cv2.VideoCapture(1, backend)

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()
        print("Read:", ret)

        if ret:
            cv2.imshow(name, frame)
            cv2.waitKey(2000)

    cap.release()

cv2.destroyAllWindows()