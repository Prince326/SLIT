def gen():
    while True:
        # Capture an image from the webcam
        ret, img = cap.read()
        if not ret:
            break

        # Resize the image
        img = cv.resize(img, (640, 480))

        # Perform object detection
        results = model(img, stream=True)
        for info in results:
            parameters = info.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_detect = box.cls[0]
                class_detect = int(class_detect)
                class_detect = classnames[class_detect]
                conf = math.ceil(confidence * 100)

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw a bounding box around the detected object and label it
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(img, f'{class_detect}', [x1 + 8, y1 + 50],
                                   scale=2, thickness=2)

        # Return the image as a video stream
        frame = cv.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')