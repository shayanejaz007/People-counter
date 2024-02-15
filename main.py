import cv2
import numpy as np


def count_people():
    # Load the pre-trained MobileNet SSD model
    net = cv2.dnn.readNetFromCaffe(r"C:\Users\koks\Desktop\jarvis\people counter\MobileNetSSD_deploy.prototxt",
                                   r"C:\Users\koks\Desktop\jarvis\people counter\MobileNetSSD_deploy.caffemodel")

    # Open a connection to the webcam (usually index 0)
    cap = cv2.VideoCapture(0)

    # Set the resolution to 1280x720
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    # Initialize variables
    total_people_count = 0

    while True:
        # Read the next frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 300x300 for input to the model
        resized_frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(
            resized_frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Process the output to count people and draw bounding boxes
        people_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])

            if confidence > 0.8 and class_id == 15:  # Class 15 corresponds to 'person' in COCO dataset
                people_count += 1

                # Get the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([resized_frame.shape[1], resized_frame.shape[0],
                                                           resized_frame.shape[1], resized_frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Scale the bounding box back to the original frame size
                startX, startY, endX, endY = int(startX * frame.shape[1] / 300), int(startY * frame.shape[0] / 300), int(
                    endX * frame.shape[1] / 300), int(endY * frame.shape[0] / 300)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
                label = f"Person {people_count}"
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result on the frame
        cv2.putText(frame, f"People: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("People Counter", frame)

        total_people_count += people_count
        print(total_people_count)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Print the total number of people counted
    print(f"Total people counted: {total_people_count}")


count_people()
