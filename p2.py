import cv2
import numpy as np

# Load the Haar cascade for license plate detection
carPlatesCascade = cv2.CascadeClassifier('haarcascade_number_plate.xml')

# Open the video file
cap = cv2.VideoCapture('carVideo.mp4')

# Check if video file opened successfully
if not cap.isOpened():
    print('Error reading video')
    exit()

# Define the desired output size
output_size = (640, 360)  # Adjust the dimensions as needed

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the frame
    car_plates = carPlatesCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(25, 25))

    # Color detected license plates with a solid red block
    for (x, y, w, h) in car_plates:
        # Draw a red rectangle around the detected license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Fill the license plate area with red
        frame[y:y + h, x:x + w] = (0, 0, 255)

    # Resize the frame to the desired output size
    resized_frame = cv2.resize(frame, output_size)

    # Display the resized frame with red license plates
    cv2.imshow('Video', resized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
