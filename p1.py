import cv2
import numpy as np

# Read input image
img = cv2.imread("audi.jpg")

# Check if the image is loaded
if img is None:
    print("Error: Image not loaded.")
    exit()

# Convert input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Read Haar cascade for number plate detection
cascade = cv2.CascadeClassifier('haarcascade_number_plate.xml')

# Check if the cascade is loaded
if cascade.empty():
    print("Error: Haar cascade not loaded.")
    exit()

# Detect license number plates
plates = cascade.detectMultiScale(gray, 1.2, 5)
print('Number of detected license plates:', len(plates))
t=1
# Loop over all plates
for (x, y, w, h) in plates:
    # Draw bounding rectangle around the license number plate
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gray_plates = gray[y:y+h, x:x+w]
    color_plates = img[y:y+h, x:x+w]
    
    # Save number plate detected
    cv2.imwrite('output/Numberplate'+str(t)+'.jpg', gray_plates)
    cv2.imshow('Number Plate', gray_plates)
    cv2.imshow('Number Plate Image', img)
    cv2.waitKey(0)
    t+=1

cv2.destroyAllWindows()
