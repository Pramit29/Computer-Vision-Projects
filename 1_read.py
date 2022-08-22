import cv2 as cv

# READING IMAGES ****************************************

# # returns the image in matrix format
# img = cv.imread('Resources/Photos/cat_large.jpg')

# # Displays the image with the name 'Cat'
# cv.imshow('Cat', img)

# # waits till any key is pressed
# cv.waitKey(0) 


# READING VIDEOS ***************************************

capture = cv.VideoCapture('Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    cv.imshow('Video',frame)
    
    # if letter d is pressed then break and stop reading frames of the video
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

# Release the video
capture.release()
# destroy the video window
cv.destroyAllWindows()