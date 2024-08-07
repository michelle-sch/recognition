import cv2

#loading pre_trained data
face_data_set = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choosing personal image
face_img = cv2.imread('mom.png')

#converting to gray scale for recogition
gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

#changes rectangle size accoring to size of face in image
img_box_size = face_data_set.detectMultiScale(gray_img)
#print(img_box_size), which gives me coordinates for image squares

#image square format (img, (x, y), (x+w), (y+h), (B, G, R), thickness) 

#iterate through different faces
for (x, y, w, h) in img_box_size:

    #creates square
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (209, 201, 124), 2)
    
    #creates coordinates and prints text
    coordinates_top = (f"({x}, {y})")
    coordinates_bottom = (f"({x+w}), ({y+h})")
    cv2.putText(face_img, coordinates_top, (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (209, 201, 124), 2, cv2.LINE_AA)
    cv2.putText(face_img, coordinates_bottom, ((x+w) - 100 , (y+h) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209, 201, 124), 2, cv2.LINE_AA)

#show the image
cv2.imshow('Programmed Face Detector', face_img)

#exit execution by pressing any key
cv2.waitKey()

print("Project Executed")