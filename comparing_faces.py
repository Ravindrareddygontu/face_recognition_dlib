import face_recognition
import cv2
import numpy as np

def get_face_locations_and_encodings(image_path):
    ''' 
        this function returns the encodings and face locations of the image
        '''
    
    original_image = face_recognition.load_image_file(image_path)  #this will give bgr image

    convert_to_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  #converting bgr to rgb image

    image_face_locations = face_recognition.face_locations(convert_to_rgb)[0] #finding face locations in the image

    image_encodings = face_recognition.face_encodings(convert_to_rgb)[0]  #finding encodings


    return {'face_locations':image_face_locations, 'encodings':image_encodings}


main_img = face_recognition.load_image_file('sample_images\pawan_kalyan.jpg')
test_img = face_recognition.load_image_file(r'sample_images\sundar_pichai.jpg')
# print(main_img)

changed_main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
changed_test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

main_face_locations = face_recognition.face_locations(changed_main_img)[0]
test_face_locations = face_recognition.face_locations(changed_test_img)[0]
# print(main_face_locations, test_face_locations)
main_face_encodings = face_recognition.face_encodings(changed_main_img)[0]
test_face_encodings = face_recognition.face_encodings(changed_test_img)[0]
# print(main_face_encodings,test_face_encodings,'9999999')

result = face_recognition.compare_faces([test_face_encodings], main_face_encodings)

faces_distance = face_recognition.face_distance([test_face_encodings], main_face_encodings)
print(result,faces_distance)
cv2.rectangle(changed_main_img, (main_face_locations[3], main_face_locations[0]),
              (main_face_locations[1],main_face_locations[2]),(255,0,0),3)
cv2.rectangle(changed_test_img, (test_face_locations[3], test_face_locations[0]),
              (test_face_locations[1],test_face_locations[2]),(255,0,0),3)

cv2.imshow('pawan kalyan', changed_main_img)
cv2.imshow('test_kalan', changed_test_img)

cv2.waitKey(0)
