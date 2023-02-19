import cv2
import face_recognition
import os
from comparing_faces import get_face_locations_and_encodings
import numpy as np
import datetime

folder_path = 'sample_images'

def collect_encodings(folder_path):
    photos_list = os.listdir(folder_path)
    total_images_faces_encodings = {}
    for image in photos_list:
        image_path = os.path.join(folder_path,image)
        result = get_face_locations_and_encodings(image_path=image_path)
        total_images_faces_encodings[image.split('.')[0]] = result['encodings']
        # image = face_recognition.load_image_file(image_path)
        # changed_main_img = cv2.imread(image_path)
        # changed_main_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.rectangle(changed_main_img, (result['face_locations'][3],result['face_locations'][0]),
        #         (result['face_locations'][1],result['face_locations'][2]),(255,0,0),3)
        # cv2.imshow('image',changed_main_img)
        # cv2.waitKey(0)
    return total_images_faces_encodings

encodings = collect_encodings(folder_path)
# print(encodings)

def on_boarding(name):
    with open('onboarding.csv', 'r+') as file:
        employees_list = file.readlines()
        name_list = []
        # print(employees_list)
        for line in employees_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.datetime.now()
            file.writelines(f"\n{name},{now}")



video = cv2.VideoCapture(0)

while True:
    captured, frame = video.read()
    resize_image = cv2.resize(frame,(0,0),None,0.25,0.25)
    image_frame = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)


    frame_locations = face_recognition.face_locations(image_frame)
    frame_encodings = face_recognition.face_encodings(image_frame,frame_locations)

    for encode_face,face_location in zip(frame_encodings,frame_locations):
        distance = face_recognition.face_distance([encodings[i] for i in encodings],encode_face)
        comparing = face_recognition.compare_faces([encodings[i] for i in encodings],encode_face)
        print(distance)
        min_encodings = np.argmin(distance)
        if comparing[min_encodings]:
            name = [i for i in encodings][min_encodings]
            on_boarding(name)
            print(min_encodings,name,comparing)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        
    # comparing = face_recognition.compare_faces([])
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) & 0xFF ==  ord('q'):
        break 
video.release()
video.destroyAllWindows()