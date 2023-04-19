# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 18:35:17 2023

@author: ARPITA
"""

from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
from tensorflow.keras.preprocessing import image
#from keras.models package
from keras.models import model_from_json

app=Flask(__name__)
camera = cv2.VideoCapture(0)
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r",encoding="utf-8").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

abhi_image = face_recognition.load_image_file('images/samples/abhi.jpg')
abhi_face_encodings = face_recognition.face_encodings(abhi_image)[0]

arpita_image = face_recognition.load_image_file('images/samples/arpitaimage.jpeg')
arpita_face_encodings = face_recognition.face_encodings(arpita_image)[0]

shahrukh_image = face_recognition.load_image_file('images/samples/shahrukh.jpg')
shahrukh_face_encodings = face_recognition.face_encodings(shahrukh_image)[0]



harshita_image = face_recognition.load_image_file('images/samples/harshita.jpeg')
harshita_face_encodings = face_recognition.face_encodings(harshita_image)[0]

janhavi_image = face_recognition.load_image_file('images/samples/janhavi.jpeg')
janhavi_face_encodings = face_recognition.face_encodings(janhavi_image)[0]

rajiv_image = face_recognition.load_image_file('images/samples/rajiv.jpeg')
rajiv_face_encodings = face_recognition.face_encodings(rajiv_image)[0]

gaurav_image = face_recognition.load_image_file('images/samples/gaurav.jpeg')
gaurav_face_encodings = face_recognition.face_encodings(gaurav_image)[0]

divyata_image = face_recognition.load_image_file('images/samples/divyata.jpeg')
divyata_face_encodings = face_recognition.face_encodings(divyata_image)[0]

ayush_image = face_recognition.load_image_file('images/samples/ayush.jpeg')
ayush_face_encodings = face_recognition.face_encodings(ayush_image)[0]

nabodita_image = face_recognition.load_image_file('images/samples/nabodita.jpeg')
nabodita_face_encodings = face_recognition.face_encodings(nabodita_image)[0]

sushil_image = face_recognition.load_image_file('images/samples/sushil.jpeg')
sushil_face_encodings = face_recognition.face_encodings(sushil_image)[0]

tejas_image = face_recognition.load_image_file('images/samples/tejas.jpeg')
tejas_face_encodings = face_recognition.face_encodings(tejas_image)[0]

mayank_image = face_recognition.load_image_file('images/samples/mayank.jpeg')
mayank_face_encodings = face_recognition.face_encodings(mayank_image)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings, abhi_face_encodings,arpita_face_encodings,shahrukh_face_encodings,harshita_face_encodings,janhavi_face_encodings,rajiv_face_encodings,gaurav_face_encodings,divyata_face_encodings,ayush_face_encodings,nabodita_face_encodings,sushil_face_encodings,tejas_face_encodings,mayank_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump", "Abhilash","Arpita","Shahrukh","Harshita","Janhavi","Rajiv","Gaurav","Divyata","Ayush","Nabodita","Sushil","Tejas","Mayank"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []


def gen_frames():  
    #camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            #image_to_detect=cv2.imread(frame)
            all_face_locations=face_recognition.face_locations(frame,model="hog")
            for index,current_face_location in enumerate(all_face_locations):
                top_pos,right_pos,bottom_pos,left_pos = current_face_location
                print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
                #current_face_image=image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
                cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)

            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            
                
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def egen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            #image_to_detect=cv2.imread(frame)
            all_face_locations=face_recognition.face_locations(frame,model="hog")
            for index,current_face_location in enumerate(all_face_locations):
                top_pos,right_pos,bottom_pos,left_pos = current_face_location
                print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
                #current_face_image=image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
                current_face_image=frame[top_pos:bottom_pos,left_pos:right_pos]
                cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
                #convert to gray scale
                current_face_image=cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)
                #convert to 48*48
                current_face_image=cv2.resize(current_face_image,(48,48))
                #convert the PIL image into a 3d numpy array
                img_pixels = image.img_to_array(current_face_image)
                #expand the shape of an array into single row multiple columns
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                #pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
                img_pixels /= 255 
                #do prodiction using model, get the prediction values for all 7 expressions
                exp_predictions = face_exp_model.predict(img_pixels) 
                #find max indexed prediction value (0 till 7)
                max_index = np.argmax(exp_predictions[0])
                #get corresponding lable from emotions_label
                emotion_label = emotions_label[max_index]
                #display the name as text in the image
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            
                
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def rgen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            #get the current frame from the video stream as an image
            #ret,current_frame = webcam_video_stream.read()
            #resize the current frame to 1/4 size to proces faster
            current_frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            #detect all faces in the image
            #arguments are image,no_of_times_to_upsample, model
            all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
            #detect face encodings for all the faces detected
            all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
            #looping through the face locations
            #looping through the face locations and the face embeddings
            for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
                #splitting the tuple to get the four position values of current face
                top_pos,right_pos,bottom_pos,left_pos = current_face_location
                
                #change the position maginitude to fit the actual size video frame
                top_pos = top_pos*4
                right_pos = right_pos*4
                bottom_pos = bottom_pos*4
                left_pos = left_pos*4
                
                #find all the matches and get the list of matches
                all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
               
                #string to hold the label
                name_of_person = 'Unknown face'
                
                #check if the all_matches have at least one item
                #if yes, get the index number of face that is located in the first index of all_matches
                #get the name corresponding to the index number and save it in name_of_person
                if True in all_matches:
                    first_match_index = all_matches.index(True)
                    name_of_person = known_face_names[first_match_index]
                
                #draw rectangle around the face    
                cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
                
                #display the name as text in the image
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            
                
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
   
def agen_frames():  
    camera = cv2.VideoCapture(0)
    while True:
        
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            #get the current frame from the video stream as an image
            #ret,current_frame = webcam_video_stream.read()
            #resize the current frame to 1/4 size to proces faster
            current_frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            #detect all faces in the image
            #arguments are image,no_of_times_to_upsample, model
            all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
            
            #looping through the face locations
            for index,current_face_location in enumerate(all_face_locations):
                #splitting the tuple to get the four position values of current face
                top_pos,right_pos,bottom_pos,left_pos = current_face_location
                #change the position maginitude to fit the actual size video frame
                top_pos = top_pos*4
                right_pos = right_pos*4
                bottom_pos = bottom_pos*4
                left_pos = left_pos*4
                #printing the location of current face
                #print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
               
                #Extract the face from the frame, blur it, paste it back to the frame
                #slicing the current face from main image
                current_face_image = frame[top_pos:bottom_pos,left_pos:right_pos]
                
                #The ‘AGE_GENDER_MODEL_MEAN_VALUES’ calculated by using the numpy. mean()        
                AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                #create blob of current flace slice
                #params image, scale, (size), (mean),RBSwap)
                current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
                
                # Predicting Gender
                #declaring the labels
                gender_label_list = ['Male', 'Female']
                #declaring the file paths
                gender_protext = "dataset/gender_deploy.prototxt"
                gender_caffemodel = "dataset/gender_net.caffemodel"
                #creating the model
                gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
                #giving input to the model
                gender_cov_net.setInput(current_face_image_blob)
                #get the predictions from the model
                gender_predictions = gender_cov_net.forward()
                #find the max value of predictions index
                #pass index to label array and get the label text
                gender = gender_label_list[gender_predictions[0].argmax()]
                
                # Predicting Age
                #declaring the labels
                age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                #declaring the file paths
                age_protext = "dataset/age_deploy.prototxt"
                age_caffemodel = "dataset/age_net.caffemodel"
                #creating the model
                age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
                #giving input to the model
                age_cov_net.setInput(current_face_image_blob)
                #get the predictions from the model
                age_predictions = age_cov_net.forward()
                #find the max value of predictions index
                #pass index to label array and get the label text
                age = age_label_list[age_predictions[0].argmax()]
                      
                #draw rectangle around the face detected
                cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
                    
                #display the name as text in the image
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 0.5, (0,255,0),1)
            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            
                
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route("/expression/", methods=['POST'])
def emotion():
    camera.release()
    return render_template('expression.html');

@app.route("/genderage/", methods=['POST'])
def agegender():
    camera.release()
    return render_template('agegender.html');

@app.route("/recognition/", methods=['POST'])
def recognition():
    camera.release()
    return render_template('recognition.html');

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
                          
@app.route('/expression_feed')
def expression_feed():
    return Response(egen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/agegender_feed')
def agegender_feed():
    return Response(agen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_feed')
def recognition_feed():
    return Response(rgen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
if __name__=='__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()