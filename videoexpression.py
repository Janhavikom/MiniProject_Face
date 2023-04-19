# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 19:03:32 2023

@author: ARPITA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:40:50 2023

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
   

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()