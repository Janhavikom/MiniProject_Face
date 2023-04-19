# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:22:38 2023

@author: ARPITA
"""

from flask import Flask, render_template, Response
import cv2
import face_recognition
app=Flask(__name__)
camera = cv2.VideoCapture(0)


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
            
                
        return(b'--frame\r\n'
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