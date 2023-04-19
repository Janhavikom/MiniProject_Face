from flask import Flask, render_template, Response
import cv2

app=Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frames():  
    #camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            

            ret, buffer = cv2.imencode('.jpeg', frame)
            frame = buffer.tobytes()
            #image_to_detect=cv2.imread(frame)
            
        yield (b'--frame\r\n'
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


 <img src="{{ url_for('video_feed') }}" width="50%" >
 @app.route("/forward/", methods=['POST'])
 def move_forward():
     camera.release()
     return render_template('expression.html');
 @app.route("/video_feed")
 def video_feed():
     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

 @app.route('/expression_feed')
 def evideo_feed():
     return Response(egen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')