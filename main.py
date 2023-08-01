from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np
from flask import Flask, request, render_template,Response,jsonify


app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
current_emotion = ""
emotion_songs = {
    'Angry': [
        {'title': 'Song 1', 'cover_photo': 'cover10.jpg', 'audio_file': 'song10.mp3'},
        {'title': 'Song 2', 'cover_photo': 'cover11.jpg', 'audio_file': 'song11.mp3'},
        {'title': 'Song 3', 'cover_photo': 'cover12.jpg', 'audio_file': 'song12.mp3'}
    ],
    # 'Disguist': [
    #     {'title': 'Song 4', 'cover_photo': 'cover4.jpg', 'audio_file': 'song4.mp3'},
    #     {'title': 'Song 5', 'cover_photo': 'cover5.jpg', 'audio_file': 'song5.mp3'},
    #     {'title': 'Song 6', 'cover_photo': 'cover6.jpg', 'audio_file': 'song6.mp3'}
    # ],
    'Fear': [
        {'title': 'Song 1', 'cover_photo': 'cover7.jpg', 'audio_file': 'song7.mp3'},
        {'title': 'Song 2', 'cover_photo': 'cover8.jpg', 'audio_file': 'song8.mp3'},
        {'title': 'Song 3', 'cover_photo': 'cover9.jpg', 'audio_file': 'song9.mp3'}
    ],
    'Happy': [
        {'title': 'Song 1', 'cover_photo': 'Cover1.png', 'audio_file': 'song1.mp3'},
        {'title': 'Song 2', 'cover_photo': 'Cover2.png', 'audio_file': 'song2.mp3'},
        {'title': 'Song 3', 'cover_photo': 'Cover3.png', 'audio_file': 'song3.mp3'}
    ],
    'Sad': [
        {'title': 'Song 1', 'cover_photo': 'cover13.jpg', 'audio_file': 'song13.mp3'},
        {'title': 'Song 2', 'cover_photo': 'cover14.jpg', 'audio_file': 'song14.mp3'},
        {'title': 'Song 3', 'cover_photo': 'cover15.jpg', 'audio_file': 'song15.mp3'}
    ],
    'Surprise': [
        {'title': 'Song 1', 'cover_photo': 'cover16.jpg', 'audio_file': 'song16.mp3'},
        {'title': 'Song 2', 'cover_photo': 'cover17.jpg', 'audio_file': 'song17.mp3'},
        {'title': 'Song 3', 'cover_photo': 'cover18.jpg', 'audio_file': 'song18.mp3'}
    ],
    'Neutral': [
        {'title': 'Song 1', 'cover_photo': 'on and on.jpg', 'audio_file': 'On and On.mp3'},
        {'title': 'Song 2', 'cover_photo': 'cover20.jpg', 'audio_file': 'song20.mp3'},
        {'title': 'Song 3', 'cover_photo': 'cover21.jpg', 'audio_file': 'song21.mp3'}
    ]
}

def detect_emotion(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if  np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            global current_emotion
            current_emotion = label
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            
    return frame


def gen_frames():
    while True:
       success,frame=camera.read()   # read the camera frame
       if not success:
           break
       else:
           frame = detect_emotion(frame)
           ret, buffer = cv2.imencode('.jpg',frame )
           frame = buffer.tobytes()
           yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html',emotion=current_emotion)

@app.route('/get_emotion', methods=['GET'])
def get_emotion():
    global current_emotion
    global songs
    songs = emotion_songs.get(current_emotion, [])
    return jsonify({'emotion': current_emotion, 'songs': songs})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    
if __name__ == "__main__":
    app.run(debug=True)


