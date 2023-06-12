# Import the required modules
import time
import cv2
import cvzone
import numpy as np
import joblib
import os
import utils
import mediapipe as mp
import math
from flask import Flask, jsonify, render_template, Response, request
from utils import animation_view, sign_to_speech,mediapipe
# Initialize the Flask application
app = Flask(__name__)
# Initialize the YOLO object detection model

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


model = joblib.load('test.sav')

# Open the webcam

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Define the text-to-sign route
@app.route('/text-to-sign', methods=['POST'])
def signToText():
    # Get the JSON data from the request
    json_data = request.get_json()
    value = json_data.get('sen')
    return jsonify(animation_view.animation_view_api('POST',value))

# About us page
@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

# privacy - policy us page
@app.route('/privacy-policy')



def privacy_policy():
    return render_template('privacy-policy.html')

# Contact us page
@app.route('/contact-us')
def contact_us():
    return render_template('contact-us.html')

# Contact us page
@app.route('/api-documentaion')
def api_documentation():
    return render_template('api-documentation.html')

# Define the video to sign route
@app.route('/video-to-sign')
def video_to_sign():
    return render_template('sign-to-text.html')
def gen():
    # Capture an image from the webcam
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    DATA_PATH = os.path.join("C:/Users/princ/Desktop/ActionDetectionforSignLanguage-main/ActionDetectionforSignLanguage-main/CollectedData/")
    data_label = []
    for filename in os.listdir(DATA_PATH):
        data_label.append(filename)
    data_label = np.array(data_label)
    label_map = {label: num for num, label in enumerate(data_label)}



    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, image = cap.read()
            if not ret:
                break


            # Make detections
            image, results = mediapipe.mediapipe_detection(image, holistic)

            # Draw landmarks
            mediapipe.draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = mediapipe.extract_keypoints(results)
            keypoints = np.array(keypoints)
            keypoints.resize((1692,), refcheck=False)

            sequence.append(keypoints)
            sequence = sequence[-20:]
            print(np.array(sequence).shape)


            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                num = np.argmax(res)
                print(num)

                print(list(label_map.keys())[list(label_map.values()).index(num)])
                text = 'Ouput: ' + list(label_map.keys())[list(label_map.values()).index(num)]

                image = cv2.putText(image, text, (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                                    lineType=cv2.LINE_AA)


        # Return the image as a video stream
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# Define the video to sign route
@app.route('/text-to-sign')
def text_to_sign():
    return render_template('text-to-sign.html')

# Define the showservices route
@app.route('/show-services')
def show_services():
    return render_template('show-services.html')

# Define the comming soon route
@app.route('/comming-soon')
def comming_soon():
    return render_template('comming-soon.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run()