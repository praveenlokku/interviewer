import cv2
import time
import pyttsx3
import threading
import numpy as np
import speech_recognition as sr
import google.generativeai as genai
from keras.models import model_from_json
from flask import Flask, Response, render_template

app = Flask(__name__)

recognizer = sr.Recognizer()

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def interviewer():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not access the camera")
        return

    threading.Thread(target=ask_question, daemon=True).start()

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Failed to read frame from webcam")
            break
        
        frame = cv2.flip(frame, 1)  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48)) 
            img = extract_features(face) 
            pred = model.predict(img)  
            prediction_label = labels[pred.argmax()] 
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x-10, y-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    webcam.release()

def ask_question():
    """Function to ask a question using text-to-speech and listen for a response."""
    genai.configure(api_key='AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc') 
    gmodel = genai.GenerativeModel('gemini-pro')
    time.sleep(2)
    response = gmodel.generate_content(f"Generate an behaviroual interview question for a software engineer role.") 
    cleaned_question = response.text.strip()
    pyttsx3.speak(cleaned_question)
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

        try:
            
            response = recognizer.recognize_google(audio, language='en-US')
            print(f"User said: {response}")
            return response
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
            return "I did not understand."
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return "Request error."

@app.route('/')
def index():
    return render_template('interviewer.html')
@app.route('/answergiven')
def answer():
    return jsonify({"Answer Given": answer_given })

@app.route('/get_question')
def get_question():
    global question
    if question:  
        return jsonify({"question": question})
    else:
        return jsonify({"question": "No question available yet."})


@app.route('/video_feed')
def video_feed():
    return Response(interviewer(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000, debug=False)
