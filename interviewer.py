import cv2
import time
import pyttsx3
import pymysql
import threading
import numpy as np
import speech_recognition as sr
import google.generativeai as genai
from keras.models import model_from_json
from flask import Flask, request, jsonify, flash, render_template, redirect, url_for, session, Response


app = Flask(__name__)
app.secret_key = 'your_secret_key'

recognizer = sr.Recognizer()

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def sqlconnection():
    try:
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="Bu@#9063808032",
            database="ap",
        )
        return connection
    except pymysql.MySQLError as e:
        print(f"Database connection error: {e}")
        return None
@app.route('/login', methods=['GET', 'POST'])
def verify():
    if request.method == 'POST':
        email = request.form.get('username')
        password = request.form.get('password')

        if not email or not password:
            flash("Please fill out all fields.")
            return redirect(url_for('verify'))
        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "SELECT password FROM login WHERE email = %s"
                    cursor.execute(query, (email,))
                    result = cursor.fetchone()
            finally:
                conn.close()

            if result is None or result[0] != password:
                flash("Invalid credentials. Please try again.")
                return redirect(url_for('verify'))

            flash("Login successful!")
            return redirect(url_for('home'))
        else:
            flash("Error from Our side We are sorry.")
            return redirect(url_for('verify'))

    return render_template('index.html')

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        name = request.form.get('fullname')
        email = request.form.get('email')
        password = request.form.get('password')

        if not name or not email or not password:
            flash("Please fill all the details")
            return redirect(url_for('signup'))

        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "INSERT INTO login (email, password) VALUES (%s, %s)"
                    cursor.execute(query, (email, password))
                    conn.commit()
            finally:
                conn.close()

            flash("Signup Successful!")
            return redirect(url_for('index'))
        else:
            flash("Error from Our side We are trying to fix it.....")
            return redirect(url_for('signup'))

    return render_template('signup.html')

@app.route('/forgetpassword', methods=['POST', 'GET'])
def changepass():
    if request.method == 'POST':
        email = request.form.get('email')
        changedpassword = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not email or not changedpassword or not confirm_password:
            flash("Please fill all the details")
            return redirect(url_for('changepass'))

        if changedpassword != confirm_password:
            flash("Passwords do not match")
            return redirect(url_for('changepass'))

        conn = sqlconnection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    query = "UPDATE login SET password = %s WHERE email = %s"
                    cursor.execute(query, (changedpassword, email))
                    conn.commit()
            finally:
                conn.close()

            flash("Successfully changed the password!")
            return redirect(url_for('index'))
        else:
            flash("Database connection failed.")
            return redirect(url_for('changepass'))

    return render_template('forgotPassword.html')


company,job_role,candidate_name = "","",""
@app.route('/home', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        global company, job_role, candidate_name
        company = request.form.get('company')
        job_role = request.form.get('job-role')
        candidate_name = request.form.get('candidate-name')

        if not company or not job_role or not candidate_name:
            flash("Please fill all the details")
            return redirect(url_for('home'))
        return render_template('interviewer.html')
        
    return render_template('home.html')

question = "" 
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
    global question
    genai.configure(api_key=("AIzaSyBaJaU65Ev2eRTX2M_rMVfwLcv3im5VMoY "))  # Load from environment variable  AIzaSyDG6lsSWErYYs5K0uwkIIYUbZmt5XfSQcc
    gmodel = genai.GenerativeModel('gemini-pro')
    time.sleep(2)
    response = gmodel.generate_content(f"hey my name is {candidate_name}and im preparing for an hr interview round for the role of {job_role} at {company} so, genrate one interview question which can be ansked in that interview") 
    question = response.text.strip()
    pyttsx3.speak(question)

answer_given = ""
def answergiven():
    global answer_given
    with sr.Microphone() as source:
        print("Listening for answer...")
        audio = recognizer.listen(source)  # Capture audio input
        try:
            answer_given = recognizer.recognize_google(audio, language='en-us')
            return answer_given
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return "Error: Could not understand the audio"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "Error: Could not request results from Google"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/feedback')
def feedback():
    return render_template('fedback.html')

@app.route('/interviewer')
def interviewer_index():
    return render_template('interviewer.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(interviewer(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video feed: {e}")
        return "Error in video feed"

@app.route('/get_question')
def get_question():
    global question
    if question:  
        return jsonify({"question": question})
    else:
        return jsonify({"question": "No question available yet."})

@app.route('/answergiven')
def answer():
    return jsonify({"Answer Given": answer_given })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
