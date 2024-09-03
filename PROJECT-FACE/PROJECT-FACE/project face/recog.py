import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox

# Load the trained model and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/model_output/trainer.yml')

# Load the name-to-ID mapping
name_to_id = {}
with open('C:/model_output/names.txt', 'r') as f:
    for line in f:
        id, name = line.strip().split(':')
        name_to_id[int(id)] = name

# Initialize face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Function to map ID to name
def get_name(id):
    return name_to_id.get(id, "Unknown")


# Path to save the attendance Excel file
attendance_file = 'C:/model_output/attendance.xlsx'

# Create an empty DataFrame for attendance if the file doesn't exist
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    df.to_excel(attendance_file, index=False)


# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Load existing attendance data
    df = pd.read_excel(attendance_file)

    # Check if the user has already been marked present for the day
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        # Create a new record
        new_record = pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})

        # Append the new record to the existing DataFrame using pd.concat
        df = pd.concat([df, new_record], ignore_index=True)

        # Save to Excel
        df.to_excel(attendance_file, index=False)
        print(f'Attendance marked for {name} at {time_str} on {date_str}')

        # Display popup notification
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("Attendance Marked", f"Attendance marked for {name} at {time_str} on {date_str}")
        root.destroy()


# Initialize webcam
cam = cv2.VideoCapture(0)

# Main loop for face recognition and attendance
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_detected = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    highest_confidence = 0
    recognized_name = None

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        if confidence < 100:
            name = get_name(id)
            confidence_score = round(100 - confidence)
            confidence_str = f"  {confidence_score}%"

            # Check if this is the highest confidence detected
            if confidence_score > highest_confidence:
                highest_confidence = confidence_score
                recognized_name = name
        else:
            confidence_str = f"  {round(100 - confidence)}%"

        cv2.putText(img, name, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(img, str(confidence_str), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # If a face was recognized, mark attendance for the person with the highest confidence score
    if recognized_name:
        mark_attendance(recognized_name)
        recognized_name = None  # Reset to prevent multiple markings

    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'ESC' to quit
        break

cam.release()
cv2.destroyAllWindows()
