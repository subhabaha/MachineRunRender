#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_option_menu import option_menu
import pytz
from datetime import date
from email.message import EmailMessage
import ssl
import smtplib

def get_log(model, video): 
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 15
    machine_status = None
    status_chk = 0
    status1 = ""
    status = ""
    IST = pytz.timezone('Asia/Kolkata') 
    
    # Open a connection to the webcam (0 represents the default webcam)
    video_path = video
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 1  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Resize the frame to match the input size of the model
        frame = cv2.resize(frame, (512, 512))

        # Preprocess the frame
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        prediction = model.predict(img_array, verbose=0)

        # Classify as running if the prediction is above 0.5 (you can adjust this threshold)
        if prediction > 0.5:
            running_frames += 1
            not_running_frames = 0
        else:
            running_frames = 0
            not_running_frames += 1

        # Capture a screenshot at the specified interval
        if frame_count % screenshot_frames == 0:
            datetime_ist = datetime.now(IST)
            current_time = datetime_ist.strftime("%H:%M:%S")
            today = date.today()
            current_day = today.strftime("%A")
            screenshot_filename = f"screenshot_{current_time}.png_{current_day}"
            cv2.imwrite(screenshot_filename, frame)
            if prediction > 0.5:
                status = "Running"
            else:
                status = "Not Running"
            st.write(f"{current_time} - Screenshot captured: {screenshot_filename}, Machine {status}")
            
            if status == status1:
                status_chk += 1
            else:
                status_chk = 0
            status1 = status


        # Check for consecutive frames and update machine status
        if status_chk >= consecutive_frames_threshold:
            print(f"Machine Status: {status}")
            status_chk = 0

        frame_count += 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()

def get_machine_status(model, video): 
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 15
    machine_status = None
    status_chk = 0
    status1 = ""
    status = ""
    chk_time = 0
    chk_time_1 = 100
    status_temp = "Loading ..."
    clear = st.empty()
    count = 0
    i = 0
    IST = pytz.timezone('Asia/Kolkata') 
    em = EmailMessage()

    sender_email = "chandrapaulcs2001@gmail.com"
    sender_password = "lbjc utkq kawe fcvd"
    #recipient = "chandrapauldas01@gmail.com"
    recipient = "chandrapauldas01@gmail.com"
    subject = f"⚠️ Warning! Machine Status changed ⚠️"
    em['From'] = sender_email
    em['To'] = recipient
    em['Subject'] = subject
    context = ssl.create_default_context()
    
    # Open a connection to the webcam (0 represents the default webcam)
    video_path = video
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 1  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Resize the frame to match the input size of the model
        frame = cv2.resize(frame, (512, 512))

        # Preprocess the frame
        img_array = image.img_to_array(frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        prediction = model.predict(img_array, verbose=0)

        # Classify as running if the prediction is above 0.5 (you can adjust this threshold)
        if prediction > 0.5:
            running_frames += 1
            not_running_frames = 0
        else:
            running_frames = 0
            not_running_frames += 1

        # Capture a screenshot at the specified interval
        if frame_count % screenshot_frames == 0:
            datetime_ist = datetime.now(IST)
            current_time = datetime_ist.strftime("%H:%M:%S")
            today = date.today()
            current_day = today.strftime("%A")
            screenshot_filename = f"screenshot_{current_time}.png"
            cv2.imwrite(screenshot_filename, frame)
            if prediction > 0.5:
                status = "Running"
            else:
                status = "Not Running"
            if status == status1:
                status_chk += 1
            else:
                status_chk = 0
            status1 = status

        if count == 0:
            with clear.container():
                st.write(f"Machine Status: {status_temp}")
            count = 1
        # Check for consecutive frames and update machine status
        if status_chk >= consecutive_frames_threshold:
            with clear.container():
                st.write(f"Machine Status: {status}")
                
        count = 1
        frame_count += 1

        if status == "Not Running":
            chk_time = int(current_time[3:5])
            if chk_time == chk_time_1 + 5 + i:
                st.write("SENT")
                message = f"The Machine was observed not running first at {chk_time_1_act} IST. It's been {5+i} minutues, and we have observed the machine is still not running. Current time is {current_time} IST."
                em.set_content(message)
                with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                    smtp.login(sender_email, sender_password)
                    smtp.sendmail(sender_email, recipient, em.as_string())
                i += 5
            if status_chk == 0:
                chk_time_1 = chk_time
                chk_time_1_act = current_time

    # Release the webcam capture object and close the OpenCV window
    cap.release()

# Set the layout
st.set_page_config(page_title="Machine Status App", page_icon="🤖", layout="wide")

# Main title
st.title("Machine Status Monitoring App")

# Load the saved model
model = tf.keras.models.load_model('machine_model_5jan10pm.h5')
video_path = "rtsp://admin:Admin@123@125.19.34.95:554/cam/realmonitor?channel=1&subtype=0"
#video_path = "sample video.mp4"

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Machine Status", "Machine runtime log"],
        icons = ["lightning-charge-fill", "list-columns"],
        default_index = 0)


if selected == "Machine Status":
    # Display the machine status
    st.subheader("Machine Status")
    # Call the function to get the machine status and log
    get_machine_status(model, video_path)

if selected == "Machine runtime log":
    # Display the machine status
    st.subheader("Machine runtime log")
    get_log(model, video_path)
