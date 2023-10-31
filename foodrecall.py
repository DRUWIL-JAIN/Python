import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
import requests
import json
from object_detector import *

food_name = ''

def predict_image(filename, model):
    img_array = np.uint8(filename[0] * 255.0)  # Convert back to uint8 range [0, 255]
    img_ = Image.fromarray(img_array).convert("RGB")
    img_ = img_.resize((299, 299))
    img_array = np.array(img_).astype('float32')
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)

    index = np.argmax(prediction)

    plt.title("Prediction - {}".format(category[index][1]))
    plt.imshow(img_array)
    global food_name
    food_name = category[index][1]



def predict_dir(filedir, model):
    cols = 5
    pos = 0
    images = []
    total_images = len(os.listdir(filedir))
    rows = total_images // cols + 1

    true = filedir.split('/')[-1]

    fig = plt.figure(1, figsize=(25, 25))

    for i in sorted(os.listdir(filedir)):
        images.append(os.path.join(filedir, i))

    for subplot, imggg in enumerate(images):
        img_ = Image.open(imggg).convert("RGB")
        img_ = img_.resize((299, 299))
        img_array = np.array(img_)

        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)

        pred = category.get(index)[0]
        if pred == true:
            pos += 1

        fig = plt.subplot(rows, cols, subplot + 1)
        fig.set_title(category.get(index)[1], pad=10, size=18)
        plt.imshow(img_array)

    acc = pos / total_images
    print("Accuracy of Test : {:.2f} ({pos}/{total})".format(acc, pos=pos, total=total_images))
    plt.tight_layout()


# Load the saved model
model = tf.keras.models.load_model('C:/Users/soham/Documents/MITAOE 24/TY Comp 22-23/Major Project/img_classification.h5')

# Define the food categories
category = {
    0: ['burger', 'Burger'], 1: ['butter_naan', 'Butter Naan'], 2: ['chai', 'Chai'],
    3: ['chapati', 'Chapati'], 4: ['chole_bhature', 'Chole Bhature'], 5: ['dal_makhani', 'Dal Makhani'],
    6: ['dhokla', 'Dhokla'], 7: ['fried_rice', 'Fried Rice'], 8: ['idli', 'Idli'], 9: ['jalegi', 'Jalebi'],
    10: ['kathi_rolls', 'Kaathi Rolls'], 11: ['kadai_paneer', 'Kadai Paneer'], 12: ['kulfi', 'Kulfi'],
    13: ['masala_dosa', 'Masala Dosa'], 14: ['momos', 'Momos'], 15: ['paani_puri', 'Paani Puri'],
    16: ['pakode', 'Pakode'], 17: ['pav_bhaji', 'Pav Bhaji'], 18: ['pizza', 'Pizza'], 19: ['samosa', 'Samosa']
}

# Set up the Streamlit app
st.title("Food Diet Recall App")
st.write("Upload the food image:")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Make predictions when an image is uploaded
if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((299, 299))  # Resize the image to match the input size of the model
    image_array = np.array(image).astype('float32')  # Convert image to float32 array
    image_processed = np.expand_dims(image_array, axis=0)  # Add an extra dimension
    image_processed /= 255.0  # Normalize the image


    # Make a prediction
    predict_image(image_processed, model)

    # Display the predicted class
    st.write("Predicted Category:", food_name)

query = food_name


# query = 'banana'
api_url = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(query)
response = requests.get(api_url, headers={'X-Api-Key': 'fZ7ykPHove7aFjEsSjRjKQ==Rfjj7NAyM2PIie6j'})
if response.status_code == requests.codes.ok:
    nutri_info = response.text
#     st.write(nutri_info)
else:
    st.write("Error:", response.status_code, response.text)

data = json.loads(response.text)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Transpose the DataFrame to display in vertical format
df_vertical = df.transpose()

# Display the transposed DataFrame as a table in Streamlit
st.table(df_vertical)


st.write("Upload the top view and side view images of the food item.")

# Create file uploaders for top view and side view images
uploaded_top_view = st.file_uploader("Upload the top view image", type=["jpg", "jpeg", "png"])
uploaded_side_view = st.file_uploader("Upload the side view image", type=["jpg", "jpeg", "png"])

# Make predictions when both images are uploaded
if uploaded_top_view and uploaded_side_view:
    # Read the uploaded images
    top_view_image = Image.open(uploaded_top_view)
    side_view_image = Image.open(uploaded_side_view)

    # Convert top view image to OpenCV format
    top_view_cv_image = np.array(top_view_image.convert("RGB"))
    top_view_cv_image = top_view_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

    # Perform object detection and measurement
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Load object detector
    detector = HomogeneousBgDetector()

    # Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(top_view_cv_image, aruco_dict, parameters=parameters)

    if corners:
        # Draw polygon around the marker
        int_corners = np.intp(corners)
        cv2.polylines(top_view_cv_image, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(int_corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(top_view_cv_image)

        # Iterate over the detected objects
        for cnt in contours:
            # Get rectangle
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get width and height of the objects by applying the ratio pixel/cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Getting the center point of the object
            cv2.circle(top_view_cv_image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw polygon
            cv2.polylines(top_view_cv_image, [box], True, (255, 0, 0), 2)
            cv2.putText(top_view_cv_image, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(top_view_cv_image, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Display the modified top view image with object measurements
        st.image(top_view_cv_image, caption='Top View with Measurements', use_column_width=True)

        object_area = round(object_height, 1) * round(object_width, 1)
        st.write("The length of the item is: {} cm".format(round(object_height,1)))
        st.write("The breadth of the item is: {} cm".format(round(object_width,1)))
        st.write("The area of the item is: {} cm^2".format(object_area))
    

        # Convert side view image to OpenCV format
        side_view_cv_image = np.array(side_view_image.convert("RGB"))
        side_view_cv_image = side_view_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

        # Perform object detection and measurement for side view image
        side_view_contours = detector.detect_objects(side_view_cv_image)

        # Iterate over the detected objects in side view image
        for cnt in side_view_contours:
            # Get rectangle
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get width and height of the objects by applying the ratio pixel/cm
            object_width1 = w / pixel_cm_ratio
            object_height1 = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Getting the center point of the object
            cv2.circle(side_view_cv_image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw polygon
            cv2.polylines(side_view_cv_image, [box], True, (255, 0, 0), 2)
            cv2.putText(side_view_cv_image, "Width {} cm".format(round(object_width1, 1)), (int(x - 100), int(y - 20)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            cv2.putText(side_view_cv_image, "Height {} cm".format(round(object_height1, 1)), (int(x - 100), int(y + 15)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Display the modified side view image with object measurements
        st.image(side_view_cv_image, caption='Side View with Measurements', use_column_width=True)

        # Calculate and display the area of the object in the side view
        side_view_object_area = round(object_height1, 1) * round(object_width1, 1)
        st.write("The height of the item is: {} cm".format(round(object_height1, 1)))
        st.write("The width of the item is: {} cm".format(round(object_width1, 1)))
        st.write("The area of the item in the side view is: {} cm^2".format(side_view_object_area))

        volume = round((object_height*object_width*object_height1),1)
        st.write("The volume of the food is: {} cm^3".format(volume))
        
# Get the density from the user
density = st.number_input("Enter the density (g/cm³):", min_value=0.0)

if density > 0:
    # Calculate the mass of the object
    mass = volume*density
    st.write("Amount of food: {} g".format(round(mass,1)))
    desired_weight = round(mass,1)
    serving_size = 100
    proportion = desired_weight/serving_size

    
    data1 = json.loads(response.text)
    for row in data1:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                row[key] = round(value * proportion, 1)

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data1)

    # Round the values in the DataFrame to one decimal point
    df = df.round(1)

    # Transpose the DataFrame to display in vertical format
    df_vertical = df.transpose()

    # Display the transposed DataFrame as a table in Streamlit
    st.table(df_vertical)