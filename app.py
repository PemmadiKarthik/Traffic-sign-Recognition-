from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model("gtsrb_traffic_sign_model.h5")

# Define class labels for GTSRB dataset
class_labels = {
    0: "Speed Limit (20km/h)",
    1: "Speed Limit (30km/h)",
    2: "Speed Limit (50km/h)",
    3: "Speed Limit (60km/h)",
    4: "Speed Limit (70km/h)",
    5: "Speed Limit (80km/h)",
    6: "End of Speed Limit (80km/h)",
    7: "Speed Limit (100km/h)",
    8: "Speed Limit (120km/h)",
    9: "No Passing",
    10: "No Passing for Vehicles Over 3.5 Metric Tons",
    11: "Right-of-Way at the Next Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles Over 3.5 Metric Tons Prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve to the Left",
    20: "Dangerous Curve to the Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on the Right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Speed and Passing Limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout Mandatory",
    41: "End of No Passing",
    42: "End of No Passing by Vehicles Over 3.5 Metric Tons"
}

# Function to predict traffic sign
def predict_traffic_sign(image_path):
    # Preprocess image
    img = load_img(image_path, target_size=(32, 32))  # Resize to model's input size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels.get(predicted_class, "Unknown Sign")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            return "No file uploaded!"
        file = request.files["file"]
        if file.filename == "":
            return "No file selected!"
        if file:
            # Save the file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Predict the traffic sign
            prediction = predict_traffic_sign(file_path)

            return render_template("index.html", prediction=prediction, image_path=file_path)

    return render_template("index.html", prediction=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
