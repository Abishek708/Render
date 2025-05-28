from flask import Flask, jsonify, render_template, request, session, redirect
import pickle
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import cv2
import datetime
import json
from googletrans import Translator
import gdown

app = Flask(__name__)

app.secret_key = os.urandom(24)  
print("Secret Key:", app.secret_key)

@app.route('/', methods= ['GET'])
def hello():
    return render_template('index.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():    
    return render_template('contact.html')

@app.route('/Explore.html')
def Explore():
    return render_template('Explore.html')

@app.route('/Plant.html')
def Plant():
    return render_template('Plant.html')

@app.route('/Crop.html')
def Crop():
    return render_template('Crop.html')

translator = Translator()

def translate_to_tamil(text):
    try:
        result = translator.translate(text, src='en', dest='ta')
        return result.text
    except Exception as e:
        return f"Translation error: {str(e)}"

import os

model_path = "model.h5"
file_id = "1hqk1BZACH7PCM1IgwZrDqpXz-XQieV_R"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(url, model_path, quiet=False)


model = keras.models.load_model(model_path)  # Or 'model.keras'


with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

@app.route('/', methods=['post'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    target_size = (224, 224)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)

    img_array = np.array(img)
    img_array_normalized = img_array.astype('float32') / 255.
    img_batch = np.expand_dims(img_array_normalized, axis=0)

    # Green pixel analysis
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]

    green_dominant = (green > red) & (green > blue)
    green_ratio = np.sum(green_dominant) / (224 * 224)

    if green_ratio > 0.4:
        prediction = model.predict(img_batch)
        confidence = np.max(prediction)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_indices[predicted_class_index]

        session['classification'] = predicted_class_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        session['c_score'] = float(confidence)
        return render_template('Plant.html', prediction=predicted_class_name, confidence=round(confidence*100, 2))
    else:
        return "❌ The image doesn't seem to be a plant. Please upload a clear plant leaf image."



GROWTH_DATA_PATH = 'growth_data.json'

def calculate_green_area(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = (25, 40, 40)
    upper_green = (90, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_area = cv2.countNonZero(mask)
    return green_area

def determine_stage(area):
    if area < 5000:
        return "🌱 Seedling"
    elif area < 15000:
        return "🌿 Vegetative"
    else:
        return "🌸 Mature"

def observation_value(data):
    observation = "Not enough data to generate observation."
    if len(data) >= 2:
            latest = data[-1]["area"]
            previous = data[-2]["area"]
            diff = latest - previous

            if diff > 500:
                observation = "📈 Plant is growing well."
            elif diff > 100:
                observation = "⚠️ Moderate growth."
            elif diff > 0:
                observation = "⚠️ Growth slowing down."
            else:
                observation = "🛑 No significant growth detected."
                
    return observation  


@app.route('/growth.html', methods=['GET', 'POST'])
def growth():
    growth_data = []
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        green_area = calculate_green_area(image_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

         # Load previous data
        if os.path.exists(GROWTH_DATA_PATH):
            with open(GROWTH_DATA_PATH, 'r') as f:
                growth_data = json.load(f)

        if len(growth_data) > 0:
            previous_area = growth_data[-1]["area"]
            growth_rate = round(((green_area - previous_area) / previous_area) * 100, 2)
        else:
            growth_rate = 0.0

        stage = determine_stage(green_area)
        temp_data = growth_data + [{
            "time": timestamp,
            "area": green_area,
            "growth_rate": growth_rate,
            "stage": stage
        }]

        observation = observation_value(temp_data)

       

        # Append new data
        growth_data.append({
            "time": timestamp,
            "area": green_area,
            "growth_rate": growth_rate,
            "stage": stage,
            "observation":observation
        })

        # Save updated data
        with open(GROWTH_DATA_PATH, 'w') as f:
            json.dump(growth_data, f)

    elif os.path.exists(GROWTH_DATA_PATH):
            with open(GROWTH_DATA_PATH, 'r') as f:
                growth_data = json.load(f)
        

    

    return render_template('growth.html', growth_data=growth_data)

@app.route('/clear-growth')
def clear_growth():
    if os.path.exists(GROWTH_DATA_PATH):
        os.remove(GROWTH_DATA_PATH)
        return redirect('/growth.html')

def detect_stage(confidence):
    if confidence >= 0.90:
        return "Severe"
    elif confidence >= 0.70:
        return "Moderate"
    else:
        return "Mild"

@app.route('/translate_diagnosis', methods=['POST'])
def translate_diagnosis():
    data = request.get_json()
    disease = data.get('disease', '')
    stage = data.get('stage', '')
    cures = data.get('cures', {})

    disease_ta = translate_to_tamil(disease.replace("_", " "))
    stage_ta = translate_to_tamil(stage)

    cures_ta = {}
    for category, items in cures.items():
        if isinstance(items, list):
            translated_items = [translate_to_tamil(item) for item in items if item]
            cures_ta[translate_to_tamil(category)] = translated_items
        elif isinstance(items, str):
            cures_ta[translate_to_tamil(category)] = [translate_to_tamil(items)]

    return jsonify({
        "disease_ta": disease_ta,
        "stage_ta": stage_ta,
        "cures_ta": cures_ta
    })

@app.route('/describe.html')
def describe():
    # Get classification from session
    diseases = session.get('classification')
    confidence = session.get('c_score', 0.0)

    # Detect stage
    stage = detect_stage(confidence)

    # Load cure recommendations
    with open('cure_recommendations.json', 'r') as f:
        cure_data = json.load(f)

    # Get cures based on stage
    if diseases in cure_data and stage.lower() in cure_data[diseases]:
        cures = cure_data[diseases][stage.lower()]
    else:
        cures = ["No stage-specific cure found. Try general practices."]
    
    return render_template("describe.html",diseases= diseases, confidence=round(confidence,2), stage=stage, cures=cures)

    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
