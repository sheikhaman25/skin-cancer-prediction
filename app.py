# Importing Essential Libraries
import numpy as np
from keras.applications import mobilenet
from tensorflow.keras.preprocessing import image
from flask import Flask, request, redirect, url_for, render_template
from tensorflow.keras.models import load_model as model_load
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

# Initialising Flask 
app = Flask(__name__)

# Path To The Model
model_path = 'skin_mnet_adam.h5'

# Defining Top 2 & Top 3 Accuracy
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 3)

# Defining Metrics
custom_objects = {'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy}

# Loading Model
model = model_load(model_path, custom_objects =  custom_objects)
model._make_predict_function()
print('Model Loaded. Ready To Go!')

# Function To Preprocess Image
def prepare_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    prc_img = mobilenet.preprocess_input(img_array)
    return (prc_img)

def convert_result(inp):
    if inp == 0:
        return 'Actinic Keratoses'
    elif inp == 1:
        return 'Basal Cell Carcinoma'
    elif inp == 2:
        return 'Benign Keratosis-Like Lesions'
    elif inp == 3:
        return 'Dermatofibroma'
    elif inp == 4:
        return 'Melanoma'
    elif inp == 5:
        return 'Melanocytic Nevi '
    elif inp == 6:
        return 'Vascular Lesions '
    
# Function To Predict On Uploaded Image 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        pro_img = prepare_image(file)
        predict = model.predict(pro_img)
        result = np.argmax(predict, axis = -1)
        res_prob = predict[0,result]
        result = convert_result(result)
    return render_template('index.html', result = result, res_prob = res_prob)

# Function To Load Main Page
@app.route('/')
def main_page():
    return render_template ('index.html')
      
# Starting The Application
if __name__ == '__main__':
    app.run()
