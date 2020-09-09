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
one_class_model_path = 'mnet_adam.h5'
detect_model_path = 'skin_mnet_adam.h5'
stage_model_path = 'stages_mnet_adam.h5'

# Defining Top 2 & Top 3 Accuracy
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 2)

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 3)

# Defining Metrics
custom_objects = {'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy}

# Loading Model
one_class_model = model_load(one_class_model_path)
one_class_model._make_predict_function()

skin_model = model_load(detect_model_path, custom_objects =  custom_objects)
skin_model._make_predict_function()

stage_model = model_load(stage_model_path)
stage_model._make_predict_function()

print('Model Loaded. Ready To Go!')

# Function To Preprocess Image
def prepare_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    prc_img = mobilenet.preprocess_input(img_array)
    return (prc_img)

# Function To Convert To String
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

# Function To Predict Severity
def stage_check(inp, pro_img):
    stage_pred = stage_model.predict(pro_img)
    stage_res = np.argmax(stage_pred, axis = -1)
    if stage_res == 0:
        return 'Stage Zero'
    elif stage_res == 1:
        return 'Stage One'
    elif stage_res == 2:
        return 'Stage Two'
    elif stage_res == 3:
        return 'Stage Three'
    elif stage_res == 4:
        return 'Stage Four'        

# Function To Predict On Correct Image
def correct_predict(pro_img):
    predict = skin_model.predict(pro_img)
    result = np.argmax(predict, axis = -1)
    res_prob = predict[0,result]
        
    if result == 4:
        stage_res = stage_check(result,pro_img)
    else:
        stage_res = 'Stage Classification Not Applicable'
            
    result = convert_result(result)
    
    return result,res_prob,stage_res
        
# Function To Predict On Uploaded Image 
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        pro_img = prepare_image(file)
        class_pred = one_class_model.predict(pro_img)
        class_result = np.argmax(class_pred, axis = -1)
        class_prob = class_pred[0,class_result]
        
        if class_result == 1:
            result,res_prob, stage_res = correct_predict(pro_img)
            
        elif class_result == 0:
            if class_prob < 1.0:
                result,res_prob, stage_res = correct_predict(pro_img)
                
            elif class_prob == 1.0:
                result = 'Upload Valid Image'
                res_prob = 'Not Applicable'
                stage_res = 'Not Applicable'
                        
    return render_template('index.html', result = result, res_prob = res_prob, stg_res = stage_res)

# Function To Load Main Page
@app.route('/')
def main_page():
    return render_template ('index.html')
      
# Starting The Application
if __name__ == '__main__':
    app.run()
