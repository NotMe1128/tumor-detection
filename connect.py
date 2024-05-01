from flask import Flask, request, render_template, jsonify
import disease as disease_source
import tumor_model as tumor
import pickle
import cv2
import numpy as np
import os
from string import capwords
app = Flask(__name__)

with open("my_dict.pkl", "rb") as f:
    disease_list = pickle.load(f)

@app.route('/')
def hello(name=None):
    return render_template('def.html', name=name)

@app.route('/website')
def website():
    return render_template('website.html')

@app.route('/tumor-detect')
def tumor_detect():
    return render_template('tumor.html')

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename


@app.route('/tumor', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400
    try:
        # Process the file and generate output
        output=process_image()
        return output
        #return jsonify({'result': output})
    except Exception as e:
        # Handle exceptions and send back an error message
        return jsonify({'error': str(e)}), 500

def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if image:
        filename = "output.jpg"  # Set the desired output filename
        static_folder = os.path.join(os.getcwd(), "static")
        filepath = os.path.join(static_folder, filename)
        image.save(filepath)
   
        img = cv2.imread(filepath)
        img = cv2.resize(img, (150, 150))
        img_array = img.reshape(1, 150, 150, 3)
        res=tumor.pred_tumor(img_array)
        reason=" "
        if res=="no tumor":
            reason="Your brain MRI looks healthy!"
        else:
            reason="Seek Medical attention immediately!"
    
    return render_template('output.html', output=capwords(res), reason=reason)

@app.route('/def', methods=['POST','GET'])
def home():
    return render_template('def.html')

@app.route('/submitdata', methods=['POST'])
def process_symptoms():
    global description,precaution1,precaution2,precaution3,precaution4
    try:
        symptoms = request.json
        separator=','
        string_symptoms = separator.join(symptoms)
        output1 = "You selected: " + ", ".join(symptoms)
        predicted=disease_source.predictDisease(string_symptoms)
        output=f"{predicted}"
        disease1 = output
        description,precaution=disease_source.get_disease(disease1,disease_list)
        precaution1,precaution2,precaution3,precaution4=precaution[0],precaution[1],precaution[2],precaution[3]
        return jsonify(output=output)
    except Exception as e:
        app.logger.error(e)
        return jsonify(msg="Sorry, The inputs you selected did not result in any predicted disease. Please make sure to select proper symptoms."), 500
        print("error sent")

@app.route('/results')
def results():
    # get all the parameters from the query string
    output = request.args.get('output')
    
    
    # pass all the parameters to the template
    return render_template('result.html', output=output, description=description, precaution1=precaution1, precaution2=precaution2, precaution3=precaution3, precaution4=precaution4)

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(msg="Sorry, The inputs you selected did not result in any predicted disease. Please make sure to select proper symptoms. "), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500)


