from flask import Flask, request, render_template, jsonify
import disease as disease_source
import tumor_model as tumor
import pickle
import cv2
import numpy as np
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
def upload_file():
    print("file received")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    try:
        # Process the file and generate output
        output = process_image(file)
        return jsonify({'result': output})
    except Exception as e:
        # Handle exceptions and send back an error message
        return jsonify({'error': str(e)}), 500

def process_image(file):
    img = cv2.imread(file)
    img = cv2.resize(img,(150,150))
    img_array = np.array(img)
    img_array = img_array.reshape(1,150,150,3)
    print((img_array).shape())
    res=tumor.pred_tumor(img_array)
    return res


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


