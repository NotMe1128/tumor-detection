import pickle
import numpy as np

with open('model.pkl', 'rb') as f:
  final_rf_model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('symptom.pkl', 'rb') as f:
  symptoms = pickle.load(f)


symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

def predictDisease(symptoms):
	symptoms = symptoms.split(",")
	
	# creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	final_prediction =rf_prediction
	print(final_prediction)
	return final_prediction

def get_disease(disease_name,disease_list):
    disease_name = disease_name.lower()
    for disease in disease_list:
        for key in disease:
            if key.lower() == disease_name:
                description, precaution = disease[key]
                return description, precaution
    return None, None