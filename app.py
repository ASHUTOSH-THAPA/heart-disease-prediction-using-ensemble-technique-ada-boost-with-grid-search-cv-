from flask import Flask, request,jsonify
import numpy as np 
import joblib
app=Flask(__name__)
feature_names = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
                 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
                 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']

model=joblib.load("/Users/ashutoshthapa/Documents/ai_health_prediction_models/heart_disease_model.pkl")
@app.route('/')
def home():
    return "jaldi project bano guyzz"
@app.route('/predict',methods=['POST','GET'])
def predict():
        data=request.get_json(force=True)
        features=np.array(data['features']).reshape(1,-1)
        prediction=model.predict(features)
        return jsonify({'prediction':int(prediction[0])})
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5050,debug=True)    