
from flask import Flask,render_template,request
import pandas as pd
import pickle
import sklearn
import numpy
import json


app = Flask(__name__)
df = pd.read_csv("airbnb prices.csv")
model = pickle.load(open('airbnb.pkl','rb'))
print("Model Load")

@app.route("/")
def index():
    room_types=sorted(df["room_type"].unique())
    Neighborhood=sorted(df["neighborhood"].unique())
    return render_template('index.html',room_types=room_types,Neighborhood=Neighborhood)

@app.route("/predict", methods=['POST'])
def predict():
    room_types= request.form.get("room_type")
    neighborhood= request.form.get("neighborhood")
    overall_satisfaction= request.form.get("overall_satisfaction")
    accomodate= request.form.get("accommodates")
    bedroom= request.form.get("bedrooms")
    latitude= request.form.get("latitude")
    longitude= request.form.get("longitude")
    print(room_types,neighborhood,overall_satisfaction,accomodate,bedroom,latitude,longitude)
    input=pd.DataFrame([[room_types,neighborhood,overall_satisfaction,accomodate,bedroom,latitude,longitude]],columns=["room_type","neighborhood","overall_satisfaction","accommodates","bedrooms","latitude","longitude"])
    prediction= model.predict(input)[0] **2
    return str(round(prediction,2))

@app.route("/about")
def about():
    return render_template('about.html')


app.run(debug=True)

