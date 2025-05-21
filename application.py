from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                distance_from_home=float(request.form.get('distance_from_home')),
                distance_from_last_transaction=float(request.form.get('distance_from_last_transaction')),
                ratio_to_median_purchase_price=float(request.form.get('ratio_to_median_purchase_price')),
                repeat_retailer=int(request.form.get('repeat_retailer')),
                used_chip=int(request.form.get('used_chip')),
                used_pin_number=int(request.form.get('used_pin_number')),
                online_order=int(request.form.get('online_order'))
            )

            pred_df = data.get_data_as_data_frame()
            print("Before Prediction")
            print(pred_df)

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")
            print(results)

            return render_template('home.html', results=results[0])

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('home.html', results="Error during prediction")

if __name__=="__main__":
    app.run(host="0.0.0.0", port= 8008)