import pickle
from flask import Flask, request, app, jsonify, render_template, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load the model
churn_model = pickle.load(open('churn_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    df = pd.DataFrame([data])
    # 1. Transform the categorical 'salary' using encoder.pkl
    salary_encoded = encoder.transform(df[['salary']])
    salary_df = pd.DataFrame(
        salary_encoded, 
        columns=encoder.get_feature_names_out(['salary'])
    )
    # 2. Drop the string 'salary' and concat the numeric dummies
    # Ensure empid is dropped bacause it wasn't used in training
    df_numeric = df.drop(['salary', 'empid'], axis=1, errors='ignore')
    final_df = pd.concat([df_numeric, salary_df], axis=1)
    
    output = churn_model.predict(final_df)
    return jsonify(int(output[0]))

if __name__ == "__main__":
    app.run(debug=True)