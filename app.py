from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model_training/salary_model.pkl')
model_columns = joblib.load('model_training/model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        hours = int(request.form['hours_per_week'])
        workclass = request.form['workclass']
        country = request.form['country']

        user_input = {
            'age': age,
            'hours-per-week': hours,
            'workclass': workclass,
            'country': country
        }

        df = pd.DataFrame([user_input])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        pred = model.predict(df_encoded)[0]
        prediction_text = ">50K" if pred == 1 else "<=50K"

        return render_template('index.html', prediction=f"Predicted Salary Class: {prediction_text}")
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
