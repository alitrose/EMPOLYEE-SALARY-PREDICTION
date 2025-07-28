from flask import Flask, render_template, request
import joblib

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = []
    for col in encoders:
        if col in data:
            data[col] = encoders[col].transform([data[col]])[0]

    feature_order = ['age', 'workclass', 'education', 'occupation', 'Hours-per-week']


    for col in feature_order:
        features.append(float(data[col]))

    prediction = model.predict([features])[0]
    result = encoders['income'].inverse_transform([prediction])[0]

    return render_template("result.html", prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
