from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prediction = model.predict(pd.DataFrame([data]))

    return jsonify({
        "Predicted Exam Score": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)