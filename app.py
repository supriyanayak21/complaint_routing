from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("models/priority_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    complaint = request.form["complaint"]
    complaint_vector = vectorizer.transform([complaint])
    prediction = model.predict(complaint_vector)[0]
    return render_template(
        "result.html",
        complaint=complaint,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)