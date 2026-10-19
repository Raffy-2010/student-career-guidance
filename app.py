from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

careers = {
    "Healthcare": "Explore medicine, nursing, pharmacy, or other health-related fields.",
    "Engineering": "Explore civil, mechanical, electrical, or other engineering fields.",
    "Technology": "Explore software development, data science, cybersecurity, or IT.",
    "Education": "Explore teaching, educational technology, or academic research.",
    "Business": "Explore accounting, marketing, entrepreneurship, or management."
}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    description = None

    if request.method == "POST":
        features = [
            "math", "science", "english",
            "creativity", "helping_people", "technology"
        ]

        answers = {
            feature: request.form[feature]
            for feature in features
        }

        student_data = pd.DataFrame([answers])
        result = model.predict(student_data)[0]
        description = careers.get(result, "")

    return render_template(
        "index.html",
        result=result,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=True)