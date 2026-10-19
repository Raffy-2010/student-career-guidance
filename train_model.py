import pandas as pd
import pickle

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load the student dataset
data = pd.read_csv("students.csv")

# Features used to make career suggestions
features = [
    "math",
    "science",
    "english",
    "creativity",
    "helping_people",
    "technology",
]

X = data[features]
y = data["career"]

# Convert text answers into numbers
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OrdinalEncoder(
                categories=[["low", "medium", "high"]] * len(features)
            ),
            features,
        )
    ]
)

# Create the machine-learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )),
])

# Train the model
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Career Guidance model trained and saved successfully!")