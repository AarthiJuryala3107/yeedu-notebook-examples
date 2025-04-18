# Download the required data (bezdekIris.data) from https://github.com/yoyolicoris/IML_HW1/blob/master/bezdekIris.data
# Upload this file into Yeedu, copy the path, and change the data_path accordingly

# ============================
# Instructions to Deploy as Yeedu ML API
# ============================

# Create a New Job in Yeedu:
#    - Go to Jobs and click "Create Job".
#    - Select Job Type as: YEEDU_FUNCTIONS.
#    - In "Script Path", select this current file (e.g., ml_function.py).
#    - Add the following packages under "Function Requirements":
#        pandas
#        joblib==1.2.0
#        scikit-learn
#    - Use the following as "Example Request Body":
#        {
#          "sepal_length": 5.8,
#          "sepal_width": 2.7,
#          "petal_length": 5.1,
#          "petal_width": 1.9
#        }

# Run the Job:
# Access the ML API:
#    - Once the job status is "Running", copy the Function URL and Token from the job configuration.
#    - Test the API (e.g., via Postman):
#       - Make a POST request to the Function URL.
#       - Set Authorization Type to "Bearer Token" and paste the copied token.
#       - Use the example input as the request body (raw JSON).
#       - You should receive a prediction response.

# Note: The API will be active only while the job is running.


import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
# Step 1: Load and Preprocess the Dataset
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data_path = "file:///files/data/bezdekIris.data"              # Change accordingly
 # Change to actual path
df = pd.read_csv(data_path, header=None, names=columns)
df.to_csv("iris.csv", index=False)  # Save dataset for reference
 
# Step 2: Split Data into Features and Target
X = df.drop(columns=["class"])
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Step 3: Train the RandomForest Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
 
# Step 4: Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
 
# Step 5: Save the Trained Model
model_path = "iris_model.pkl"
joblib.dump(clf, model_path)
print(f"Model saved as {model_path}")
 
# Step 6: Define Functions for Yeedu Integration
def init():
    """Initialization function to load the trained model."""
    global clf
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
 
def yeedu_function(payload, context):
    """
    Yeedu function to make predictions based on input data.
    :param payload: Input features as JSON.
    :param context: Request metadata (not used here).
    :return: Predicted class as JSON.
    """
    try:
        # Extract input features
        features = [[
            float(payload["sepal_length"]),
            float(payload["sepal_width"]),
            float(payload["petal_length"]),
            float(payload["petal_width"])
        ]]
 
        # Make prediction
        prediction = clf.predict(features)
        return json.dumps({"predicted_class": prediction[0]})
    except Exception as e:
        return json.dumps({"error": str(e)})
