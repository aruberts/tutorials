import catboost as cb
import pandas as pd

from flask import Flask, jsonify, request

# Load the model
model = cb.CatBoostClassifier()
model.load_model("loan_catboost_model.cbm")

# Init the app
app = Flask("default")


# Setup prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    X = request.get_json()
    preds = model.predict_proba(pd.DataFrame(X, index=[0]))[0, 1]
    result = {"default_proba": preds}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8989)
