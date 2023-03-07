import catboost as cb
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI


# Pydantic classes for input and output
class LoanApplication(BaseModel):
    Term: int
    NoEmp: int
    CreateJob: int
    RetainedJob: int
    longitude: float
    latitude: float
    GrAppv: float
    SBA_Appv: float
    is_new: str
    FranchiseCode: str
    UrbanRural: int
    City: str
    State: str
    Bank: str
    BankState: str
    RevLineCr: str
    naics_first_two: str
    same_state: str


class PredictionOut(BaseModel):
    default_proba: float


# Load the model
model = cb.CatBoostClassifier()
model.load_model("loan_catboost_model.cbm")

# Start the app
app = FastAPI()

# Home page
@app.get("/")
def home():
    return {"message": "Loan Default Prediction App", "model_version": 0.1}


# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: LoanApplication):
    cust_df = pd.DataFrame([payload.dict()])
    preds = model.predict_proba(cust_df)[0, 1]
    result = {"default_proba": preds}
    return result
