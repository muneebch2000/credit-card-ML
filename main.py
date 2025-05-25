from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")

# Load hybrid models
supervised_model = joblib.load("supervised_model.pkl")
unsupervised_model = joblib.load("unsupervised_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input model
class TransactionInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to Hybrid Credit Card Fraud Detection"}

def hybrid_predict(data: pd.DataFrame):
    iso_pred = unsupervised_model.predict(data)
    sup_prob = supervised_model.predict_proba(data)[:, 1]

    results = []
    for anomaly, prob in zip(iso_pred, sup_prob):
        prob = float(prob)  # Convert numpy.float32 to Python float
        if anomaly == -1 and prob > 0.5:
            label = "Fraud"
        elif anomaly == -1:
            label = "Suspicious"
        elif anomaly == 1 and prob > 0.8:
            label = "Likely Fraud"
        else:
            label = "Not Fraud"
        results.append({"Prediction": label, "Fraud Probability": round(prob, 4)})
    return results

@app.post("/predict_file/")
def predict_from_csv(file: UploadFile = File(...)):
    contents = file.file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Drop 'id' column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])

    return hybrid_predict(df)
