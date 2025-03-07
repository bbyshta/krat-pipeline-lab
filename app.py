from fastapi import FastAPI, File, UploadFile
import uvicorn
import joblib
import pandas as pd
from io import BytesIO
from pyngrok import ngrok

app = FastAPI()

model_path = "laptop_price_model.pkl"
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
  content = await file.read()
  df = pd.read_csv(BytesIO(content))
  predictions = model.predict(df)
  return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print("API доступно по адресу: ", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)