# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import torch
from app.schemas import PredictRequest, PredictResponse
from app.utils import load_model

app = FastAPI(title="ML GPU Inference API", version="1.0")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)

API_KEY = "secret-key"

def verify_token(request: Request):
    token = request.headers.get("x-api-key")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_token)])
def predict(payload: PredictRequest):
    try:
        x = torch.tensor(payload.features, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        return {"prediction": output.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ⬇️ ESTA PARTE ES CLAVE
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


# example usage:
"""
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -H "x-api-key: secret-key" ^
  -d "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}"
## example output: {"prediction":[0.22890420258045197,-0.20764021575450897]}
"""

# example health check:
"""
curl -X GET http://localhost:8000/health
## example output: {"status":"ok","device":"cuda"}
"""
# example token verification:
"""
curl -X GET http://localhost:8000/health -H "x-api-key
## example output: {"status": "ok", "device": "cuda"}
"""

# example error handling:
"""
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -H "x-api-key: secret-key" -d "{\"features\": [0.1, 0.2, 0.3]}"
## example error output: {"detail": "Invalid input data"}
"""

# example request invalidad api key:
"""
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -H "x-api-key: wrong-key" -d "{\"features\": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}"
"""
# example error output: {"detail": "Invalid API Key"}
# example deployment:
# kubectl apply -f deployment.yaml  

