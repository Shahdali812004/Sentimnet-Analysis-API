from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
#Custom modules
from src.helper.config import API_KEY, APP_NAME, VERSION
from src.controllers.NLPController import NLPTrainer
from src.models.schemas import(
    TrainingData, TestingData, QueryData,
      StatusObject, PredectionObject, BatchPredectionObject
)

#Initialize NLP Trainer
nlp_trainer = NLPTrainer()

#Initilize an app
app = FastAPI(title=APP_NAME, version=VERSION)
#CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Authorization 
API_KEY_HEADER = APIKeyHeader(name="X-API-Key") #Expecting API key in the header request with the name "X-API-Key"
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.get("/",tags=['Healthy Check'],description="Check if the API is running")
async def healthy_check(api_key:str=Depends(verify_api_key)):
    return {
            "app_name": APP_NAME,
            "version": VERSION,
            "status": "API is running"}

@app.get("/model/status",tags=['Model Status'],
         description="Get the current status of the model",response_model=StatusObject)
async def model_status(api_key:str=Depends(verify_api_key))->StatusObject:
       response = nlp_trainer.get_status()
       return StatusObject(**response)

@app.post("/model/train",tags=['Model Training'],
          description="Train the model with provided data",response_model=StatusObject)
async def train_model(data:TrainingData,api_key:str=Depends(verify_api_key))->StatusObject:
          try:
            nlp_trainer.train(data.texts, data.labels)
            new_status = nlp_trainer.get_status()
            return StatusObject(**new_status)
          except Exception as e:
               raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
          
@app.post("/model/predict",tags=['Single Prediction'],
          description="Get prediction for a single text input",response_model=PredectionObject)
async def predict_single(data:QueryData,api_key:str=Depends(verify_api_key))->PredectionObject:
    try:
      predection = nlp_trainer.predict([data.text])[0]
      return PredectionObject(**predection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

@app.post("/model/Batch_prediction",tags=['Batch Prediction']
          ,description="Get prediction for a Batch text input",response_model=BatchPredectionObject)
async def predict_batch(data:TestingData,api_key:str=Depends(verify_api_key))->BatchPredectionObject:
    try:
      predection = nlp_trainer.predict(data.texts)
      return BatchPredectionObject(predictions=predection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    