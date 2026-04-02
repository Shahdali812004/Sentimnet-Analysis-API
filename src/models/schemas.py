from pydantic import BaseModel
from typing import List,Union, Dict

class TrainingData(BaseModel):
    texts: List[str]
    labels: List[Union[str, int]]

#for Batch Prediction
class TestingData(BaseModel):
    texts: List[str]

#for Single Prediction
class QueryData(BaseModel):
    text: str

class StatusObject(BaseModel):
    status: str
    timestamp: str
    classes: List[str]   
    evaluation: Dict

#for single prediction
class PredectionObject(BaseModel):
    text: str
    prediction: Dict

#for batch prediction
class BatchPredectionObject(BaseModel):
    predictions: List[PredectionObject]
     