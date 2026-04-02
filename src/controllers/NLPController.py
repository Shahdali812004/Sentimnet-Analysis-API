import os
import json
import joblib
from threading import get_ident,Thread
from typing import Dict,Union,List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from datetime import datetime
#custom Modules
from src.helper.config import STORAGE_FOLDER_PATH
class NLPTrainer:
    def __init__(self)->None:
        self.storage_path = STORAGE_FOLDER_PATH

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        
        self.model_path = os.path.join(self.storage_path, "model.joblib")
        self.status_path = os.path.join(self.storage_path, "model_status.json")

        if os.path.exists(self.status_path):
            with open(self.status_path, "r") as f:
                self.model_status = json.load(f)

        else:
            self.model_status = {
                "status": "No Model Trained",
                "timestamp": datetime.now().isoformat(),
                "classes": [],
                "evaluation": {}
            }

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

        else:
            self.model = None

        self._running_threads = []
        self._pipline = None

    def _update_status(self,status:str,classes:List[Union[str,int]]=[],evaluation:Dict={})->None:
        self.model_status['status'] = status
        self.model_status['timestamp'] = datetime.now().isoformat()
        self.model_status['classes'] = classes
        self.model_status['evaluation'] = evaluation

        with open(self.status_path, "w+") as file:
            json.dump(self.model_status,file,indent=2)


    def _train_job(self, X_train:List[str],y_train:List[Union[str,int]],
                   X_test:List[str],y_test:List[Union[str,int]]):
        
        self._pipline.fit(X_train, y_train)

        report = classification_report(y_test,self._pipline.predict(X_test)
                                       ,output_dict=True,zero_division=0)
        classes = self._pipline.classes_.tolist()

        self._update_status(
            status="Model Ready",
            classes=classes,
            evaluation=report
        )
        #save model
        joblib.dump(self._pipline, self.model_path, compress=9)

        #update model
        self.model = self._pipline
        self._pipline = None
        
        #free completed threads
        thread_id = get_ident()
        for i , t in enumerate(self._running_threads):
            if t.ident == thread_id:
                self._running_threads.pop(i)
                break
    
    def train(self,texts:List[str], labels:List[str])->None:

        if len(self._running_threads):
            raise Exception("A training procss is already running, Please Wait!")

        X_train, X_test, y_train,y_test = train_test_split(texts, labels)
        clf = LogisticRegression()
        vec = TfidfVectorizer(stop_words='english',
                              min_df=0.01,max_df=0.35,ngram_range=(1,2))
        
        self._pipline = make_pipeline(vec,clf)

        self._update_status(status = "Training")

        self.model = None
        #move to seperated thread
        t = Thread(target=self._train_job, args=(X_train,y_train,X_test,y_test))
        self._running_threads.append(t)
        t.start()

    def predict(self, texts:List[str]):
        responses = []
        if self.model:
            probs = self.model.predict_proba(texts)
            for i, row in enumerate(probs):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['prediction'] ={cls: round(float(prob), 3)
                    for cls , prob in zip(self.model_status['classes'],row)}
                responses.append(row_pred)
        else:
            raise Exception("No models founded")
        
        return responses

    def get_status(self)->Dict:
        return self.model_status