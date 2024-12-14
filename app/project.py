from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from  Assignments.Project.BClassify import BClassify

app = FastAPI()

user_integer = None

# Query params vs Path params


@app.post("/make_prediction/")
def make_prediction(rawData: str = Query(..., description="Path to the raw data file"), 
                          typeOfData: str = Query(..., description="Type of the data source (e.g., 'path' or 'json')")):
    print("Inside the predict function")

    bkrp = BClassify(rawData, typeOfData)
    processed_features = bkrp.preProcessing()
    X_train, X_test, y_train, y_test = bkrp.split_the_Data(processed_features)
    #score1 = bkrp.trainLogisticModel(X_train, X_test, y_train, y_test)
    score2= bkrp.xgboostModel(X_train, X_test, y_train, y_test)
    #score3= bkrp.catBoostModel(X_train, X_test, y_train, y_test)
    #score4= bkrp.randomForestModel(X_train, X_test, y_train, y_test)
    
    response = {
        "xgboost_score": score2,
    }
    return response
 