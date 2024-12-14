import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import plotly.express as px
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import logging
from xgboost import XGBClassifier
import csv
import json
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score, accuracy_score
import catboost
from catboost import CatBoostClassifier
from pandas import json_normalize
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import xgboost as xgb
import pdb 
import joblib

class BClassify:
    print("inside bclassify class")
    

    def __init__(self, rawData, typeOfData):
        self.data = None 
        self.lr = None 
        self.randForst = None 
        self.xgbModel = None 
        self.catBModel = None
        self.dt_classifier = None
        self.grid_search1 = None
        self.xgbModelGridSearch = None

    
        self.lr = LogisticRegression(max_iter=100)

        self.randForst = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

        #self.xgbModel = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',objective='binary:logistic')
        self.xgbModel = XGBClassifier(random_state=42)    
        self.catBModel = CatBoostClassifier(random_state=42)
        
        if typeOfData == 'path':
            self.data = pd.read_csv(rawData)
            print(self.data)
        else:
            # Convert JSON string to Python list
            information = json.loads(rawData)

            # Convert Python list to DataFrame
            self.data  = pd.json_normalize(information)




    def preProcessing(self):

        print(self.data.columns)

        #since headers has unwanted space removing them
        self.data.columns = self.data.columns.str.strip()

        print(self.data.isnull().sum())

        self.histogramofFeature(self.data)
        print(self.data)

        colmn = {cols : len(self.data[cols].unique()) for cols in self.data.columns}
        print(f"The unique count of values of the features are {colmn}")

        #since 'Net Income Flag' column has constant value 1 removing it from the datafram
        self.data = self.data.drop('Net Income Flag', axis=1)


        self.visualizeTarget(self.data)


        self.getCorrelation(self.data)
        print(self.data.corr())

        #get the multicollinear features
        matrix_header = self.get_highly_correlated_cols(self.data, 0.8)


        #remove them from the actual dataframe
        newdata =  self.data.drop(matrix_header, axis=1)

        #see the data again
        print(self.data.corr())

        
        return newdata


    def visualizeTarget(self, info):
        

        # Count the occurrences of each value in the 'Bankrupt?' column
        bankrupt_counts = info['Bankrupt?'].value_counts()

        # Create a bar chart
        plt.bar(bankrupt_counts.index, bankrupt_counts.values)

        # Customize the chart
        plt.title('Bankruptcy Count')
        plt.xlabel('Bankrupt?')
        plt.ylabel('Count')

        # Add labels to the bars
        for index, value in enumerate(bankrupt_counts.values):
            plt.text(index, value, str(value), ha='center', va='bottom')

        # Show the chart
        plt.show()
        
    def getCorrelation(self, info):
        # Import the required libraries
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Calculate the correlation coefficients
        correlation_matrix = info.corr()

        # Create a heatmap to visualize the correlation matrix
        plt.figure(figsize=(45, 45))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.show()



    def get_highly_correlated_cols(self, info, threshold):   

        col_corr = set()
        corr_matrix = info.corr()
    
        for i in range(len(corr_matrix.columns)):
            for j in range(i):

                if i!=j and ((corr_matrix.iloc[i,j]) > threshold ):
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)

        return col_corr

    def histogramofFeature(self, data):
        
        data.hist(figsize=(50, 30), edgecolor='white')
        averages = data.mean()
        variances = data.var()

        plt.show()


    def split_the_Data(self, newdata):

        X = newdata.drop(['Bankrupt?'], axis=1)
        y = newdata['Bankrupt?']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42,stratify=y)

        return X_train, X_test, y_train, y_test
    


    def scaleXfeatures(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def trainLogisticModel(self, X_train, X_test, y_train, y_test):
        
        X_train_scaled, X_test_scaled = self.scaleXfeatures(X_train, X_test)

        # Fit the model on the training data
        self.lr.fit(X_train, y_train)

        # Predict on the training and test data
        y_train_pred = self.lr.predict(X_train_scaled)
        y_test_pred = self.lr.predict(X_test_scaled)
        y_prob = self.lr.predict_proba(X_test_scaled)[:, 1]


        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        self.roc_curve_graph(fpr, tpr, roc_auc)


        fsc = f1_score(y_test, y_test_pred)
        print(f"F1-score for logistic is {fsc} ")
        print(f"Roc Auc value for logistic is {roc_auc} ")

        return fsc, roc_auc
    
        
    def roc_curve_graph(self, fpr, tpr, roc_auc):
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


    def xgboostModel(self, X_train, X_test, y_train, y_test):
        #pdb.set_trace()

        self.xgbModel.fit(X_train, y_train)
         
        # Make predictions
        y_pred1 = self.xgbModel.predict(X_test)
        y_prob1 = self.xgbModel.predict_proba(X_test)[:, 1]


        # Compute ROC curve and ROC area
        fpr1, tpr1, _ = roc_curve(y_test, y_prob1)
        roc_auc1 = auc(fpr1, tpr1)
        self.roc_curve_graph(fpr1, tpr1, roc_auc1)


        fsc1 = f1_score(y_test, y_pred1)
        print(f"F1-score for xgboost is {fsc1} ")
        print(f"Roc Auc value for xgboost is {roc_auc1} ")


        joblib.dump(self.xgbModel, 'xgb_model.joblib')

        # param_grid = {
        #     'max_depth': [3, 6, 10],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'n_estimators': [50, 100, 200],
        #     'subsample': [0.8, 0.9, 1.0],
        #     'colsample_bytree': [0.8, 0.9, 1.0],
        #     'gamma': [0, 0.1, 0.2]
        # }

        # grid_search = GridSearchCV(estimator=self.xgbModel, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        # grid_search.fit(X_train, y_train)
        # best_params = grid_search.best_params_
        # print("Best Hyperparameters: ", best_params)

        # # Use the best model to make predictions
        # best_model = grid_search.best_estimator_
        # y_pred = best_model.predict(X_test)
        # y_prob = best_model.predict_proba(X_test)[:, 1]


        # fsc = f1_score(y_test, y_pred)
        # print(f"F1-score for XGBoost is {fsc}")

        # roc_auc = roc_auc_score(y_test, y_prob)
        # print(f"ROC AUC value for XGBoost is {roc_auc}")

        

        return fsc1, roc_auc1



    def catBoostModel(self, X_train, X_test, y_train, y_test):
            
        # Train the model
        self.catBModel.fit(X_train, y_train)

        # Make predictions
        y_pred = self.catBModel.predict(X_test)
        y_prob = self.catBModel.predict_proba(X_test)[:, 1]

    

        # Evaluate the model with the best hyperparameters
        y_pred_best = self.catBModel.predict(X_test)
        y_prob = self.catBModel.predict_proba(X_test)[:, 1]


        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        self.roc_curve_graph(fpr, tpr, roc_auc)


        fsc = f1_score(y_test, y_pred)

        print(f"F1-score for catBoost is {fsc} ")
        print(f"Roc Auc value for catBoost is {roc_auc} ")
        # classification_report = classification_report(y_test, y_pred)
        # print(f"classification_report for catBoost Model is {classification_report} ")


        return fsc, roc_auc
    
    