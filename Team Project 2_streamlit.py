import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import streamlit as st
import pickle
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
import pandas as pd
from sklearn.preprocessing import Imputer
# Load the pipeline and data
# pipe = pickle.load(open('pipe_logistic.sav', 'rb'))
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))
X_test = X_test.reset_index(drop = True)
describe = X_test.describe()

imputer_cat = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
X_test[['MaxDelq2PublicRecLast12M', 'MaxDelqEver']] = imputer_cat.fit_transform(X_test[['MaxDelq2PublicRecLast12M', 'MaxDelqEver']])
imputer_num = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
colnames = X_test.columns
X_test = imputer_num.fit_transform(X_test)
X_test = pd.DataFrame(X_test,columns = colnames)

dic = {0: 'bad', 1: 'good'} #prediction targets

#%%
# Function to test certain index of dataset
def test_demo(index):
    values = X_test.iloc[index]  # Input the value from dataset

    # Create four sliders in the sidebar
    a = st.sidebar.slider('ExternalRiskEstimate', 42.0, 93.0, values[0], 1.0)
    b = st.sidebar.slider('MSinceOldestTradeOpen', 11.0, 589.0, values[1], 1.0)
    c = st.sidebar.slider('MSinceMostRecentTradeOpen', 0.0, 383.0, values[2], 1.0)
    d = st.sidebar.slider('AverageMInFile', 0.0, 383.0, values[3], 1.0)
    e = st.sidebar.slider('NumSatisfactoryTrades', 0.0, 74.0, values[4], 1.0)
    f = st.sidebar.slider('NumTrades60Ever2DerogPubRec', 0.0, 12.0, values[5], 1.0)
    g = st.sidebar.slider('NumTrades90Ever2DerogPubRec', 0.0, 12.0, values[6], 1.0)
    h = st.sidebar.slider('PercentTradesNeverDelq', 0.0, 100.0, values[7], 1.0)
    i = st.sidebar.slider('MaxDelq2PublicRecLast12M', 0.0, 9.0, values[8], 1.0)
    j = st.sidebar.slider('MaxDelqEver', 0.0, 8.0, values[9], 0.1)
    k = st.sidebar.slider('NumTotalTrades', 0.0, 87.0, values[10], 1.0)
    l = st.sidebar.slider('NumTradesOpeninLast12M', 0.0, 12.0, values[11], 0.1)
    m = st.sidebar.slider('PercentInstallTrades', 0.0, 100.0, values[12], 1.0)
    n = st.sidebar.slider('NumInqLast6M', 0.0, 24.0, values[13], 0.1)
    o = st.sidebar.slider('NumInqLast6Mexcl7days', 0.0, 24.0, values[14], 0.1)
    p = st.sidebar.slider('NetFractionRevolvingBurden', 0.0, 117.0, values[15], 1.0)
    q = st.sidebar.slider('NumRevolvingTradesWBalance', 0.0, 20.0, values[16], 0.1)
    r = st.sidebar.slider('PercentTradesWBalance', 0.0, 100.0, values[17], 1.0)
    # Print the prediction result
    alg = ['Gradient Boosting Classifier', 'SVC', 'Logistic Regression']
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier == 'Gradient Boosting Classifier':
        # different trained models should be saved in pipe with the help pickle
       
        pipe = pickle.load(open('Boost_Pipeline.sav', 'rb'))
        
        arr = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r]).reshape(1,18)
        df = pd.DataFrame(arr, columns = X_test.columns)
        
        res = pipe.predict(df)
        st.write('Prediction:  ', dic[int(res)])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        st.markdown('Gradient Boosting Classifier Chosen')


    elif classifier == 'SVC':
        pipe = pickle.load(open('SVC_Pipeline.sav', 'rb'))
        
        arr = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r]).reshape(1,18)
        df = pd.DataFrame(arr, columns = X_test.columns)
        
        res = pipe.predict(df)
        st.write('Prediction:  ', dic[int(res)])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        st.markdown('SVC Chosen')


    else:
        pipe = pickle.load(open('Logit_Pipeline.sav', 'rb'))
        
        arr = np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r]).reshape(1,18)
        df = pd.DataFrame(arr, columns = X_test.columns)
        
        res = pipe.predict(df)
        st.write('Prediction:  ', dic[int(res)])
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        st.markdown('Logistic Regression Chosen')
    if res == 1:
        st.spinner('Wait for it...')
        st.balloons()
        st.success('**Good!**')
    else:
        st.spinner('Wait for it...')
        st.warning('**Risky!**')

#%%
# title
st.title('Credit Risk Performance Prediction')
import time
my_bar = st.progress(0)

for percent_complete in range(100):
    my_bar.progress(percent_complete + 1)
# show data
if st.checkbox('Show dataframe'):
     st.dataframe(X_test)
# st.write(X_train) # Show the dataset

number = st.text_input('Choose a row of information in the dataset:', 1)  # for users: Input the index number

test_demo(int(number))  # Run the test function



