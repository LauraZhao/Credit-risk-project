# Credit-risk-project
A predictive model of credit risk: Data Cleaning:  1. Checked and deleted the columns where missing values make up more than 90%.  2. Imputed the median of each respective numerical column and for categorical columns and the most frequent value for categorical columns.  3. Scaled them with Standardized scaler in the pipeline.  Model Building:  1. Separated the data into a training set and testing set.  2. Created a function to input different models and return the results.  3. Conducted prediction with 13 models (SVC, Logit, Decision Tree, Random Forest, MPL,  Decision Tree Classifier, KNN, Ada Boost, Gaussian NB, Quadratic Discriminant Analysis, Gradient Boosting, Bagging, Extra Trees), compared the performan and chose 3 models with the best performance(Gradient Boosting,SVC, Logit). Interface Building with Streamlit:  1. Created an interface to present the prediction outcomes of the credit risk performance for each customer and the accuracy and confusion matrix of different predictive models.
