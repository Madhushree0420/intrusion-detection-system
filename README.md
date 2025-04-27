# intrusion-detection-system
 

**Technology Stack:**

Layer Tool/Tech Why It Is Used Frontend Streamlit Used to build a simple, fast, interactive web app for real-time intrusion detection without complex frontend coding. Backend Python Main programming language to develop logic, handle model loading, user inputs, and predictions. Backend Scikit-learn Used to apply machine learning algorithms, including model training and prediction (Random Forest here). ML Algorithm RandomForestClassifier A powerful, ensemble learning algorithm that combines multiple decision trees to classify data accurately as attack or normal. Explanation SHAP (SHapley Additive exPlanations) Used to explain why the model predicted something, by showing which features influenced the output most — bringing trust to AI. Visualization Matplotlib Basic plotting library used to create graphs and charts for feature importance, SHAP values, etc. Visualization Seaborn Built on Matplotlib; makes the plots more beautiful, colorful, and easier to read for quick insights. File Handling Pandas Used to load, clean, and manipulate datasets (like NSL-KDD) and manage input/output data easily in DataFrames. File Handling joblib Used to save (serialize) and load the trained machine learning model files quickly without retraining every time.

Proof of Concept (POC): 

Steps Implemented: 
Data Upload: Accepts .csv or .txt NSL-KDD files. 
Preprocessing: Label encoding + feature scaling. 
Model Training: Random Forest classifier with saved model reuse. 

Evaluation: 
Accuracy score with a progress bar 
Confusion matrix & classification report 
SHAP plot to explain feature contributions 
Real-Time Input: Form-based input to classify a new observation 
Reporting: One-click download of prediction report and SHAP image 
Model Performance: 
Accuracy Achieved: ~90% (may vary) 
Model Used: Random Forest (100 trees, max depth = 10) 
Evaluation Metrics: 
Precision, Recall, F1-Score 
Confusion Matrix 
ROC Curve (optional) 

 

 Features: 

Upload NSL-KDD dataset 
Automated preprocessing 
Machine Learning prediction 
SHAP explanation 
Real-time intrusion testing 
Report download (Text + Image) 
Save prediction inputs 

 

 

 
