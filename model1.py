import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/sumit/Desktop/jupyter notebook/cleaned_dataset.csv')
from textblob import TextBlob

df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)
import re
def mentions_someone(text):
    return int(bool(re.search(r'<mention>', text)))
df['mentions_anyone'] = df['content'].apply(mentions_someone)
df['hashtag_count'] = df['content'].apply(lambda x: len(re.findall(r'#\w+', x)))

import datetime
df['datetime']=pd.to_datetime(df['datetime'])
df['hour']=df['datetime'].dt.hour

numeric_features = ['word_count', 'char_count', 'hour', 'sentiment', 
                   'hashtag_count']
categorical_features = ['has_media', 'mentions_anyone', 'company_encoded']
df["log1p"]=np.log1p(df["likes"])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['company_encoded'] = le.fit_transform(df['inferred company'])

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, random_state=42)
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)


from sklearn.model_selection import train_test_split
X = df[numeric_features + categorical_features]
y = df['log1p']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# training
search.fit(X_train, y_train)
best_model = search.best_estimator_
# Evaluation
preds = best_model.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(np.expm1(y_test),np.expm1(preds))
print("RMSE:", rmse)

import joblib
joblib.dump(model, 'model1.pkl')

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model1 = joblib.load('model1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert to DataFrame (assuming same structure as training)
        input_data = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model1.predict(input_data)
        
        # Prepare response
        response = {
            'prediction': prediction.tolist(),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/')
def home():
    return "TF-IDF Model API - Send POST requests to /predict"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)