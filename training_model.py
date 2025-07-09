import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

class Model:
    def __init__(self):
        uri = "mongodb+srv://harshsharma91318:haWQiXgevOmOG3U9@clustermlapp.jruvabd.mongodb.net/?retryWrites=true&w=majority&appName=ClusterMLAPP"
        client = MongoClient(uri, server_api=ServerApi('1'))
        db = client.get_database('OverallPrediction')
        self.records = db['fifaplayerdata']
        self.df = pd.DataFrame(list(self.records.find({})))

        # Drop the '_id' column if it exists
        if '_id' in self.df.columns:
            self.df.drop('_id', axis=1, inplace=True)   

    def train(self):
        # Drop 'nationality_name' only if it exists
        if 'nationality_name' in self.df.columns:
            df1 = self.df.drop(['nationality_name'], axis=1)
        else:
            df1 = self.df.copy()

        # Ensure 'overall' exists
        if 'overall' not in self.df.columns:
            raise ValueError("'overall' column not found in the data!")

        print(" Columns in DataFrame:", self.df.columns.tolist())

        y = self.df['overall']
        df_numeric = df1.select_dtypes(include='number')
        X = df_numeric.drop('overall', axis=1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("🔍 Sample Features:\n", X.head())

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SGDRegressor(max_iter=1000, learning_rate='adaptive', eta0=0.01, random_state=42))
        ])

        # Train and save
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, 'linear_regression.pkl')

        # Load and predict
        loaded_pipeline = joblib.load('linear_regression.pkl')
        y_test_pred = loaded_pipeline.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        print(f" Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.4f}")

        return r2

if __name__ == '__main__':
    m = Model()
    score = m.train()
    print("✅ Final R² Score:", score)

    
    
     