from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

class Database:
    def __init__(self):
        self.uri = "mongodb+srv://harshsharma91318:haWQiXgevOmOG3U9@clustermlapp.jruvabd.mongodb.net/?retryWrites=true&w=majority&appName=ClusterMLAPP"

        # Create a new client and connect to the server
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(" Connection failed:", e)

        # Connect to the database and collection
        self.db = self.client.get_database('OverallPrediction')
        self.records = self.db['fifaplayerdata']

    def check_connection(self):
        print("Total documents in collection:", self.records.count_documents({}))

    def insert_csv_data(self, csv_file):
        df = pd.read_csv(csv_file)
        print("CSV Columns:", df.columns.tolist())  # Optional: See what’s in your data
        examples = df.to_dict(orient='records')
        self.records.insert_many(examples)
        print(f"Inserted {len(examples)} documents into MongoDB.")
        
    def add_single_document(self, input_data):
        try:
                result = self.records.insert_one(input_data)
                print("Inserted one document:", result.inserted_id)
                return result
        except Exception as e:
                print("Insert failed:", e)
                return None 



 
if __name__ == '__main__':
    D = Database()
    
    D.insert_csv_data('fifa_players.csv')
    
    D.check_connection()
   
        