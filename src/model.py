import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

class UserEngagementModel:
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath
        self.model = LogisticRegression()
        self.trained = False
        self.setup_logging()

    def setup_logging(self):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'model_training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Logging is set up.")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_filepath)
            logging.info(f"Data loaded from {self.data_filepath}.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        try:
            X = self.df[["age", "session_duration", "number_of_actions", "last_active_days"]]
            y = self.df["engagement_label"]
            return X, y
        except KeyError as e:
            logging.error(f"Missing expected columns in data: {e}")
            raise

    def train_model(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        self.trained = True
        print("training done")
        logging.info("Model trained successfully.")

        self.save_model('engagement_model.pkl')

    def save_model(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self.model, file)
            logging.info(f"Model saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as file:
                self.model = pickle.load(file)
            logging.info(f"Model loaded from {filename}.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict_engagement(self, user_data):
        if not self.trained:
            logging.error("Model is not trained. Please train the model before making predictions.")
            raise Exception("Model is not trained.")
            
        
        try:
            model_data = UserEngagementModel("D:/Virtual Collaboration and Innovation Hub/data/user_engagement_data.csv")
            model_data.load_data()
            print("loading done")
            model_data.train_model()
            prediction = self.model.predict([user_data])
            logging.info(f"Predicted engagement for user data {user_data}: {prediction[0]}")
            return prediction[0]
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

if __name__ == "__main__":
    model = UserEngagementModel("D:/Virtual Collaboration and Innovation Hub/data/user_engagement_data.csv")
    model.load_data()
    model.train_model()
