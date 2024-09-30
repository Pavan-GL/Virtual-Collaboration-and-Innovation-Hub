import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import joblib
import logging

class UserEngagementModel:
    def __init__(self, data_filepath, model_save_dir='models'):
        self.data_filepath = data_filepath
        self.model_save_dir = model_save_dir
        self.model = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(os.path.join(model_save_dir, 'D:/Virtual Collaboration and Innovation Hub/logs/model_training.log')),
                                logging.StreamHandler()
                            ])

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.data_filepath}")
            data = pd.read_csv(self.data_filepath)
            logging.info("Data loaded successfully.")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, data):
        try:
            logging.info("Preprocessing data.")
            # Identify categorical columns
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            logging.info(f"Categorical columns identified: {categorical_cols}")

            # Encode categorical features
            for col in categorical_cols:
                if col != 'engagement_label':
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    logging.info(f"Encoded column: {col}")

            X = data.drop('engagement_label', axis=1)  # Features
            y = data['engagement_label']  # Target variable
            
            logging.info("Data preprocessing completed.")
            return X, y
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise

    def train_model(self, X_train, y_train):
        try:
            logging.info("Training the model.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return self.model
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def evaluate_model(self, X_test, y_test):
        try:
            logging.info("Evaluating the model.")
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Model accuracy: {accuracy * 100:.2f}%")
            return accuracy
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

    def save_model(self, filename):
        try:
            os.makedirs(self.model_save_dir, exist_ok=True)
            joblib.dump(self.model, os.path.join(self.model_save_dir, filename))
            logging.info(f"Model saved to {filename}.")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load_model(self, filename):
        try:
            self.model = joblib.load(os.path.join(self.model_save_dir, filename))
            logging.info(f"Model loaded from {filename}.")
            return self.model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    # Example workflow
    try:
        engagement_model = UserEngagementModel(data_filepath="D:/Virtual Collaboration and Innovation Hub/data/user_engagement_data.csv")
        data = engagement_model.load_data()
        X, y = engagement_model.preprocess_data(data)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        engagement_model.train_model(X_train, y_train)

        # Evaluate the model
        accuracy = engagement_model.evaluate_model(X_test, y_test)

        # Save the model for future use
        engagement_model.save_model('user_engagement_model.pkl')

    except Exception as e:
        logging.error(f"An error occurred in the workflow: {e}")
