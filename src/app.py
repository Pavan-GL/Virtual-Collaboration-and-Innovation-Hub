import os
import logging
from flask import Flask, request, jsonify
from model import UserEngagementModel
from sentiment_analysis_model import SentimentAnalysisModel
from idea_generation_model import IdeaGenerationModel

# Setup logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

log_file_path = os.path.join(log_directory, 'app.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Initialize models
        self.sentiment_model = SentimentAnalysisModel()
        self.idea_model = IdeaGenerationModel()
        
        # File path for user engagement model
        data_filepath = 'D:/Virtual Collaboration and Innovation Hub/data/user_engagement_data.csv'
        self.user_model = UserEngagementModel(data_filepath)

        # Load and train user engagement model once at startup
        self.user_model.load_data()
        self.user_model.train_model()

    def setup_routes(self):
        @self.app.route('/analyze_sentiment', methods=['POST'])
        def sentiment():
            try:
                data = request.json
                text = data.get('text', '')
                if not text:
                    raise ValueError("Text is required for sentiment analysis.")
                
                sentiment_result = self.sentiment_model.analyze_sentiment(text)
                logging.info(f"Sentiment analysis performed on text: {text}")
                return jsonify(sentiment_result)
            except Exception as e:
                logging.error(f"Error during sentiment analysis: {e}")
                return jsonify({"error": str(e)}), 400

        @self.app.route('/generate_ideas', methods=['POST'])
        def ideas():
            try:
                data = request.json
                prompt = data.get('prompt', '')
                if not prompt:
                    raise ValueError("Prompt is required for idea generation.")
                
                generated_ideas = self.idea_model.generate_ideas(prompt)
                logging.info(f"Ideas generated for prompt: {prompt}")
                return jsonify(generated_ideas)
            except Exception as e:
                logging.error(f"Error during idea generation: {e}")
                return jsonify({"error": str(e)}), 400
            
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Get user data from the request
                age = int(request.form.get('age', 0))
                session_duration = int(request.form.get('session_duration', 0))
                number_of_actions = int(request.form.get('number_of_actions', 0))
                last_active_days = int(request.form.get('last_active_days', 0))

                # Prepare input for prediction
                user_data = [age, session_duration, number_of_actions, last_active_days]
                
                # Predict engagement
                prediction = self.user_model.predict_engagement(user_data)
                prediction_label = "Engaged" if prediction == 1 else "Not Engaged"
                
                logging.info(f"Prediction made: {prediction_label}")
                return jsonify(prediction_label)
            except Exception as e:
                logging.error(f"Error during Prediction: {e}")
                return jsonify({"error": str(e)}), 400

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    FlaskApp().run()
