import os
import logging
from transformers import pipeline

# Set up logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

log_file_path = os.path.join(log_directory, 'sentiment_analysis.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentAnalysisModel:
    def __init__(self):
        try:
            self.sentiment_model = pipeline("sentiment-analysis")
            logging.info("Sentiment analysis model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading sentiment analysis model: {e}")
            raise RuntimeError("Failed to load the sentiment analysis model.") from e

    def analyze_sentiment(self, text):
        try:
            if not isinstance(text, str):
                raise ValueError("Input must be a string.")
            
            results = self.sentiment_model(text)
            logging.info(f"Sentiment analyzed for text: {text}")
            return results
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            raise RuntimeError("Failed to analyze sentiment.") from e

if __name__ == "__main__":
    # Example usage
    model = SentimentAnalysisModel()
    
    test_text = "I hate using this library!. Don't use"
    try:
        sentiment_result = model.analyze_sentiment(test_text)
        print(sentiment_result)
    except Exception as e:
        print(e)
