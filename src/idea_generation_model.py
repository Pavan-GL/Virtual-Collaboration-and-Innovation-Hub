import os
import logging
import json
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch 

# Set up logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

log_file_path = os.path.join(log_directory, 'idea_generation.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IdeaGenerationModel:
    def __init__(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
             # Ensure the pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token 
            logging.info("GPT-2 model and tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise RuntimeError("Failed to load the GPT-2 model or tokenizer.") from e

    def generate_ideas(self, prompt, output_dir="generated_ideas"):
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Encode the input prompt
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = (inputs != self.tokenizer.pad_token_id).to(torch.int64)
            outputs = self.model.generate(
            inputs,
            max_length=200,            
            num_return_sequences=1,
            temperature=1.5,
            num_beams=5,          
            top_k=50,                  
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=attention_mask
            
                                              
        )
            ideas = [
                self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for output in outputs
            ]

            unique_ideas = []
            seen = set()
            for idea in ideas:
                if idea not in seen:
                    unique_ideas.append(idea)
                    seen.add(idea)
         


            logging.info(f"Generated ideas for prompt: '{prompt}'")

            # Save generated ideas to JSON and pickle files
            json_path = os.path.join(output_dir, 'ideas.json')
            pickle_path = os.path.join(output_dir, 'ideas.pkl')

            with open(json_path, 'w') as json_file:
                json.dump(ideas, json_file)
            logging.info(f"Generated ideas saved to {json_path}")

            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(ideas, pickle_file)
            logging.info(f"Generated ideas saved to {pickle_path}")

            return unique_ideas
        except Exception as e:
            logging.error(f"Error generating ideas: {e}")
            raise RuntimeError("Failed to generate ideas.") from e

if __name__ == "__main__":
    # Example usage
    model = IdeaGenerationModel()
    
    prompt_text = "Generate innovative ideas for improving urban transportation in Seattle."

    try:
        generated_ideas = model.generate_ideas(prompt_text)
        print("Generated Ideas:")
        for idea in generated_ideas:
            print(idea)
    except Exception as e:
        print(e)
