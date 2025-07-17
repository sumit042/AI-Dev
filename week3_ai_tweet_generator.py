from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask import Flask, request, jsonify

print("loading")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print("GPT-2 loaded.")

class AITweetGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate_ai_tweet(self, prompt, max_length=60):
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
    
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tweet = generated_text[len(prompt):].strip()
        return tweet[:280]
app = Flask(__name__)
ai_generator = AITweetGenerator(tokenizer, model)

@app.route('/generate_ai', methods=['POST'])
def generate_ai():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        if not prompt.strip():
            return jsonify({'error': 'Prompt is required.'}), 400
        tweet = ai_generator.generate_ai_tweet(prompt)
        return jsonify({'tweet': tweet})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
