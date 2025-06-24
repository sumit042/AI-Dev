import requests

# Test prediction API (from Week 2)
prediction_response = requests.post('http://localhost:5000/predict', json={
    'word_count': 15,
    'char_count': 120,
    'hour': 14,
    'sentiment': 0.8,
    'hashtag_count':1,
    'has_media':1,
    'mentions_anyone':1,
    'company_encoded':0
    
})
curl -X POST http://localhost:5000/generate_ai \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Exciting news from our AI team:"}'
generated_tweet=requests.post('http://localhost:5001/generate_ai', json={
    'prompt':'Exciting news from our company Nike:'
})

print("Predicted Likes:", prediction_response.json())
print("generated tweet:",generated_tweet)