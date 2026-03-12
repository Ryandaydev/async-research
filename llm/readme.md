# LLM examples


## Jupyter notebook
Uses OpenAI completions API with fan-out, fan-in pattern


## Streaming API - Chat Competions
Uses OpenAI
Test with this
```
curl -N   -X POST http://127.0.0.1:8000/chat/stream   -H "Content-Type: application/json"   -d '{
    "prompt": "Tell me about the Oklahoma Sooners football history.",
    "model": "gpt-4o-mini"
  }'
```

