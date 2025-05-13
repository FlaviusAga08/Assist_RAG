# Assist_RAG

## Stept to run
- Activate the virtual environment RAG like this:
``` source RAG/bin/activate ```
 - Fill OPENAI_API_KEY environment variable with a valid openAPI key.

- start fastapi:``` uvicorn answers:app --reload```

## Optional
Before training remove the database: ```rm -rf db/```