# Assist_RAG

Activate the virtual environment RAG like this:
source RAG/bin/activate
Then start fastapi:
uvicorn answers:app --reload
Make sure to remove the database before training like this:
rm -rf db/
De asemenea, schimbati cheia OpenAI din .env cu cheia dumneavoastra
