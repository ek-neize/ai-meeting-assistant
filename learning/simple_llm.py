import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load OPENAI_API_KEY and OPENAI_MODEL from the .env file
load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.5,   # Lower = more deterministic
    max_tokens=512,    # Cap on generated tokens
    top_p=1,
)

response = llm.invoke("How to read a book effectively?")
print(response.content)
