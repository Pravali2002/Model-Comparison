import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
aws_region = os.getenv("AWS_REGION")

print("LangSmith Key:", bool(langchain_api_key))
print("AWS Region:", aws_region)
