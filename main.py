from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Load a prompt template from the Hub
prompt = hub.pull("hardkothari/prompt-maker")

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5
)

# Initialize the output parser
parser = StrOutputParser()

# Create the chain
chain = prompt | llm | parser

# Example input (depends on your prompt template input variable)
response = chain.invoke({
    "lazy_prompt": "You are an AI assistant.",
    "task": "Summarize the AI"
})


# Print the response
print(response)
