# reverse_prompt_builder.py

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
parser = StrOutputParser()

# Reverse Prompting Template
clarification_prompt_template = ChatPromptTemplate.from_template("""
You are a prompt engineering assistant. Your task is to iteratively build the BEST POSSIBLE AI PROMPT based on user conversations.

Conversation History:
{history}

Latest User Message:
{user_input}

Current Prompt Draft:
{draft_prompt}

### Instructions:
1. First, improve the draft prompt based on the latest user message. Show this as "Updated Prompt:".
2. Then, ask your next clarifying question as "Clarifying Question:".

🔑 Clarifying Question Rules:
- Keep it **short (max 1 sentence)**.
- No explanations, just ask.
- Stay relevant to the user's last message.
-always give example like  What aspect of AI are you most interested in? (e.g Machine learning, Deep Learning, Generative AI at least 5 e.g)
- Example: "Which topic in AI would you like to focus on?" instead of "Do you want a very brief definition, or a more detailed explanation of AI, including its different types and applications?"

If the user says "generate now" or similar, stop asking questions and output the FINAL prompt as:
Final Prompt: <final prompt>

### Format:
Updated Prompt: <revised prompt>
Clarifying Question: <your question>
""")


# Chain Setup
chain = clarification_prompt_template | llm | parser

# Initialize state
conversation_history = []
draft_prompt = "No draft yet."

print("🤖 Prompt Enhancement Assistant Ready!")
print("Type your messages. Type 'exit' to quit.\n")

while True:
    user_message = input("User: ").strip()
    if user_message.lower() in ["exit", "quit"]:
        print("🤖 Exiting. Goodbye!")
        break

    # Add user message to history
    conversation_history.append(f"User: {user_message}")

    # Prepare input for LLM chain
    chain_inputs = {
        "history": "\n".join(conversation_history),
        "user_input": user_message,
        "draft_prompt": draft_prompt,
    }

    try:
        ai_response = chain.invoke(chain_inputs)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        continue

    # Show the entire AI response
    print(f"\n{ai_response}\n")

    # Extract updated prompt
    if "Final Prompt:" in ai_response:
        final_prompt = ai_response.split("Final Prompt:")[1].strip()
        print("\n✅ Final Enhanced Prompt:\n")
        print(final_prompt)
        break
    elif "Updated Prompt:" in ai_response:
        # Update the prompt for the next turn
        draft_prompt = ai_response.split("Updated Prompt:")[1].split("Clarifying Question:")[0].strip()
    else:
        print("⚠️ Could not extract Updated Prompt. Keeping previous draft.")

    # Add the AI's full response to history
    conversation_history.append(f"AI: {ai_response}")
