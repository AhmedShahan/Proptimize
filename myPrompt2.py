from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
parser = StrOutputParser()

# Refined Prompt Template
clarification_prompt_template = ChatPromptTemplate.from_template("""
You are a prompt engineering assistant tasked with iteratively crafting the BEST POSSIBLE AI PROMPT.

Conversation History:
{history}

Latest User Message:
{user_input}

Input Context:
{input_context}

Current Prompt Draft:
{draft_prompt}

### Instructions:
1. Analyze the latest user message, conversation history, and input context.
2. Revise the draft prompt to be **specific, concise, and aligned** with the user's goals.
3. Output the revised prompt as "Updated Prompt:".
4. Ask a single, **focused clarifying question** to refine the prompt further, as "Clarifying Question:".

🔑 Clarifying Question Rules:
- Must be **one sentence, under 15 words**.
- Be **action-oriented** and directly tied to the user's latest message.
- Include **exactly 5 example options** in parentheses (e.g., "What AI task do you need? (e.g., summarization, translation, code generation, classification, storytelling)").
- Avoid vague phrases like "What do you mean?" or broad questions.

If the user says "generate now" or similar, output the FINAL prompt as:
Final Prompt: <final prompt>

### Format:
Updated Prompt: <revised prompt>
Clarifying Question: <your question>
""")

# Chain Setup
chain = clarification_prompt_template | llm | parser

# Input classification
def classify_input(user_message):
    keywords = {
        "summarization": ["summary", "summarize", "condense"],
        "code": ["code", "programming", "script"],
        "generation": ["generate", "create", "write"],
        "analysis": ["analyze", "insights", "data"],
        "translation": ["translate", "language"]
    }
    for task, words in keywords.items():
        if any(word in user_message.lower() for word in words):
            return task
    return "general"

# Truncate clarifying question
def truncate_question(question):
    if question:
        words = question.split()
        if len(words) > 15:
            return " ".join(words[:15]) + "?"
    return question

# Extract response parts
def extract_response_parts(ai_response, current_draft):
    lines = ai_response.split("\n")
    updated_prompt = current_draft
    clarifying_question = None
    for line in lines:
        if line.startswith("Updated Prompt:"):
            updated_prompt = line.replace("Updated Prompt:", "").strip()
        elif line.startswith("Clarifying Question:"):
            clarifying_question = truncate_question(line.replace("Clarifying Question:", "").strip())
        elif line.startswith("Final Prompt:"):
            return line.replace("Final Prompt:", "").strip(), None, True
    return updated_prompt, clarifying_question, False

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

    # Classify input
    input_context = classify_input(user_message)

    # Prepare input for LLM chain
    chain_inputs = {
        "history": "\n".join(conversation_history),
        "user_input": user_message,
        "input_context": input_context,
        "draft_prompt": draft_prompt,
    }

    try:
        ai_response = chain.invoke(chain_inputs)
    except Exception as e:
        print(f"⚠️ Error: {e}")
        continue

    # Extract response parts
    updated_prompt, clarifying_question, is_final = extract_response_parts(ai_response, draft_prompt)
    if is_final:
        print("\n✅ Final Enhanced Prompt:\n")
        print(updated_prompt)
        break
    elif updated_prompt and clarifying_question:
        draft_prompt = updated_prompt
        print(f"\nUpdated Prompt: {draft_prompt}")
        print(f"Clarifying Question: {clarifying_question}\n")
    else:
        print("⚠️ Malformed AI response. Keeping previous draft.")

    # Add AI response to history
    conversation_history.append(f"AI: {ai_response}")