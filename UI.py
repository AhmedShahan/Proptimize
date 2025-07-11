import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import re
from langchain_ollama import ChatOllama
# Load environment variables
load_dotenv()

# Initialize Gemini Model
llm = ChatOllama(model="llama3.2:latest")
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

# Extract response parts with regex for robustness
def extract_response_parts(ai_response, current_draft):
    updated_prompt_match = re.search(r"Updated Prompt:\s*(.*)", ai_response, re.IGNORECASE)
    clarifying_question_match = re.search(r"Clarifying Question:\s*(.*)", ai_response, re.IGNORECASE)
    final_prompt_match = re.search(r"Final Prompt:\s*(.*)", ai_response, re.IGNORECASE)
    
    if final_prompt_match:
        return final_prompt_match.group(1).strip(), None, True
    
    updated_prompt = updated_prompt_match.group(1).strip() if updated_prompt_match else current_draft
    clarifying_question = truncate_question(clarifying_question_match.group(1).strip()) if clarifying_question_match else None
    
    return updated_prompt, clarifying_question, False

# Function to stream text character by character
def stream_text(text, delay=0.02):
    placeholder = st.empty()
    streamed = ""
    for char in text:
        streamed += char
        placeholder.markdown(streamed)
        time.sleep(delay)

# Streamlit App
st.title("🤖 Prompt Enhancement Assistant")
st.markdown("**Iteratively refine your AI prompts with Gemini AI. Type your messages below. Say 'generate now' to finalize.**")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = "No draft yet."

# Display conversation history dynamically
chat_container = st.container()
with chat_container:
    for message in st.session_state.conversation_history:
        if message.startswith("User:"):
            with st.chat_message("user"):
                st.markdown(message.replace("User: ", ""))
        elif message.startswith("AI: Updated Prompt:"):
            with st.chat_message("assistant"):
                st.markdown("**Updated Prompt:** " + message.split("Updated Prompt:")[1].split("Clarifying Question:")[0].strip())
                if "Clarifying Question:" in message:
                    st.markdown("**Clarifying Question:** " + message.split("Clarifying Question:")[1].strip())
        elif message.startswith("AI: Final Prompt:"):
            with st.chat_message("assistant"):
                st.success("✅ Final Enhanced Prompt:")
                st.markdown(message.replace("AI: Final Prompt: ", ""))
        else:
            with st.chat_message("assistant"):
                st.markdown(message.replace("AI: ", ""))

# Chat input
user_message = st.chat_input("Type your message here...")

if user_message:
    if user_message.lower() in ["exit", "quit"]:
        st.session_state.conversation_history.append("User: " + user_message)
        st.session_state.conversation_history.append("AI: Exiting. Goodbye!")
        st.rerun()  # Refresh to show exit message
    else:
        # Add user message to history
        st.session_state.conversation_history.append(f"User: {user_message}")

        # Classify input
        input_context = classify_input(user_message)

        # Prepare input for LLM chain
        chain_inputs = {
            "history": "\n".join(st.session_state.conversation_history),
            "user_input": user_message,
            "input_context": input_context,
            "draft_prompt": st.session_state.draft_prompt,
        }

        with st.spinner("🤔 Thinking..."):
            try:
                ai_response = chain.invoke(chain_inputs)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                ai_response = None

        if ai_response:
            # Extract response parts
            updated_prompt, clarifying_question, is_final = extract_response_parts(ai_response, st.session_state.draft_prompt)
            
            if is_final:
                # Stream final prompt character by character
                with chat_container.chat_message("assistant"):
                    st.success("✅ Final Enhanced Prompt:")
                    stream_text(updated_prompt)
                st.session_state.conversation_history.append(f"AI: Final Prompt: {updated_prompt}")
            else:
                # Update draft
                st.session_state.draft_prompt = updated_prompt
                
                # Prepare AI response for history (combined updated prompt and question)
                ai_display = f"Updated Prompt: {updated_prompt}\nClarifying Question: {clarifying_question}" if clarifying_question else f"Updated Prompt: {updated_prompt}"
                
                # Stream AI response character by character
                with chat_container.chat_message("assistant"):
                    stream_text(ai_display)
                
                # Add to history
                st.session_state.conversation_history.append(f"AI: {ai_display}")
            
            # Rerun to update the UI dynamically
            st.rerun()

# Dynamic status in sidebar for more interactivity
with st.sidebar:
    st.header("Session Status")
    st.write(f"**Current Draft Length:** {len(st.session_state.draft_prompt)} characters")
    st.write(f"**Messages in History:** {len(st.session_state.conversation_history)}")
    if st.button("Reset Session"):
        st.session_state.conversation_history = []
        st.session_state.draft_prompt = "No draft yet."
        st.rerun()
    st.markdown("---")
    st.caption("Powered by Gemini AI & Streamlit")