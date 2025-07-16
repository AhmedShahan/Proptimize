import streamlit as st
from dotenv import load_dotenv
import re
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Load env vars
load_dotenv()

# === Model and Parser ===
llm = ChatOllama(model="llama3.2:latest")
parser = StrOutputParser()

# === Prompt Template ===
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
- Must be **one sentence, under 15 words**. always with in example as like (e.g) at Clarifying Question
- Be **action-oriented** and directly tied to the user's latest message.
- Include **exactly 5 example options** in parentheses (e.g., "What AI task do you need? (e.g., summarization, translation, code generation, classification, storytelling)").

If the user says "generate now", output the FINAL prompt as:
Final Prompt: <final prompt>

### Format:
Updated Prompt: <revised prompt>
Clarifying Question: <your question>
""")

chain = clarification_prompt_template | llm | parser

# === Helper Functions ===
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

def truncate_question(question):
    if question:
        words = question.split()
        if len(words) > 15:
            return " ".join(words[:15]) + "?"
    return question

def extract_response_parts(ai_response, current_draft):
    updated_prompt_match = re.search(r"Updated Prompt:\s*(.*)", ai_response, re.IGNORECASE)
    clarifying_question_match = re.search(r"Clarifying Question:\s*(.*)", ai_response, re.IGNORECASE)
    final_prompt_match = re.search(r"Final Prompt:\s*(.*)", ai_response, re.IGNORECASE)
    
    if final_prompt_match:
        return final_prompt_match.group(1).strip(), None, True
    
    updated_prompt = updated_prompt_match.group(1).strip() if updated_prompt_match else current_draft
    clarifying_question = truncate_question(clarifying_question_match.group(1).strip()) if clarifying_question_match else None
    
    return updated_prompt, clarifying_question, False

def stream_text(text, delay=0.01):
    placeholder = st.empty()
    streamed = ""
    for char in text:
        streamed += char
        placeholder.markdown(streamed)
        time.sleep(delay)

# === Session State Initialization ===
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = "No draft yet."
if "history_expanded" not in st.session_state:
    st.session_state.history_expanded = True

# === UI Header ===
st.set_page_config(page_title="Prompt Enhancer", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Prompt Enhancement Assistant</h1>", unsafe_allow_html=True)
st.caption("Refine AI prompts interactively using LLM guidance. Type your intent and say 'generate now' when you're ready.")

# === Sidebar Features ===
with st.sidebar:
    st.header("🔧 Settings & Tools")
    st.write("**Current Draft Prompt:**")
    st.code(st.session_state.draft_prompt, language='markdown')
    if st.button("📋 Copy Final Prompt"):
        st.toast("Copied to clipboard!", icon="✅")
    if st.button("🗑️ Reset Session"):
        st.session_state.conversation_history = []
        st.session_state.draft_prompt = "No draft yet."
        st.rerun()
    st.markdown("---")
    st.markdown("### 🎯 Choose Prompt Type")
    st.session_state.input_context = st.radio(
        "Prompt Context", 
        ["general", "summarization", "code", "generation", "analysis", "translation"],
        horizontal=True
    )
    st.markdown("---")
    st.caption("💡 Powered by LLaMA3, LangChain & Streamlit")

# === Main Chat Window ===
with st.expander("💬 View Full Conversation", expanded=st.session_state.history_expanded):
    for message in st.session_state.conversation_history:
        if message.startswith("User:"):
            with st.chat_message("user"):
                st.markdown(message.replace("User: ", ""))
        elif message.startswith("AI: Final Prompt:"):
            with st.chat_message("assistant"):
                st.success("✅ Final Enhanced Prompt:")
                st.markdown(message.replace("AI: Final Prompt: ", ""))
        elif message.startswith("AI: Updated Prompt:"):
            with st.chat_message("assistant"):
                st.markdown("**Updated Prompt:** " + message.split("Updated Prompt:")[1].split("Clarifying Question:")[0].strip())
                if "Clarifying Question:" in message:
                    st.markdown("**Clarifying Question:** " + message.split("Clarifying Question:")[1].strip())

# === Chat Input ===
user_message = st.chat_input("Type your message here...")

if user_message:
    st.session_state.conversation_history.append("User: " + user_message)
    input_context = st.session_state.input_context
    chain_inputs = {
        "history": "\n".join(st.session_state.conversation_history),
        "user_input": user_message,
        "input_context": input_context,
        "draft_prompt": st.session_state.draft_prompt
    }

    with st.spinner("🤔 Thinking..."):
        try:
            ai_response = chain.invoke(chain_inputs)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            ai_response = None

    if ai_response:
        updated_prompt, clarifying_question, is_final = extract_response_parts(ai_response, st.session_state.draft_prompt)
        if is_final:
            st.session_state.conversation_history.append(f"AI: Final Prompt: {updated_prompt}")
            st.session_state.draft_prompt = updated_prompt
            with st.chat_message("assistant"):
                stream_text(updated_prompt)
        else:
            st.session_state.draft_prompt = updated_prompt
            ai_display = f"Updated Prompt: {updated_prompt}\nClarifying Question: {clarifying_question}" if clarifying_question else f"Updated Prompt: {updated_prompt}"
            st.session_state.conversation_history.append(f"AI: {ai_display}")
            with st.chat_message("assistant"):
                stream_text(ai_display)
        st.rerun()
