import streamlit as st
from dotenv import load_dotenv
import re
import time
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import plotly.graph_objects as go
import plotly.express as px

# Load env vars
load_dotenv()

# === Model and Parser ===
llm = ChatOllama(model="llama3.2:latest")
parser = StrOutputParser()

# === Enhanced Prompt Template ===
clarification_prompt_template = ChatPromptTemplate.from_template("""
You are an expert prompt engineering assistant that creates the BEST POSSIBLE AI PROMPTS with strategic guidance.

Conversation History:
{history}

Latest User Message:
{user_input}

Input Context: {input_context}
Prompt Quality Score: {quality_score}/100
User Preferences: {preferences}

Current Prompt Draft:
{draft_prompt}

### Instructions:
1. Analyze the conversation, user message, and context to understand intent
2. Revise the draft prompt to be **specific, actionable, and results-focused**
3. Consider the quality score and user preferences when refining
4. Output the revised prompt as "Updated Prompt:"
5. Provide a **strategic clarifying question** with practical suggestions

🔑 Enhanced Clarifying Question Rules:
- Must be **one focused sentence under 15 words**
- Be **action-oriented** and directly improve prompt effectiveness
- Include **exactly 5 concrete, actionable suggestions** in parentheses
- Suggestions should be **specific and immediately implementable**
- Format: "What specific aspect needs refinement? (e.g., suggestion1, suggestion2, suggestion3, suggestion4, suggestion5)"

### Quality Enhancement Focus:
- **Clarity**: Remove ambiguity, add specific instructions
- **Context**: Include relevant background information
- **Constraints**: Define clear boundaries and limitations
- **Output Format**: Specify desired structure and style
- **Examples**: Add concrete examples when beneficial

If the user says "generate now" or "finalize", output:
Final Prompt: <final optimized prompt>
Quality Score: <estimated score>/100
Key Improvements: <list 3 main enhancements made>

### Format:
Updated Prompt: <revised prompt>
Clarifying Question: <strategic question with 5 actionable suggestions>
""")

chain = clarification_prompt_template | llm | parser

# === Enhanced Helper Functions ===
def classify_input(user_message):
    """Advanced input classification with confidence scoring"""
    keywords = {
        "summarization": ["summary", "summarize", "condense", "brief", "overview"],
        "code": ["code", "programming", "script", "function", "algorithm", "debug"],
        "generation": ["generate", "create", "write", "produce", "compose"],
        "analysis": ["analyze", "insights", "data", "examine", "evaluate"],
        "translation": ["translate", "language", "convert", "interpret"],
        "classification": ["classify", "categorize", "sort", "group", "organize"],
        "extraction": ["extract", "pull", "find", "identify", "locate"],
        "creative": ["story", "creative", "imagination", "fiction", "narrative"]
    }
    
    scores = {}
    for task, words in keywords.items():
        score = sum(1 for word in words if word in user_message.lower())
        if score > 0:
            scores[task] = score
    
    return max(scores, key=scores.get) if scores else "general"

def calculate_quality_score(prompt):
    """Calculate prompt quality based on multiple factors"""
    if not prompt or prompt == "No draft yet.":
        return 0
    
    score = 0
    
    # Length and detail (0-20 points)
    word_count = len(prompt.split())
    if word_count >= 20:
        score += min(20, word_count // 5)
    
    # Specificity indicators (0-25 points)
    specific_words = ["specific", "exactly", "must", "should", "include", "format", "example"]
    score += min(25, sum(5 for word in specific_words if word.lower() in prompt.lower()))
    
    # Structure and formatting (0-20 points)
    structure_indicators = [":", "1.", "2.", "3.", "-", "•", "###", "**"]
    score += min(20, sum(3 for indicator in structure_indicators if indicator in prompt))
    
    # Context and constraints (0-20 points)
    context_words = ["context", "background", "constraint", "limitation", "requirement", "goal"]
    score += min(20, sum(4 for word in context_words if word.lower() in prompt.lower()))
    
    # Output format specification (0-15 points)
    format_words = ["format", "structure", "output", "result", "response", "style"]
    score += min(15, sum(3 for word in format_words if word.lower() in prompt.lower()))
    
    return min(100, score)

def get_improvement_suggestions(current_prompt, context):
    """Generate contextual improvement suggestions"""
    suggestions_db = {
        "general": [
            "Add specific output format requirements",
            "Include relevant background context",
            "Define clear success criteria",
            "Add concrete examples or templates",
            "Specify tone and style preferences"
        ],
        "summarization": [
            "Specify desired summary length",
            "Define key points to emphasize",
            "Choose summary style (bullet points, paragraph, etc.)",
            "Include target audience context",
            "Add format for citations or sources"
        ],
        "code": [
            "Specify programming language and version",
            "Define code structure and naming conventions",
            "Include error handling requirements",
            "Add performance or efficiency constraints",
            "Specify testing and documentation needs"
        ],
        "analysis": [
            "Define analysis methodology or framework",
            "Specify metrics and KPIs to focus on",
            "Include data visualization requirements",
            "Add confidence levels or uncertainty handling",
            "Define actionable insights format"
        ]
    }
    
    return suggestions_db.get(context, suggestions_db["general"])

def extract_response_parts(ai_response, current_draft):
    """Enhanced response parsing with quality scoring"""
    updated_prompt_match = re.search(r"Updated Prompt:\s*(.*?)(?=Clarifying Question:|$)", ai_response, re.IGNORECASE | re.DOTALL)
    clarifying_question_match = re.search(r"Clarifying Question:\s*(.*?)(?=Quality Score:|$)", ai_response, re.IGNORECASE | re.DOTALL)
    final_prompt_match = re.search(r"Final Prompt:\s*(.*?)(?=Quality Score:|Key Improvements:|$)", ai_response, re.IGNORECASE | re.DOTALL)
    quality_score_match = re.search(r"Quality Score:\s*(\d+)", ai_response, re.IGNORECASE)
    improvements_match = re.search(r"Key Improvements:\s*(.*)", ai_response, re.IGNORECASE | re.DOTALL)
    
    if final_prompt_match:
        final_prompt = final_prompt_match.group(1).strip()
        quality_score = quality_score_match.group(1) if quality_score_match else "85"
        improvements = improvements_match.group(1).strip() if improvements_match else "Enhanced clarity and specificity"
        return final_prompt, None, True, quality_score, improvements
    
    updated_prompt = updated_prompt_match.group(1).strip() if updated_prompt_match else current_draft
    clarifying_question = clarifying_question_match.group(1).strip() if clarifying_question_match else None
    
    return updated_prompt, clarifying_question, False, None, None

def stream_text(text, delay=0.01):
    """Enhanced text streaming with better formatting"""
    placeholder = st.empty()
    streamed = ""
    for char in text:
        streamed += char
        placeholder.markdown(streamed)
        time.sleep(delay)

def export_session_data():
    """Export conversation history and prompts"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation_history": st.session_state.conversation_history,
        "final_prompt": st.session_state.draft_prompt,
        "quality_score": st.session_state.get("quality_score", 0),
        "preferences": st.session_state.get("user_preferences", {}),
        "metrics": st.session_state.get("session_metrics", {})
    }
    return json.dumps(export_data, indent=2)

# === Session State Initialization ===
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = "No draft yet."
if "history_expanded" not in st.session_state:
    st.session_state.history_expanded = True
if "quality_score" not in st.session_state:
    st.session_state.quality_score = 0
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}
if "session_metrics" not in st.session_state:
    st.session_state.session_metrics = {"interactions": 0, "improvements": 0}
if "prompt_versions" not in st.session_state:
    st.session_state.prompt_versions = []

# === UI Header ===
st.set_page_config(page_title="AI Prompt Enhancer Pro", page_icon="🧠", layout="wide")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; margin: 0;'>🧠 AI Prompt Enhancement Studio</h1>
    <p style='color: #e0e0e0; margin: 10px 0 0 0;'>Transform your ideas into powerful AI prompts with intelligent guidance</p>
</div>
""", unsafe_allow_html=True)

# === Enhanced Sidebar ===
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Quality Score Display
    current_score = calculate_quality_score(st.session_state.draft_prompt)
    st.session_state.quality_score = current_score
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prompt Quality"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)
    
    # Current Draft Display
    st.markdown("**📝 Current Draft:**")
    with st.expander("View Current Prompt", expanded=False):
        st.code(st.session_state.draft_prompt, language='markdown')
    
    # Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Copy", use_container_width=True):
            st.toast("Copied to clipboard!", icon="✅")
    with col2:
        if st.button("💾 Export", use_container_width=True):
            export_data = export_session_data()
            st.download_button(
                label="Download Session",
                data=export_data,
                file_name=f"prompt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["conversation_history", "draft_prompt", "quality_score", "prompt_versions", "session_metrics"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    
    # Enhanced Settings
    st.markdown("### ⚙️ Preferences")
    
    st.session_state.input_context = st.selectbox(
        "🎯 Prompt Type",
        ["general", "summarization", "code", "generation", "analysis", "translation", "classification", "extraction", "creative"],
        help="Select the type of AI task you're designing the prompt for"
    )
    
    st.session_state.user_preferences["complexity"] = st.select_slider(
        "🎚️ Complexity Level",
        options=["Simple", "Moderate", "Advanced", "Expert"],
        value="Moderate",
        help="How detailed should the prompt be?"
    )
    
    st.session_state.user_preferences["tone"] = st.selectbox(
        "🎭 Tone Style",
        ["Professional", "Casual", "Academic", "Creative", "Technical"],
        help="What tone should the AI use?"
    )
    
    st.session_state.user_preferences["output_format"] = st.multiselect(
        "📄 Preferred Formats",
        ["Structured", "Bullet Points", "Paragraphs", "Code", "Tables", "JSON"],
        default=["Structured"],
        help="What output formats do you typically need?"
    )
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🎯 Focus on Clarity", use_container_width=True):
        st.session_state.conversation_history.append("User: Please focus on making the prompt clearer and more specific")
        st.rerun()
    
    if st.button("🔧 Add Examples", use_container_width=True):
        st.session_state.conversation_history.append("User: Add concrete examples to make the prompt more effective")
        st.rerun()
    
    if st.button("📊 Include Metrics", use_container_width=True):
        st.session_state.conversation_history.append("User: Add success metrics and evaluation criteria")
        st.rerun()
    
    # Session Metrics
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    metrics = st.session_state.session_metrics
    st.metric("Interactions", metrics.get("interactions", 0))
    st.metric("Improvements", metrics.get("improvements", 0))
    st.metric("Current Score", f"{current_score}/100")
    
    st.markdown("---")
    st.caption("💡 Powered by LLaMA3.2 & Advanced Prompt Engineering")

# === Main Content Area ===
col1, col2 = st.columns([2, 1])

with col1:
    # Chat Interface
    st.markdown("### 💬 Interactive Enhancement")
    
    # Quick Start Templates
    with st.expander("🚀 Quick Start Templates", expanded=False):
        templates = {
            "Content Creation": "Create engaging content about [topic] for [audience] in [format]",
            "Data Analysis": "Analyze [data type] to identify [insights] and provide [recommendations]",
            "Code Generation": "Write [language] code that [functionality] with [constraints]",
            "Summarization": "Summarize [content] focusing on [key points] in [length] for [audience]",
            "Problem Solving": "Solve [problem] by [approach] considering [constraints] and [goals]"
        }
        
        for name, template in templates.items():
            if st.button(f"Use {name} Template", key=f"template_{name}"):
                st.session_state.conversation_history.append(f"User: {template}")
                st.rerun()
    
    # Conversation Display
    with st.expander("💬 Conversation History", expanded=st.session_state.history_expanded):
        for i, message in enumerate(st.session_state.conversation_history):
            if message.startswith("User:"):
                with st.chat_message("user"):
                    st.markdown(message.replace("User: ", ""))
            elif message.startswith("AI: Final Prompt:"):
                with st.chat_message("assistant"):
                    st.success("✅ **Final Enhanced Prompt Generated!**")
                    st.markdown(message.replace("AI: Final Prompt: ", ""))
            elif message.startswith("AI: Updated Prompt:"):
                with st.chat_message("assistant"):
                    content = message.replace("AI: ", "")
                    if "Updated Prompt:" in content:
                        prompt_part = content.split("Updated Prompt:")[1].split("Clarifying Question:")[0].strip()
                        st.markdown(f"**🔄 Updated Prompt:**\n{prompt_part}")
                    if "Clarifying Question:" in content:
                        question_part = content.split("Clarifying Question:")[1].strip()
                        st.markdown(f"**❓ Next Step:**\n{question_part}")

with col2:
    # Prompt Analytics
    st.markdown("### 📈 Prompt Analytics")
    
    if st.session_state.prompt_versions:
        # Version History
        st.markdown("**📚 Version History:**")
        for i, version in enumerate(st.session_state.prompt_versions[-5:]):  # Show last 5 versions
            score = calculate_quality_score(version)
            st.markdown(f"v{i+1}: {score}/100")
    
    # Improvement Suggestions
    st.markdown("**💡 Smart Suggestions:**")
    suggestions = get_improvement_suggestions(st.session_state.draft_prompt, st.session_state.input_context)
    for suggestion in suggestions[:3]:  # Show top 3
        if st.button(f"✨ {suggestion}", key=f"suggestion_{suggestion[:20]}"):
            st.session_state.conversation_history.append(f"User: {suggestion}")
            st.rerun()

# === Enhanced Chat Input ===
st.markdown("---")
user_message = st.chat_input("💭 Describe your AI task or ask for improvements... (Type 'generate now' when ready to finalize)")

if user_message:
    # Update metrics
    st.session_state.session_metrics["interactions"] += 1
    
    # Save current prompt version
    if st.session_state.draft_prompt != "No draft yet.":
        st.session_state.prompt_versions.append(st.session_state.draft_prompt)
    
    st.session_state.conversation_history.append("User: " + user_message)
    
    # Prepare enhanced inputs
    chain_inputs = {
        "history": "\n".join(st.session_state.conversation_history),
        "user_input": user_message,
        "input_context": st.session_state.input_context,
        "draft_prompt": st.session_state.draft_prompt,
        "quality_score": st.session_state.quality_score,
        "preferences": str(st.session_state.user_preferences)
    }

    with st.spinner("🤖 Analyzing and enhancing your prompt..."):
        try:
            ai_response = chain.invoke(chain_inputs)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            ai_response = None

    if ai_response:
        result = extract_response_parts(ai_response, st.session_state.draft_prompt)
        updated_prompt, clarifying_question, is_final, quality_score, improvements = result
        
        if is_final:
            st.session_state.conversation_history.append(f"AI: Final Prompt: {updated_prompt}")
            st.session_state.draft_prompt = updated_prompt
            st.session_state.session_metrics["improvements"] += 1
            
            # Show final result with celebration
            st.balloons()
            with st.chat_message("assistant"):
                st.success("🎉 **Final Enhanced Prompt Generated!**")
                st.markdown(f"**Quality Score:** {quality_score}/100")
                if improvements:
                    st.markdown(f"**Key Improvements:** {improvements}")
                st.markdown("---")
                stream_text(updated_prompt)
        else:
            old_score = st.session_state.quality_score
            st.session_state.draft_prompt = updated_prompt
            new_score = calculate_quality_score(updated_prompt)
            
            if new_score > old_score:
                st.session_state.session_metrics["improvements"] += 1
            
            ai_display = f"Updated Prompt: {updated_prompt}"
            if clarifying_question:
                ai_display += f"\nClarifying Question: {clarifying_question}"
            
            st.session_state.conversation_history.append(f"AI: {ai_display}")
            
            with st.chat_message("assistant"):
                st.info(f"📊 Quality Score: {old_score} → {new_score}")
                stream_text(ai_display)
        
        st.rerun()

# === Footer ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Built with advanced prompt engineering techniques | 
    <a href='#' style='color: #667eea;'>Documentation</a> | 
    <a href='#' style='color: #667eea;'>GitHub</a> | 
    <a href='#' style='color: #667eea;'>Feedback</a></p>
</div>
""", unsafe_allow_html=True)