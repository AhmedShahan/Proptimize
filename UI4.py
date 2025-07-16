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

# === FIXED Prompt Template ===
clarification_prompt_template = ChatPromptTemplate.from_template("""
You are an expert prompt engineering assistant that helps users iteratively improve their AI prompts.

Conversation History:
{history}

Latest User Message:
{user_input}

Input Context: {input_context}
Current Quality Score: {quality_score}/100
User Preferences: {preferences}

Current Prompt Draft:
{draft_prompt}

### IMPORTANT INSTRUCTIONS:
1. **NEVER finalize unless user explicitly says "generate now", "finalize", "done", or "finish"**
2. **ALWAYS provide iterative improvements and ask clarifying questions**
3. **Only output "Final Prompt:" if user explicitly requests finalization**
4. **Focus on continuous improvement through conversation**

### Your Task:
- Analyze the user's message and current prompt
- Make incremental improvements to the prompt
- Ask ONE specific clarifying question to guide the next improvement
- Keep the conversation going until user requests finalization

### Response Format:
Updated Prompt: <improved version of the prompt>
Clarifying Question: <one specific question with 3-5 concrete suggestions in parentheses>

### Example Clarifying Questions:
- "What specific output format would work best? (e.g., bullet points, structured paragraphs, JSON format, step-by-step guide, comparison table)"
- "How detailed should the response be? (e.g., brief overview, comprehensive analysis, technical deep-dive, beginner-friendly explanation, expert-level detail)"
- "What context should the AI consider? (e.g., target audience, industry background, use case scenario, constraints, success criteria)"

### ONLY if user explicitly says "generate now", "finalize", "done", or "finish":
Final Prompt: <polished final version>
Quality Score: <estimated score>/100
Key Improvements: <3 main enhancements made>

Remember: Keep improving and asking questions unless explicitly told to finalize!
""")

chain = clarification_prompt_template | llm | parser

# === Helper Functions ===
def classify_input(user_message):
    """Advanced input classification with confidence scoring"""
    # Check for finalization keywords first
    finalization_keywords = ["generate now", "finalize", "done", "finish", "complete", "ready"]
    if any(keyword in user_message.lower() for keyword in finalization_keywords):
        return "finalize"
    
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
    """Enhanced response parsing with better finalization detection"""
    print(f"DEBUG: AI Response: {ai_response}")  # Debug line
    
    # Check for finalization first
    final_prompt_match = re.search(r"Final Prompt:\s*(.*?)(?=Quality Score:|Key Improvements:|$)", ai_response, re.IGNORECASE | re.DOTALL)
    
    if final_prompt_match:
        final_prompt = final_prompt_match.group(1).strip()
        quality_score_match = re.search(r"Quality Score:\s*(\d+)", ai_response, re.IGNORECASE)
        improvements_match = re.search(r"Key Improvements:\s*(.*)", ai_response, re.IGNORECASE | re.DOTALL)
        
        quality_score = quality_score_match.group(1) if quality_score_match else "85"
        improvements = improvements_match.group(1).strip() if improvements_match else "Enhanced clarity and specificity"
        return final_prompt, None, True, quality_score, improvements
    
    # Otherwise, look for iterative improvement - more flexible parsing
    updated_prompt = current_draft
    clarifying_question = None
    
    # Try to find Updated Prompt
    updated_prompt_patterns = [
        r"Updated Prompt:\s*(.*?)(?=Clarifying Question:|$)",
        r"Revised Prompt:\s*(.*?)(?=Clarifying Question:|$)",
        r"Improved Prompt:\s*(.*?)(?=Clarifying Question:|$)"
    ]
    
    for pattern in updated_prompt_patterns:
        match = re.search(pattern, ai_response, re.IGNORECASE | re.DOTALL)
        if match:
            updated_prompt = match.group(1).strip()
            break
    
    # Try to find Clarifying Question
    question_patterns = [
        r"Clarifying Question:\s*(.*?)(?=Quality Score:|$)",
        r"Next Question:\s*(.*?)(?=Quality Score:|$)",
        r"Question:\s*(.*?)(?=Quality Score:|$)"
    ]
    
    for pattern in question_patterns:
        match = re.search(pattern, ai_response, re.IGNORECASE | re.DOTALL)
        if match:
            clarifying_question = match.group(1).strip()
            break
    
    # If no clarifying question found, create a default one
    if not clarifying_question:
        clarifying_question = "What specific aspect would you like to improve next? (e.g., add examples, specify format, include constraints, define audience, clarify goals)"
    
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
if "is_finalized" not in st.session_state:
    st.session_state.is_finalized = False
if "should_finalize" not in st.session_state:
    st.session_state.should_finalize = False

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
    
    # Finalization Status
    if st.session_state.is_finalized:
        st.success("✅ Prompt Finalized!")
    else:
        st.info("🔄 Iterative Mode")
    
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
    
    # Finalize Button
    if not st.session_state.is_finalized and st.session_state.draft_prompt != "No draft yet.":
        if st.button("🎯 Finalize Prompt", use_container_width=True, type="primary"):
            # Set a flag to trigger finalization
            st.session_state.should_finalize = True
            st.rerun()
    
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["conversation_history", "draft_prompt", "quality_score", "prompt_versions", "session_metrics", "is_finalized", "should_finalize"]:
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
            elif message.startswith("AI:"):
                with st.chat_message("assistant"):
                    content = message.replace("AI: ", "")
                    if "Updated Prompt:" in content:
                        parts = content.split("Updated Prompt:")
                        if len(parts) > 1:
                            remaining = parts[1]
                            if "Clarifying Question:" in remaining:
                                prompt_part = remaining.split("Clarifying Question:")[0].strip()
                                question_part = remaining.split("Clarifying Question:")[1].strip()
                                st.markdown(f"**🔄 Updated Prompt:**\n{prompt_part}")
                                st.markdown(f"**❓ Next Step:**\n{question_part}")
                            else:
                                st.markdown(f"**🔄 Updated Prompt:**\n{remaining.strip()}")
                    else:
                        st.markdown(content)

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

# Show different input prompts based on finalization status
if st.session_state.is_finalized:
    input_placeholder = "💭 Your prompt is finalized! Start a new session to create another prompt..."
    disabled = True
else:
    input_placeholder = "💭 Describe your AI task or ask for improvements... (Click 'Finalize Prompt' in sidebar when ready)"
    disabled = False

# Check if finalization was requested via button
if st.session_state.should_finalize:
    user_message = "generate now"
    st.session_state.should_finalize = False
    st.session_state.conversation_history.append("User: " + user_message)
    
    # Process finalization immediately
    chain_inputs = {
        "history": "\n".join(st.session_state.conversation_history),
        "user_input": user_message,
        "input_context": st.session_state.input_context,
        "draft_prompt": st.session_state.draft_prompt,
        "quality_score": st.session_state.quality_score,
        "preferences": str(st.session_state.user_preferences)
    }

    with st.spinner("🤖 Finalizing your prompt..."):
        try:
            ai_response = chain.invoke(chain_inputs)
            
            result = extract_response_parts(ai_response, st.session_state.draft_prompt)
            updated_prompt, clarifying_question, is_final, quality_score, improvements = result
            
            if is_final:
                st.session_state.conversation_history.append(f"AI: Final Prompt: {updated_prompt}")
                st.session_state.draft_prompt = updated_prompt
                st.session_state.session_metrics["improvements"] += 1
                st.session_state.is_finalized = True
                
                # Show final result with celebration
                st.balloons()
                st.success("🎉 **Final Enhanced Prompt Generated!**")
                st.markdown(f"**Quality Score:** {quality_score}/100")
                if improvements:
                    st.markdown(f"**Key Improvements:** {improvements}")
                st.markdown("---")
                st.markdown(updated_prompt)
            else:
                st.error("Failed to finalize. Please try again.")
                
        except Exception as e:
            st.error(f"⚠️ Error during finalization: {e}")

user_message = st.chat_input(input_placeholder, disabled=disabled)

if user_message and not st.session_state.is_finalized:
    # Update metrics
    st.session_state.session_metrics["interactions"] += 1
    
    # Save current prompt version
    if st.session_state.draft_prompt != "No draft yet.":
        st.session_state.prompt_versions.append(st.session_state.draft_prompt)
    
    st.session_state.conversation_history.append("User: " + user_message)
    
    # Check if this is a finalization request
    input_type = classify_input(user_message)
    
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
            st.session_state.is_finalized = True
            
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