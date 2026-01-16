"""Main Streamlit application for Multimodal Math Mentor."""
import streamlit as st
import json
from typing import Dict, Optional
import traceback
from datetime import datetime
from pathlib import Path

# Load .env file from project root BEFORE importing config
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    try:
        # Load with UTF-8 encoding
        load_dotenv(env_path, override=True, encoding='utf-8')
    except Exception as e:
        # Fallback: try without encoding specification
        try:
            load_dotenv(env_path, override=True)
        except Exception:
            print(f"Warning: Could not load .env file: {e}")
else:
    # Try current directory
    try:
        load_dotenv(override=True, encoding='utf-8')
    except Exception:
        pass

# Import components
from src.multimodal.image_processor import ImageProcessor
from src.multimodal.audio_processor import AudioProcessor
from src.multimodal.text_processor import TextProcessor
from src.agents.parser_agent import ParserAgent, ParsedProblem
from src.agents.router_agent import RouterAgent
from src.agents.solver_agent import SolverAgent
from src.agents.verifier_agent import VerifierAgent
from src.agents.explainer_agent import ExplainerAgent
from src.rag.retriever import RAGRetriever
from src.memory.storage import MemoryStorage
from src.memory.retrieval import MemoryRetriever
from src.utils.hitl import HITLHandler, HITLTrigger


# Page configuration
st.set_page_config(
    page_title="Multimodal Math Mentor",
    page_icon="📚",
    layout="wide"
)

# Initialize session state with error handling
if "image_processor" not in st.session_state:
    try:
        st.session_state.image_processor = ImageProcessor()
    except Exception as e:
        st.error(f"Error initializing ImageProcessor: {e}")
        st.session_state.image_processor = None

if "audio_processor" not in st.session_state:
    try:
        st.session_state.audio_processor = AudioProcessor()
    except Exception as e:
        st.warning(f"AudioProcessor initialization warning: {e}")
        st.session_state.audio_processor = None

if "text_processor" not in st.session_state:
    st.session_state.text_processor = TextProcessor()

if "parser_agent" not in st.session_state:
    try:
        st.session_state.parser_agent = ParserAgent()
    except Exception as e:
        st.error(f"Error initializing ParserAgent: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.parser_agent = None

if "router_agent" not in st.session_state:
    try:
        st.session_state.router_agent = RouterAgent()
    except Exception as e:
        st.error(f"Error initializing RouterAgent: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.router_agent = None

if "rag_retriever" not in st.session_state:
    try:
        st.session_state.rag_retriever = RAGRetriever()
    except Exception as e:
        st.warning(f"RAGRetriever initialization warning: {e}")
        st.info("RAG features may be limited, but the app will still work for basic problem solving.")
        # Create a minimal retriever that returns empty results
        try:
            st.session_state.rag_retriever = RAGRetriever()
        except:
            st.session_state.rag_retriever = None

if "memory_storage" not in st.session_state:
    try:
        st.session_state.memory_storage = MemoryStorage()
    except Exception as e:
        st.error(f"Error initializing MemoryStorage: {e}")
        st.session_state.memory_storage = None

if "memory_retriever" not in st.session_state:
    try:
        st.session_state.memory_retriever = MemoryRetriever()
    except Exception as e:
        st.warning(f"MemoryRetriever initialization warning: {e}")
        st.info("Memory features may be limited, but the app will still work.")
        # Create a minimal retriever
        try:
            st.session_state.memory_retriever = MemoryRetriever()
        except:
            st.session_state.memory_retriever = None

if "hitl_handler" not in st.session_state:
    st.session_state.hitl_handler = HITLHandler()

# Initialize workflow state
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"
if "raw_input" not in st.session_state:
    st.session_state.raw_input = ""
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "parsed_problem" not in st.session_state:
    st.session_state.parsed_problem = None
if "solution" not in st.session_state:
    st.session_state.solution = None
if "verification" not in st.session_state:
    st.session_state.verification = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "rag_context" not in st.session_state:
    st.session_state.rag_context = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []
if "needs_hitl" not in st.session_state:
    st.session_state.needs_hitl = False
if "hitl_data" not in st.session_state:
    st.session_state.hitl_data = None


def add_to_trace(agent_name: str, action: str, result: Dict):
    """Add entry to agent trace."""
    st.session_state.agent_trace.append({
        "agent": agent_name,
        "action": action,
        "result": result,
        "timestamp": datetime.now().isoformat()
    })


def process_image_input(uploaded_file):
    """Process image input."""
    try:
        if st.session_state.image_processor is None:
            st.error("ImageProcessor not initialized. Please refresh the page.")
            return "", 0.0, True
            
        text, confidence, needs_hitl = st.session_state.image_processor.process_uploaded_image(uploaded_file)
        
        st.session_state.extracted_text = text
        st.session_state.needs_hitl = needs_hitl
        
        if needs_hitl:
            st.session_state.hitl_data = {
                "trigger": HITLTrigger.LOW_OCR_CONFIDENCE.value,
                "data": {"text": text, "confidence": confidence},
                "message": f"OCR confidence is low ({confidence:.2f}). Please review and correct the extracted text."
            }
        
        add_to_trace("ImageProcessor", "OCR Extraction", {
            "text": text,
            "confidence": confidence,
            "needs_hitl": needs_hitl
        })
        
        return text, confidence, needs_hitl
    except Exception as e:
        st.error(f"Error processing image: {e}")
        import traceback
        st.code(traceback.format_exc())
        return "", 0.0, True


def process_audio_input(uploaded_file):
    """Process audio input."""
    try:
        if st.session_state.audio_processor is None:
            st.error("AudioProcessor not initialized. Please refresh the page.")
            return "", 0.0, True
        
        if st.session_state.audio_processor.model is None:
            st.warning("⚠️ Whisper model not loaded. Audio processing may not work.")
            st.info("The app will continue, but you may need to type the problem manually.")
            return "", 0.0, True
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading audio file...")
        progress_bar.progress(10)
        
        status_text.text("Transcribing audio (this may take a moment)...")
        progress_bar.progress(30)
        
        text, confidence, needs_hitl, metadata = st.session_state.audio_processor.process_uploaded_audio(uploaded_file)
        
        progress_bar.progress(90)
        status_text.text("Processing complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a moment
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Normalize math phrases
        if text:
            text = st.session_state.audio_processor.normalize_math_phrases(text)
        
        st.session_state.extracted_text = text
        st.session_state.needs_hitl = needs_hitl
        
        if needs_hitl:
            # Show the transcript even if confidence is low
            if text:
                st.info(f"📝 Transcript extracted: \"{text}\"")
            st.session_state.hitl_data = {
                "trigger": HITLTrigger.LOW_ASR_CONFIDENCE.value,
                "data": {"text": text, "confidence": confidence, "metadata": metadata},
                "message": f"ASR confidence is low ({confidence:.2%}). Please review and correct the transcript."
            }
        elif text:
            # If we got text with reasonable confidence, show success
            st.success(f"✓ Audio transcribed successfully (confidence: {confidence:.2%})")
        else:
            # No text extracted at all
            st.warning("⚠️ No text could be extracted from the audio. Please try again or type the problem manually.")
        
        add_to_trace("AudioProcessor", "ASR Transcription", {
            "text": text,
            "confidence": confidence,
            "needs_hitl": needs_hitl,
            "metadata": metadata
        })
        
        return text, confidence, needs_hitl
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        import traceback
        st.code(traceback.format_exc())
        return "", 0.0, True


def process_text_input(text: str):
    """Process text input."""
    is_valid, error_msg = st.session_state.text_processor.validate_text(text)
    
    if not is_valid:
        st.error(error_msg)
        return False
    
    cleaned_text = st.session_state.text_processor.clean_text(text)
    st.session_state.extracted_text = cleaned_text
    st.session_state.needs_hitl = False
    
    add_to_trace("TextProcessor", "Text Validation", {
        "text": cleaned_text,
        "is_valid": is_valid
    })
    
    return True


def run_parser_agent(text: str) -> Optional[ParsedProblem]:
    """Run parser agent."""
    try:
        parsed = st.session_state.parser_agent.parse(text)
        st.session_state.parsed_problem = parsed
        
        needs_hitl = st.session_state.parser_agent.needs_hitl(parsed)
        if needs_hitl:
            st.session_state.needs_hitl = True
            st.session_state.hitl_data = {
                "trigger": HITLTrigger.PARSER_AMBIGUITY.value,
                "data": parsed.model_dump(),
                "message": "Parser detected ambiguity. Please clarify the problem.",
                "clarification_questions": parsed.clarification_questions
            }
        
        add_to_trace("ParserAgent", "Problem Parsing", parsed.model_dump())
        
        return parsed
    except Exception as e:
        st.error(f"Parser error: {e}")
        traceback.print_exc()
        return None


def run_router_agent(parsed: ParsedProblem) -> Dict:
    """Run router agent."""
    try:
        routing = st.session_state.router_agent.route(parsed)
        add_to_trace("RouterAgent", "Problem Routing", routing)
        return routing
    except Exception as e:
        st.error(f"Router error: {e}")
        return {}


def run_rag_retrieval(parsed: ParsedProblem) -> list:
    """Run RAG retrieval."""
    try:
        query = f"{parsed.problem_text} {parsed.topic}"
        retrieved = st.session_state.rag_retriever.retrieve(query)
        st.session_state.rag_context = retrieved
        add_to_trace("RAGRetriever", "Context Retrieval", {
            "num_chunks": len(retrieved),
            "sources": [r["source"] for r in retrieved]
        })
        return retrieved
    except Exception as e:
        st.error(f"RAG retrieval error: {e}")
        return []


def run_solver_agent(parsed: ParsedProblem, routing: Dict, rag_context: list) -> Dict:
    """Run solver agent."""
    try:
        # Format RAG context
        context_text = "\n".join([chunk["content"] for chunk in rag_context])
        
        solver = SolverAgent(rag_context=context_text)
        solution = solver.solve(parsed, routing)
        st.session_state.solution = solution
        
        add_to_trace("SolverAgent", "Problem Solving", {
            "method": solution.get("method_used", ""),
            "confidence": solution.get("confidence", 0.0)
        })
        
        return solution
    except Exception as e:
        st.error(f"Solver error: {e}")
        traceback.print_exc()
        return {}


def run_verifier_agent(parsed: ParsedProblem, solution: Dict) -> Dict:
    """Run verifier agent."""
    try:
        verifier = VerifierAgent()
        verification = verifier.verify(parsed, solution)
        st.session_state.verification = verification
        
        if verification.get("needs_hitl", False):
            st.session_state.needs_hitl = True
            st.session_state.hitl_data = {
                "trigger": HITLTrigger.VERIFIER_UNCERTAINTY.value,
                "data": verification,
                "message": "Verifier is uncertain about the solution. Please review."
            }
        
        add_to_trace("VerifierAgent", "Solution Verification", verification)
        
        return verification
    except Exception as e:
        st.error(f"Verifier error: {e}")
        return {}


def run_explainer_agent(parsed: ParsedProblem, solution: Dict, verification: Dict) -> Dict:
    """Run explainer agent."""
    try:
        explainer = ExplainerAgent()
        explanation = explainer.explain(parsed, solution, verification)
        st.session_state.explanation = explanation
        
        add_to_trace("ExplainerAgent", "Solution Explanation", {
            "key_concepts": explanation.get("key_concepts", [])
        })
        
        return explanation
    except Exception as e:
        st.error(f"Explainer error: {e}")
        return {}


def store_in_memory(input_type: str, raw_input: str, parsed: ParsedProblem, 
                   solution: Dict, verification: Dict):
    """Store problem-solving session in memory."""
    try:
        # Store problem
        problem_id = st.session_state.memory_storage.store_problem(
            input_type, raw_input, parsed.model_dump()
        )
        
        # Store solution
        solution_id = st.session_state.memory_storage.store_solution(problem_id, solution)
        
        # Store RAG context
        st.session_state.memory_storage.store_rag_context(
            problem_id, st.session_state.rag_context,
            [r["source"] for r in st.session_state.rag_context]
        )
        
        return problem_id, solution_id
    except Exception as e:
        st.error(f"Memory storage error: {e}")
        return None, None


def main():
    """Main application."""
    # Check if critical components are initialized
    if st.session_state.parser_agent is None:
        st.error("⚠️ **Initialization Error**")
        st.error("ParserAgent failed to initialize. Please check:")
        st.code("""
1. Ensure GROQ_API_KEY is set in .env file
2. Install langchain-core: pip install langchain-core
3. Check terminal for detailed error messages
        """)
        st.stop()
    
    if st.session_state.router_agent is None:
        st.error("⚠️ **Initialization Error**")
        st.error("RouterAgent failed to initialize. Please check your configuration.")
        st.stop()
    
    st.title("📚 Multimodal Math Mentor")
    st.markdown("Solve JEE-style math problems with step-by-step explanations")
    
    # Sidebar for input mode selection
    with st.sidebar:
        st.header("Input Mode")
        input_mode = st.radio(
            "Choose input method:",
            ["Text", "Image", "Audio"],
            index=["Text", "Image", "Audio"].index(st.session_state.input_mode)
        )
        st.session_state.input_mode = input_mode
        
        st.header("Settings")
        show_trace = st.checkbox("Show Agent Trace", value=True, key="show_trace_checkbox")
        show_rag = st.checkbox("Show RAG Context", value=True, key="show_rag_checkbox")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input")
        
        # Input handling based on mode
        if input_mode == "Text":
            text_input = st.text_area(
                "Enter math problem:",
                value=st.session_state.extracted_text,
                height=100
            )
            if st.button("Process Text"):
                if process_text_input(text_input):
                    st.session_state.raw_input = text_input
                    st.rerun()
        
        elif input_mode == "Image":
            uploaded_file = st.file_uploader(
                "Upload image (JPG/PNG):",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                if st.button("Process Image"):
                    text, confidence, needs_hitl = process_image_input(uploaded_file)
                    st.session_state.raw_input = "image_upload"
                    st.rerun()
        
        elif input_mode == "Audio":
            uploaded_file = st.file_uploader(
                "Upload audio or record:",
                type=["wav", "mp3", "m4a", "ogg"]
            )
            if uploaded_file is not None:
                if st.button("Process Audio", disabled=st.session_state.get("processing_audio", False)):
                    # Prevent multiple simultaneous processing
                    if not st.session_state.get("processing_audio", False):
                        st.session_state.processing_audio = True
                        try:
                            with st.spinner("🔄 Processing audio... This may take 10-30 seconds. Please wait..."):
                                text, confidence, needs_hitl = process_audio_input(uploaded_file)
                                st.session_state.raw_input = "audio_upload"
                                st.session_state.processing_audio = False
                                st.rerun()
                        except Exception as e:
                            st.session_state.processing_audio = False
                            st.error(f"❌ Error processing audio: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                    else:
                        st.info("⏳ Audio processing already in progress... Please wait.")
        
        # Show extracted text if available
        if st.session_state.extracted_text:
            st.subheader("Extracted Text")
            edited_text = st.text_area(
                "Review and edit if needed:",
                value=st.session_state.extracted_text,
                height=100,
                key="extracted_text_editor"
            )
            st.session_state.extracted_text = edited_text
            
            if st.button("Solve Problem"):
                with st.spinner("Processing..."):
                    try:
                        # Run pipeline
                        parsed = run_parser_agent(edited_text)
                        
                        # If parser succeeded (even with HITL), proceed with solution
                        if parsed:
                            # If HITL was triggered, show warning but allow continuation
                            if st.session_state.needs_hitl:
                                st.warning("⚠️ Ambiguity detected, but proceeding with solution...")
                            
                            # Proceed with solution pipeline
                            routing = run_router_agent(parsed)
                            rag_context = run_rag_retrieval(parsed)
                            solution = run_solver_agent(parsed, routing, rag_context)
                            verification = run_verifier_agent(parsed, solution)
                            explanation = run_explainer_agent(parsed, solution, verification)
                            
                            # Store in memory
                            store_in_memory(
                                input_mode.lower(),
                                edited_text,
                                parsed,
                                solution,
                                verification
                            )
                            
                            # Clear HITL flag after successful solve
                            st.session_state.needs_hitl = False
                            
                            st.rerun()
                        else:
                            st.error("Failed to parse the problem. Please check the input and try again.")
                    except Exception as e:
                        st.error(f"Error solving problem: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.header("Status")
        
        if st.session_state.parsed_problem:
            st.success("✓ Problem Parsed")
            st.info(f"Topic: {st.session_state.parsed_problem.topic}")
        
        if st.session_state.solution:
            st.success("✓ Solution Generated")
            confidence = st.session_state.solution.get("confidence", 0.0)
            st.metric("Confidence", f"{confidence:.2%}")
        
        if st.session_state.verification:
            is_correct = st.session_state.verification.get("is_correct", False)
            if is_correct:
                st.success("✓ Verified")
            else:
                st.warning("⚠ Needs Review")
    
    # HITL Section
    if st.session_state.needs_hitl and st.session_state.hitl_data:
        st.warning("⚠ Human-in-the-Loop Required")
        with st.expander("HITL Request", expanded=True):
            st.write(st.session_state.hitl_data.get("message", ""))
            
            # For ASR, show editable transcript
            if st.session_state.hitl_data.get("trigger") == HITLTrigger.LOW_ASR_CONFIDENCE.value:
                current_text = st.session_state.extracted_text or st.session_state.hitl_data.get("data", {}).get("text", "")
                st.write("**Current Transcript:**")
                if current_text:
                    st.code(current_text)
                else:
                    st.warning("No transcript was extracted. Please type the problem manually.")
                
                corrected_text = st.text_area(
                    "Edit the transcript if needed (or type the problem if no transcript was extracted):",
                    value=current_text,
                    height=100,
                    key="hitl_transcript_editor",
                    placeholder="Type or paste the math problem here..."
                )
                st.session_state.extracted_text = corrected_text
                
                if st.button("Use Corrected Transcript"):
                    if corrected_text.strip():
                        st.session_state.needs_hitl = False
                        st.success("✓ Transcript updated. You can now click 'Solve Problem' below.")
                        st.rerun()
                    else:
                        st.error("Please enter a transcript or problem text.")
            
            if st.session_state.hitl_data.get("clarification_questions"):
                st.write("**Clarification Questions:**")
                for q in st.session_state.hitl_data["clarification_questions"]:
                    st.write(f"- {q}")
            
            if st.button("Continue Anyway"):
                st.session_state.needs_hitl = False
                # If we have parsed problem and extracted text, proceed with solution
                if st.session_state.parsed_problem and st.session_state.extracted_text:
                    with st.spinner("Processing with current interpretation..."):
                        try:
                            parsed = st.session_state.parsed_problem
                            routing = run_router_agent(parsed)
                            rag_context = run_rag_retrieval(parsed)
                            solution = run_solver_agent(parsed, routing, rag_context)
                            verification = run_verifier_agent(parsed, solution)
                            explanation = run_explainer_agent(parsed, solution, verification)
                            
                            # Store in memory
                            store_in_memory(
                                st.session_state.input_mode.lower(),
                                st.session_state.extracted_text,
                                parsed,
                                solution,
                                verification
                            )
                        except Exception as e:
                            st.error(f"Error solving problem: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                st.rerun()
    
    # Solution Display
    if st.session_state.solution:
        st.header("Solution")
        
        # Final Answer
        st.subheader("Final Answer")
        st.success(st.session_state.solution.get("final_answer", "N/A"))
        
        # Solution Steps
        if st.session_state.solution.get("solution_steps"):
            st.subheader("Solution Steps")
            for i, step in enumerate(st.session_state.solution["solution_steps"], 1):
                st.write(f"**Step {i}:** {step}")
        
        # Explanation
        if st.session_state.explanation:
            st.subheader("Explanation")
            st.markdown(st.session_state.explanation.get("explanation", ""))
            
            if st.session_state.explanation.get("key_concepts"):
                st.write("**Key Concepts:**")
                for concept in st.session_state.explanation["key_concepts"]:
                    st.write(f"- {concept}")
        
        # Feedback Section
        st.subheader("Feedback")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("✅ Correct", type="primary"):
                if st.session_state.solution:
                    # Store positive feedback
                    st.success("Thank you for your feedback!")
        with col_fb2:
            if st.button("❌ Incorrect"):
                feedback_comment = st.text_input("Please provide details:")
                if feedback_comment:
                    # Store negative feedback
                    st.success("Thank you for your feedback!")
    
    # RAG Context Panel
    if show_rag and st.session_state.rag_context:
        st.header("Retrieved Context")
        for i, chunk in enumerate(st.session_state.rag_context, 1):
            with st.expander(f"Source {i}: {chunk.get('title', 'Unknown')}"):
                st.write(chunk["content"])
                st.caption(f"Source: {chunk.get('source', 'N/A')}")
    
    # Agent Trace
    if show_trace and st.session_state.agent_trace:
        st.header("Agent Trace")
        for entry in st.session_state.agent_trace:
            with st.expander(f"{entry['agent']}: {entry['action']}"):
                st.json(entry["result"])


if __name__ == "__main__":
    main()
