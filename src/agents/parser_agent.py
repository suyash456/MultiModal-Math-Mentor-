"""Parser Agent for structured problem extraction."""
from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.config import GROQ_API_KEY, DEFAULT_LLM_MODEL


class ParsedProblem(BaseModel):
    """Structured representation of a math problem."""
    problem_text: str
    topic: str
    variables: List[str]
    constraints: List[str]
    needs_clarification: bool
    clarification_questions: List[str]
    confidence: float


class ParserAgent:
    """Agent that parses raw input into structured math problems."""
    
    def __init__(self):
        """Initialize the parser agent."""
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Please set it in .env file or environment variables. "
                "Get your API key from https://console.groq.com/"
            )
        self.llm = ChatGroq(
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.1,
            groq_api_key=GROQ_API_KEY
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a math problem parser. Your job is to extract structured information from raw math problem text.

Extract the following:
1. Clean problem text
2. Topic (algebra, probability, calculus, linear_algebra)
3. Variables mentioned
4. Constraints or conditions
5. Whether clarification is needed
6. Questions to ask if clarification is needed

Be precise and identify any ambiguities."""),
            ("human", """Parse this math problem:

{raw_text}

Return a JSON object with:
- problem_text: cleaned version of the problem
- topic: one of [algebra, probability, calculus, linear_algebra]
- variables: list of variable names
- constraints: list of constraints/conditions
- needs_clarification: boolean
- clarification_questions: list of questions if clarification needed
- confidence: float between 0 and 1

Only return valid JSON."""),
        ])
    
    def parse(self, raw_text: str) -> ParsedProblem:
        """
        Parse raw text into structured problem.
        
        Args:
            raw_text: Raw input text from OCR/ASR/typing
            
        Returns:
            ParsedProblem object
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({"raw_text": raw_text})
            
            # Extract JSON from response
            content = response.content
            
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Clean control characters from JSON
            import json
            import re
            # Remove control characters that break JSON parsing
            content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)
            
            # Parse JSON
            parsed_data = json.loads(content)
            
            return ParsedProblem(**parsed_data)
        except Exception as e:
            print(f"Parser error: {e}")
            # Return default structure
            return ParsedProblem(
                problem_text=raw_text,
                topic="algebra",
                variables=[],
                constraints=[],
                needs_clarification=True,
                clarification_questions=["Could you clarify the problem?"],
                confidence=0.0
            )
    
    def needs_hitl(self, parsed: ParsedProblem) -> bool:
        """Determine if human-in-the-loop is needed."""
        return (
            parsed.needs_clarification or
            parsed.confidence < 0.7 or
            len(parsed.problem_text.strip()) < 20
        )
