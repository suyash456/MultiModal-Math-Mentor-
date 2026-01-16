"""Intent Router Agent for problem classification and workflow routing."""
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.config import GROQ_API_KEY, DEFAULT_LLM_MODEL
from src.agents.parser_agent import ParsedProblem


class RouterAgent:
    """Agent that classifies problems and routes workflow."""
    
    def __init__(self):
        """Initialize the router agent."""
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
            ("system", """You are a math problem router. Classify problems and determine the best solution approach.

Topics: algebra, probability, calculus, linear_algebra
Difficulty: easy, medium, hard
Solution approach: direct_calculation, symbolic_manipulation, numerical_method, proof

Be precise in classification."""),
            ("human", """Classify this math problem:

Problem: {problem_text}
Topic: {topic}
Variables: {variables}

Return JSON with:
- topic: confirmed topic
- difficulty: easy/medium/hard
- solution_approach: best approach
- requires_tools: boolean (e.g., calculator, symbolic solver)
- estimated_steps: number of steps needed

Only return valid JSON."""),
        ])
    
    def route(self, parsed_problem: ParsedProblem) -> Dict:
        """
        Route problem to appropriate solution workflow.
        
        Args:
            parsed_problem: Parsed problem structure
            
        Returns:
            Routing information dictionary
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "topic": parsed_problem.topic,
                "variables": ", ".join(parsed_problem.variables)
            })
            
            content = response.content
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Clean control characters from JSON
            import json
            import re
            # Remove control characters that break JSON parsing
            content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)
            
            routing = json.loads(content)
            
            return routing
        except Exception as e:
            print(f"Router error: {e}")
            return {
                "topic": parsed_problem.topic,
                "difficulty": "medium",
                "solution_approach": "direct_calculation",
                "requires_tools": False,
                "estimated_steps": 3
            }
