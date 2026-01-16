"""Solver Agent for solving math problems using RAG and tools."""
from typing import Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from sympy import symbols, solve, diff, integrate, limit, simplify
from sympy.parsing.sympy_parser import parse_expr
from src.utils.config import GROQ_API_KEY, DEFAULT_LLM_MODEL
from src.agents.parser_agent import ParsedProblem


class SolverAgent:
    """Agent that solves math problems using RAG context and symbolic tools."""
    
    def __init__(self, rag_context: Optional[str] = None):
        """
        Initialize the solver agent.
        
        Args:
            rag_context: Retrieved context from RAG pipeline
        """
        self.llm = ChatGroq(
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.2,
            groq_api_key=GROQ_API_KEY
        )
        self.rag_context = rag_context or ""
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a math problem solver. Use the provided context and tools to solve problems step by step.

Available tools:
- SymPy for symbolic math
- Python calculator for numerical computation

Always show your work and reasoning."""),
            ("human", """Solve this math problem:

Problem: {problem_text}
Topic: {topic}
Variables: {variables}
Constraints: {constraints}

Relevant context:
{rag_context}

Solve step by step. Show all work. Return JSON with:
- solution_steps: list of step descriptions
- final_answer: the final answer
- method_used: description of method
- confidence: float between 0 and 1

Only return valid JSON."""),
        ])
    
    def solve(self, parsed_problem: ParsedProblem, routing: Dict) -> Dict:
        """
        Solve the math problem.
        
        Args:
            parsed_problem: Parsed problem structure
            routing: Routing information from router
            
        Returns:
            Solution dictionary
        """
        try:
            # Try symbolic solution first if applicable
            symbolic_solution = self._try_symbolic_solve(parsed_problem)
            
            # Use LLM for complex reasoning
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "topic": parsed_problem.topic,
                "variables": ", ".join(parsed_problem.variables),
                "constraints": ", ".join(parsed_problem.constraints),
                "rag_context": self.rag_context
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
            
            solution = json.loads(content)
            
            # Merge symbolic solution if available
            if symbolic_solution:
                solution["symbolic_solution"] = symbolic_solution
                solution["confidence"] = min(1.0, solution.get("confidence", 0.7) + 0.2)
            
            return solution
        except Exception as e:
            print(f"Solver error: {e}")
            return {
                "solution_steps": ["Error solving problem"],
                "final_answer": "Unable to solve",
                "method_used": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _try_symbolic_solve(self, parsed_problem: ParsedProblem) -> Optional[Dict]:
        """Attempt symbolic solution using SymPy."""
        try:
            # This is a simplified version - would need more sophisticated parsing
            problem_lower = parsed_problem.problem_text.lower()
            
            if "solve" in problem_lower or "find" in problem_lower:
                # Try to extract equation
                if "=" in parsed_problem.problem_text:
                    parts = parsed_problem.problem_text.split("=")
                    if len(parts) == 2:
                        # Simple symbolic solving
                        var_names = parsed_problem.variables if parsed_problem.variables else ["x"]
                        syms = symbols(" ".join(var_names))
                        
                        # This would need more sophisticated parsing
                        # For now, return None to use LLM
                        return None
            
            return None
        except Exception as e:
            print(f"Symbolic solve error: {e}")
            return None
