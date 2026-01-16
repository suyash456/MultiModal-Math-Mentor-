"""Verifier/Critic Agent for solution validation."""
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.config import GROQ_API_KEY, DEFAULT_LLM_MODEL, VERIFIER_CONFIDENCE_THRESHOLD
from src.agents.parser_agent import ParsedProblem


class VerifierAgent:
    """Agent that verifies solution correctness and quality."""
    
    def __init__(self):
        """Initialize the verifier agent."""
        self.llm = ChatGroq(
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.1,
            groq_api_key=GROQ_API_KEY
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a math solution verifier. Check solutions for:
1. Correctness
2. Unit consistency
3. Domain constraints
4. Edge cases
5. Logical consistency

Be thorough and critical."""),
            ("human", """Verify this solution:

Original Problem: {problem_text}
Topic: {topic}
Variables: {variables}
Constraints: {constraints}

Solution Steps:
{solution_steps}

Final Answer: {final_answer}

Check:
1. Is the answer correct?
2. Are units consistent?
3. Do constraints hold?
4. Are there edge cases?
5. Is the logic sound?

Return JSON with:
- is_correct: boolean
- confidence: float between 0 and 1
- issues: list of issues found
- warnings: list of warnings
- needs_hitl: boolean (if verification is uncertain)
- verification_notes: detailed notes

Only return valid JSON."""),
        ])
    
    def verify(self, parsed_problem: ParsedProblem, solution: Dict) -> Dict:
        """
        Verify the solution.
        
        Args:
            parsed_problem: Original parsed problem
            solution: Solution from solver
            
        Returns:
            Verification result dictionary
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "topic": parsed_problem.topic,
                "variables": ", ".join(parsed_problem.variables),
                "constraints": ", ".join(parsed_problem.constraints),
                "solution_steps": "\n".join(solution.get("solution_steps", [])),
                "final_answer": solution.get("final_answer", "")
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
            
            verification = json.loads(content)
            
            # Determine if HITL is needed
            verification["needs_hitl"] = (
                verification.get("needs_hitl", False) or
                verification.get("confidence", 0.0) < VERIFIER_CONFIDENCE_THRESHOLD or
                not verification.get("is_correct", False)
            )
            
            return verification
        except Exception as e:
            print(f"Verifier error: {e}")
            return {
                "is_correct": False,
                "confidence": 0.0,
                "issues": [f"Verification error: {e}"],
                "warnings": [],
                "needs_hitl": True,
                "verification_notes": "Error during verification"
            }
