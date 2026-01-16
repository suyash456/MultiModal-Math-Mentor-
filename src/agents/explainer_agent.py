"""Explainer/Tutor Agent for step-by-step explanations."""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.utils.config import GROQ_API_KEY, DEFAULT_LLM_MODEL
from src.agents.parser_agent import ParsedProblem


class ExplainerAgent:
    """Agent that provides student-friendly explanations."""
    
    def __init__(self):
        """Initialize the explainer agent."""
        self.llm = ChatGroq(
            model_name=DEFAULT_LLM_MODEL,
            temperature=0.3,
            groq_api_key=GROQ_API_KEY
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a math tutor. Explain solutions in a clear, step-by-step manner that helps students learn.

Your explanations should:
- Be easy to understand
- Show reasoning at each step
- Connect concepts
- Use examples when helpful
- Be encouraging

Write as if teaching a student."""),
            ("human", """Explain this solution to a student:

Problem: {problem_text}
Topic: {topic}

Solution Steps:
{solution_steps}

Final Answer: {final_answer}

Verification: {verification_notes}

Create a clear, educational explanation. Return JSON with:
- explanation: full step-by-step explanation
- key_concepts: list of concepts used
- tips: helpful tips for similar problems
- practice_suggestions: suggestions for practice

Only return valid JSON."""),
        ])
    
    def explain(self, parsed_problem: ParsedProblem, solution: Dict, verification: Dict) -> Dict:
        """
        Generate explanation for the solution.
        
        Args:
            parsed_problem: Original parsed problem
            solution: Solution from solver
            verification: Verification result
            
        Returns:
            Explanation dictionary
        """
        try:
            chain = self.prompt_template | self.llm
            response = chain.invoke({
                "problem_text": parsed_problem.problem_text,
                "topic": parsed_problem.topic,
                "solution_steps": "\n".join(solution.get("solution_steps", [])),
                "final_answer": solution.get("final_answer", ""),
                "verification_notes": verification.get("verification_notes", "")
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
            # Remove control characters except newlines and tabs in string values
            # This is a simple approach - for production, consider using a proper JSON sanitizer
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)  # Remove control chars
            
            # Try to parse JSON
            try:
                explanation = json.loads(content)
            except json.JSONDecodeError as e:
                # If parsing fails, try to extract JSON object more carefully
                print(f"JSON parse error: {e}, attempting to fix...")
                # Try to find JSON object boundaries
                start_idx = content.find('{')
                end_idx = content.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = content[start_idx:end_idx+1]
                    # Remove control characters again
                    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
                    try:
                        explanation = json.loads(content)
                    except json.JSONDecodeError:
                        # Last resort: return a structured response even if JSON is malformed
                        explanation = {
                            "explanation": content if content else "Error generating explanation",
                            "key_concepts": [],
                            "tips": [],
                            "practice_suggestions": []
                        }
            
            return explanation
        except Exception as e:
            print(f"Explainer error: {e}")
            return {
                "explanation": "Error generating explanation",
                "key_concepts": [],
                "tips": [],
                "practice_suggestions": []
            }
