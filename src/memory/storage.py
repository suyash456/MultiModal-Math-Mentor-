"""Memory storage for learning from past solutions."""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.config import MEMORY_DB_PATH


class MemoryStorage:
    """Store and retrieve problem-solving history."""
    
    def __init__(self):
        """Initialize memory storage."""
        self.db_path = Path(MEMORY_DB_PATH)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_type TEXT,
                raw_input TEXT,
                parsed_problem TEXT,
                topic TEXT,
                variables TEXT,
                timestamp DATETIME,
                UNIQUE(raw_input, timestamp)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER,
                solution_steps TEXT,
                final_answer TEXT,
                method_used TEXT,
                confidence REAL,
                timestamp DATETIME,
                FOREIGN KEY (problem_id) REFERENCES problems(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER,
                is_correct BOOLEAN,
                user_comment TEXT,
                corrected_answer TEXT,
                timestamp DATETIME,
                FOREIGN KEY (solution_id) REFERENCES solutions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER,
                retrieved_chunks TEXT,
                sources TEXT,
                timestamp DATETIME,
                FOREIGN KEY (problem_id) REFERENCES problems(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT,
                corrected_text TEXT,
                confidence REAL,
                timestamp DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_problem(self, input_type: str, raw_input: str, parsed_problem: Dict) -> int:
        """Store a problem and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO problems (input_type, raw_input, parsed_problem, topic, variables, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            input_type,
            raw_input,
            json.dumps(parsed_problem),
            parsed_problem.get("topic", ""),
            json.dumps(parsed_problem.get("variables", [])),
            datetime.now().isoformat()
        ))
        
        problem_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return problem_id
    
    def store_solution(self, problem_id: int, solution: Dict) -> int:
        """Store a solution and return its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO solutions (problem_id, solution_steps, final_answer, method_used, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            problem_id,
            json.dumps(solution.get("solution_steps", [])),
            solution.get("final_answer", ""),
            solution.get("method_used", ""),
            solution.get("confidence", 0.0),
            datetime.now().isoformat()
        ))
        
        solution_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return solution_id
    
    def store_feedback(self, solution_id: int, is_correct: bool, 
                      user_comment: str = "", corrected_answer: str = "") -> int:
        """Store user feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (solution_id, is_correct, user_comment, corrected_answer, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            solution_id,
            is_correct,
            user_comment,
            corrected_answer,
            datetime.now().isoformat()
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def store_rag_context(self, problem_id: int, retrieved_chunks: List[Dict], sources: List[str]):
        """Store RAG context used for a problem."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rag_context (problem_id, retrieved_chunks, sources, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            problem_id,
            json.dumps(retrieved_chunks),
            json.dumps(sources),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def store_ocr_correction(self, original_text: str, corrected_text: str, confidence: float):
        """Store OCR correction for learning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ocr_corrections (original_text, corrected_text, confidence, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            original_text,
            corrected_text,
            confidence,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_similar_problems(self, topic: str, variables: List[str], limit: int = 5) -> List[Dict]:
        """Retrieve similar problems from memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        variables_json = json.dumps(variables)
        
        cursor.execute("""
            SELECT p.id, p.raw_input, p.parsed_problem, p.topic,
                   s.solution_steps, s.final_answer, s.method_used,
                   f.is_correct
            FROM problems p
            LEFT JOIN solutions s ON p.id = s.problem_id
            LEFT JOIN feedback f ON s.id = f.solution_id
            WHERE p.topic = ? AND f.is_correct = 1
            ORDER BY p.timestamp DESC
            LIMIT ?
        """, (topic, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "raw_input": row[1],
                "parsed_problem": json.loads(row[2]),
                "topic": row[3],
                "solution_steps": json.loads(row[4]) if row[4] else [],
                "final_answer": row[5],
                "method_used": row[6],
                "is_correct": bool(row[7])
            })
        
        conn.close()
        return results
