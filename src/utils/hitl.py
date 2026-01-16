"""Human-in-the-Loop utilities."""
from typing import Dict, Optional, Callable
from enum import Enum


class HITLTrigger(Enum):
    """Reasons for triggering HITL."""
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    LOW_ASR_CONFIDENCE = "low_asr_confidence"
    PARSER_AMBIGUITY = "parser_ambiguity"
    VERIFIER_UNCERTAINTY = "verifier_uncertainty"
    USER_REQUEST = "user_request"


class HITLHandler:
    """Handle Human-in-the-Loop interactions."""
    
    def __init__(self):
        """Initialize HITL handler."""
        self.pending_approvals = {}
    
    def should_trigger(self, trigger: HITLTrigger, confidence: float, 
                      threshold: float = 0.7) -> bool:
        """Determine if HITL should be triggered."""
        if trigger in [HITLTrigger.USER_REQUEST]:
            return True
        
        return confidence < threshold
    
    def format_hitl_request(self, trigger: HITLTrigger, data: Dict, 
                           context: Optional[str] = None) -> Dict:
        """Format a HITL request for the UI."""
        return {
            "trigger": trigger.value,
            "data": data,
            "context": context,
            "requires_action": True
        }
    
    def process_hitl_response(self, response: Dict) -> Dict:
        """Process user response to HITL request."""
        return {
            "approved": response.get("approved", False),
            "corrections": response.get("corrections", {}),
            "notes": response.get("notes", "")
        }
