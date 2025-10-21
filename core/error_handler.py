# Error handling utilities
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Error handling for batch processing."""
    
    ERROR_MESSAGES = {
        'api_error': "API接続エラーが発生しました。しばらく待ってから再試行してください。",
        'timeout': "操作がタイムアウトしました。",
        'molecule_not_found': "分子データが見つかりませんでした。",
        'invalid_data': "無効なデータが返されました。",
        'processing_error': "分子データの処理中にエラーが発生しました。",
        'parse_error': "データの解析に失敗しました。",
        'display_error': "表示中にエラーが発生しました。",
        'no_data': "データが見つかりません。最初からやり直してください。",
        'general_error': "予期しないエラーが発生しました。",
    }
    
    @staticmethod
    def handle_error(e: Exception, error_type: str = "general_error") -> str:
        """Handle all types of errors with simplified messaging."""
        logger.error(f"Error ({error_type}): {str(e)}")
        
        return ErrorHandler.ERROR_MESSAGES.get(error_type, ErrorHandler.ERROR_MESSAGES['general_error'])
    
    @staticmethod
    def log_error(error_type: str, message: str, details: Dict[str, Any] = None):
        """Log error with structured information."""
        error_data = {
            'error_type': error_type,
            'message': message,
            'details': details or {}
        }
        logger.error(f"Error logged: {error_data}")

def is_no_result_response(response_text: str) -> bool:
    """Check if response indicates no results."""
    if not response_text:
        return True
    
    no_result_indicators = [
        "該当なし",
        "見つかりません",
        "該当する分子を思いつかなかった",
        "No results",
        "Not found"
    ]
    
    response_lower = response_text.lower()
    return any(indicator in response_lower for indicator in no_result_indicators)

def parse_json_response(response_text: str) -> Dict[str, Any]:
    """Parse JSON response from Gemini API."""
    import json
    import re
    
    if not response_text:
        return {}
    
    try:
        # Try to extract JSON from response
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning("No JSON found in response")
                return {}
        
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Response text: {response_text[:200]}...")
        return {}
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return {}
