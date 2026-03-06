"""
Feedback Handler Module
Manages user feedback storage and retrieval for adaptive learning
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FeedbackType(Enum):
    """Types of user feedback"""
    POSITIVE = "POSITIVE"      
    NEGATIVE = "NEGATIVE"      
    CLICKED = "CLICKED"        
    IGNORED = "IGNORED"        


class FeedbackHandler:
    """
    Handles feedback storage, retrieval, and basic analytics
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize feedback handler
        
        Args:
            db_path: Path to feedback database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize feedback database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            query_intent TEXT NOT NULL,
            weights_used TEXT NOT NULL,
            result_path TEXT NOT NULL,
            result_rank INTEGER NOT NULL,
            result_scores TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_query_intent 
        ON search_feedback(query_intent)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON search_feedback(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def record_feedback(
        self,
        query: str,
        query_intent: str,
        weights_used: Dict[str, float],
        result_path: str,
        result_rank: int,
        result_scores: Dict[str, float],
        feedback_type: FeedbackType
    ) -> bool:
        """
        Record user feedback for a search result
        
        Args:
            query: The search query
            query_intent: Detected intent (METADATA/VISUAL/TEXT/HYBRID)
            weights_used: Weights at search time
            result_path: Path to the result image
            result_rank: Position in results (1-indexed)
            result_scores: Score breakdown {visual, ocr, metadata, final}
            feedback_type: Type of feedback
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO search_feedback 
            (query, query_intent, weights_used, result_path, result_rank, 
             result_scores, feedback_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                query,
                query_intent,
                json.dumps(weights_used),
                result_path,
                result_rank,
                json.dumps(result_scores),
                feedback_type.value
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return False
    
    def get_learning_data(self, intent_type: str, min_samples: int = 10) -> Optional[Dict]:
        """
        Get aggregated learning data for a specific intent type
        
        Args:
            intent_type: Query intent to analyze
            min_samples: Minimum feedback samples required
        
        Returns:
            Dict with learning data or None if insufficient samples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT weights_used, result_scores, feedback_type, result_rank
        FROM search_feedback
        WHERE query_intent = ?
        ORDER BY timestamp DESC
        ''', (intent_type,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < min_samples:
            return None
        
        # Aggregate data
        successful_weights = []
        all_weights = []
        
        for weights_json, scores_json, feedback, rank in rows:
            weights = json.loads(weights_json)
            all_weights.append(weights)
            
            if feedback in ['POSITIVE', 'CLICKED']:
                successful_weights.append(weights)
        
        if not successful_weights:
            return None
        
        success_rate = len(successful_weights) / len(all_weights)
        avg_weights = {
            'visual': sum(w['visual'] for w in successful_weights) / len(successful_weights),
            'ocr': sum(w['ocr'] for w in successful_weights) / len(successful_weights),
            'metadata': sum(w['metadata'] for w in successful_weights) / len(successful_weights),
        }
        
        return {
            'intent': intent_type,
            'total_samples': len(all_weights),
            'successful_samples': len(successful_weights),
            'success_rate': success_rate,
            'avg_successful_weights': avg_weights,
        }
    
    def get_feedback_stats(self) -> Dict:
        """
        Get overall feedback statistics
        
        Returns:
            Dict with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM search_feedback')
        total = cursor.fetchone()[0]
        
        cursor.execute('''
        SELECT query_intent, COUNT(*) 
        FROM search_feedback 
        GROUP BY query_intent
        ''')
        by_intent = dict(cursor.fetchall())
        
        cursor.execute('''
        SELECT feedback_type, COUNT(*) 
        FROM search_feedback 
        GROUP BY feedback_type
        ''')
        by_type = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_feedback': total,
            'by_intent': by_intent,
            'by_type': by_type,
        }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get recent search queries with feedback
        
        Args:
            limit: Maximum number of queries to return
        
        Returns:
            List of query data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT DISTINCT query, query_intent, timestamp
        FROM search_feedback
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {'query': q, 'intent': i, 'timestamp': t}
            for q, i, t in rows
        ]
    
    def get_result_penalty(self, result_path: str, query: str = None) -> float:
        """
        Get penalty score for a specific result based on negative feedback
        
        Args:
            result_path: Path to the result image
            query: Optional query to check (if None, checks all queries)
        
        Returns:
            Penalty multiplier (0.0 to 1.0, where 1.0 = no penalty)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if query:
            cursor.execute('''
            SELECT feedback_type, COUNT(*) as count
            FROM search_feedback
            WHERE result_path = ? AND query = ?
            GROUP BY feedback_type
            ''', (result_path, query))
        else:
            cursor.execute('''
            SELECT feedback_type, COUNT(*) as count
            FROM search_feedback
            WHERE result_path = ?
            GROUP BY feedback_type
            ''', (result_path,))
        
        feedback_counts = dict(cursor.fetchall())
        conn.close()
        
        positive = feedback_counts.get('POSITIVE', 0)
        clicked = feedback_counts.get('CLICKED', 0)
        negative = feedback_counts.get('NEGATIVE', 0)
        
        total_positive = positive + clicked
        
        if negative > 0 and total_positive == 0:
            penalty = max(0.1, 1.0 - (negative * 0.12))  
            return penalty
        
        if negative > 0 and total_positive > 0:
            ratio = negative / (total_positive + negative)
            penalty = max(0.3, 1.0 - ratio * 0.7)  
            return penalty
        
        if total_positive > 0:
            boost = min(1.3, 1.0 + (total_positive * 0.06))  
            return boost
        
        return 1.0  


if __name__ == "__main__":
    handler = FeedbackHandler()
    print(f"Feedback database initialized at: {handler.db_path}")
    
    stats = handler.get_feedback_stats()
    print(f"\nCurrent stats: {stats}")
