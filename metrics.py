from typing import Dict, List

class ConformalMetrics:
    """Class for computing conformal prediction metrics."""
    
    @staticmethod
    def compute_lac_score(probabilities: Dict[str, float], correct_answer: str) -> float:
        """
        Compute LAC (Least Ambiguous Classifier) score.
        LAC score = 1 - P(y_true | x)
        
        Args:
            probabilities: Dict mapping options to their probabilities
            correct_answer: The correct answer option
            
        Returns:
            LAC score (lower is better)
        """
        return 1.0 - probabilities.get(correct_answer, 0.0)
    
    @staticmethod
    def compute_aps_score(probabilities: Dict[str, float], correct_answer: str) -> float:
        """
        Compute APS (Adaptive Prediction Sets) score.
        APS score is the cumulative probability mass up to and including the correct answer
        when options are sorted by probability (descending).
        
        Args:
            probabilities: Dict mapping options to their probabilities
            correct_answer: The correct answer option
            
        Returns:
            APS score (lower is better)
        """
        # Sort probabilities in descending order
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        cumulative_prob = 0.0
        for option, prob in sorted_probs:
            cumulative_prob += prob
            if option == correct_answer:
                return cumulative_prob
        
        # If correct answer not found, return 1.0 (worst case)
        return 1.0
    
    @staticmethod
    def compute_prediction_set_lac(probabilities: Dict[str, float], threshold: float) -> List[str]:
        """
        Compute prediction set using LAC threshold.
        
        Args:
            probabilities: Dict mapping options to their probabilities
            threshold: LAC threshold (q_hat)
            
        Returns:
            List of options in the prediction set
        """
        return [option for option, prob in probabilities.items() if prob >= 1 - threshold]
    
    @staticmethod
    def compute_prediction_set_aps(probabilities: Dict[str, float], threshold: float) -> List[str]:
        """
        Compute prediction set using APS threshold.
        
        Args:
            probabilities: Dict mapping options to their probabilities
            threshold: APS threshold (q_hat)
            
        Returns:
            List of options in the prediction set
        """
        # Sort probabilities in descending order
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        prediction_set = []
        cumulative_prob = 0.0
        
        for option, prob in sorted_probs:
            cumulative_prob += prob
            prediction_set.append(option)
            if cumulative_prob >= threshold:
                break
        
        return prediction_set