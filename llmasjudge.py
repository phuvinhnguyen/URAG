from typing import List, Tuple, Set, Union

def correct_prediction(predictions: List[str], 
                      labels: List[Union[str, List[str]]],
                      question: str = None,
                      ) -> Tuple[List[int], Set[str]]:
    """
    Evaluate predictions against labels using available backends.
    
    Returns:
        Tuple of (correctness_list, correct_predictions_set)
    """
    # Using exact matching before using llm judge
    correctness = []
    correct_predictions = set()

    if isinstance(labels, str): labels = [labels]
    for pred in predictions:
        for label in labels:
            if pred.lower().strip() == label.lower().strip():
                correctness.append(1)
                correct_predictions.add(pred)
                break
        else:
            correctness.append(0)

    if len(''.join(predictions)) == len(predictions) and ((isinstance(labels, List) and len(labels) == len(''.join(labels))) or (isinstance(labels, str) and len(labels) == 1)):
        return correctness, correct_predictions
    
    raise RuntimeError("All backends are disabled")

# Example usage
if __name__ == "__main__":
    predictions = ["Paris is the capital", "London", "Tokyo is in Japan"]
    labels = ["Paris", ["London", "UK capital"], "Tokyo"]
    
    correctness, correct_set = correct_prediction(predictions, labels)
    print(f"Correctness: {correctness}")
    print(f"Correct predictions: {correct_set}")