import requests

from typing import List, Dict, Optional

from logger_config import logger

# Global dataset filter - set by systems that need to filter search results
_current_dataset: Optional[str] = None


def set_search_dataset(dataset: Optional[str]):
    """
    Set the dataset filter for subsequent searches.
    
    Args:
        dataset: Dataset name to filter by (e.g., 'scifact', 'healthver', 'crag_task_1_and_2')
                 Set to None to search all datasets.
    """
    global _current_dataset
    _current_dataset = dataset
    if dataset:
        logger.info(f"E5 search will filter to dataset: {dataset}")
    else:
        logger.info("E5 search will query all datasets")


def get_search_dataset() -> Optional[str]:
    """Get the current dataset filter."""
    return _current_dataset


def search_by_http(
    query: str, 
    host: str = 'localhost', 
    port: int = 8090,
    k: int = 20,
    dataset: Optional[str] = None,
    timeout: float = 30.0
) -> List[Dict]:
    """
    Search documents via the E5 HTTP server.
    
    Args:
        query: Search query string
        host: E5 server host
        port: E5 server port
        k: Number of results to return
        dataset: Optional dataset name to filter results (e.g., 'scifact', 'crag_task_1_and_2')
                 If None, uses the global _current_dataset setting
        timeout: Request timeout in seconds
        
    Returns:
        List of document results with scores
    """
    url = f"http://{host}:{port}"
    
    # Use provided dataset or fall back to global setting
    effective_dataset = dataset if dataset is not None else _current_dataset
    
    payload = {
        'query': query,
        'k': k
    }
    
    # Add dataset filter if specified
    if effective_dataset:
        payload['dataset'] = effective_dataset
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get a response. Status code: {response.status_code}")
            return []
    except requests.exceptions.Timeout:
        logger.warning(f"E5 search timed out after {timeout}s for query: {query[:50]}...")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"E5 search request failed: {e}")
        return []
