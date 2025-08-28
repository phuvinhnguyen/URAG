import requests
from requests.exceptions import RequestException

def get_web_content(url, timeout=10):
    """
    Get HTML content from a URL using requests library
    
    Args:
        url (str): The URL to fetch
        timeout (int): Request timeout in seconds
    
    Returns:
        str: HTML content or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise exception for bad status codes
        
        return response.text
    
    except RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ''

if __name__ == "__main__":
    # Usage
    url = "https://www.google.com"
    html_content = get_web_content(url)
    if html_content:
        print(f"Successfully fetched {len(html_content)} characters")
        print(html_content[:500])  # Print first 500 characters