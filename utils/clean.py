from bs4 import BeautifulSoup
import re

def clean_web_content(html_source):
    """
    Cleans HTML content by extracting and processing text content.
    
    Parameters:
        html_source (str): HTML content to be cleaned
        
    Returns:
        str: Cleaned text content extracted from HTML
    """
    if not html_source or not isinstance(html_source, str):
        return ""
    
    try:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_source, "lxml")
        
        # Remove unwanted elements (scripts, styles, etc.)
        for element in soup(["script", "style", "meta", "link", "head", "title"]):
            element.decompose()
        
        # Get text content with space as separator and strip whitespaces
        text = soup.get_text(" ", strip=True)
        
        if not text:
            return ""
        
        # Clean up the text - remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
        text = text.strip()  # Remove leading/trailing whitespace
        
        return text
        
    except Exception as e:
        print(f"Error cleaning HTML content: {e}")
        return ""