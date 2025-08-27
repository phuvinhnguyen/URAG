import re
from io import BytesIO
from markitdown import MarkItDown
from bs4 import BeautifulSoup
import html

def clean_web_content(html_content, use_markitdown=True):
    """
    Clean web content and convert to markdown or plain text.
    
    Args:
        html_content (str): Raw HTML content to clean
        use_markitdown (bool): Whether to use MarkItDown for conversion
    
    Returns:
        str: Cleaned content in markdown or plain text format
    """
    
    if use_markitdown:
        # Method 1: Using MarkItDown library
        try:
            from io import BytesIO
            md = MarkItDown()
            # Create a file-like object from the HTML content
            html_stream = BytesIO(html_content.encode('utf-8'))
            result = md.convert_stream(html_stream, file_extension='.html')
            return result.text_content
        except Exception as e:
            print(f"MarkItDown conversion failed: {e}")
            print("Falling back to BeautifulSoup method...")
            use_markitdown = False
    
    if not use_markitdown:
        # Method 2: Using BeautifulSoup for manual cleaning
        return clean_with_beautifulsoup(html_content)

def clean_with_beautifulsoup(html_content):
    """
    Clean HTML content using BeautifulSoup.
    
    Args:
        html_content (str): Raw HTML content
    
    Returns:
        str: Cleaned text content
    """
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted elements
    unwanted_tags = [
        'script', 'style', 'meta', 'link', 'noscript', 'iframe',
        'embed', 'object', 'applet', 'form', 'input', 'button',
        'select', 'textarea', 'nav', 'header', 'footer', 'aside'
    ]
    
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    
    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
        comment.extract()
    
    # Convert common HTML elements to markdown-like format
    # Headers
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            header.string = f"{'#' * i} {header.get_text().strip()}\n\n"
    
    # Paragraphs
    for p in soup.find_all('p'):
        if p.get_text().strip():
            p.string = f"{p.get_text().strip()}\n\n"
    
    # Lists
    for ul in soup.find_all('ul'):
        items = []
        for li in ul.find_all('li'):
            items.append(f"- {li.get_text().strip()}")
        if items:
            ul.string = '\n'.join(items) + '\n\n'
    
    for ol in soup.find_all('ol'):
        items = []
        for i, li in enumerate(ol.find_all('li'), 1):
            items.append(f"{i}. {li.get_text().strip()}")
        if items:
            ol.string = '\n'.join(items) + '\n\n'
    
    # Links
    for a in soup.find_all('a'):
        href = a.get('href', '')
        text = a.get_text().strip()
        if text and href:
            a.string = f"[{text}]({href})"
    
    # Bold and italic
    for b in soup.find_all(['b', 'strong']):
        b.string = f"**{b.get_text().strip()}**"
    
    for i in soup.find_all(['i', 'em']):
        i.string = f"*{i.get_text().strip()}*"
    
    # Extract clean text
    clean_text = soup.get_text()
    
    # Post-processing cleanup
    clean_text = html.unescape(clean_text)
    
    # Remove excessive whitespace
    clean_text = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_text)
    clean_text = re.sub(r'[ \t]+', ' ', clean_text)
    clean_text = re.sub(r'\n[ \t]+', '\n', clean_text)
    
    # Remove empty lines at the beginning and end
    clean_text = clean_text.strip()
    
    return clean_text

def clean_web_content_advanced(html_content):
    """
    Advanced cleaning with additional filters for web-specific content.
    
    Args:
        html_content (str): Raw HTML content
    
    Returns:
        str: Cleaned and formatted text
    """
    
    # First pass with basic cleaning
    cleaned = clean_web_content(html_content)
    
    # Additional cleaning patterns for web content
    patterns_to_remove = [
        r'window\..*?;',  # JavaScript window objects
        r'document\..*?;',  # JavaScript document objects
        r'var\s+\w+\s*=.*?;',  # JavaScript variable declarations
        r'function\s*\([^)]*\)\s*\{[^}]*\}',  # JavaScript functions
        r'https?://[^\s<>"{}|\\^`\[\]]*',  # URLs (optional - remove if you want to keep links)
        r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\b',  # ISO timestamps
        r'consent|cookie|privacy policy|terms of service',  # Common web notices (case insensitive)
    ]
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Final cleanup
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

if __name__ == "__main__":
    html_content = '''...'''

    # Method 1: Using MarkItDown (recommended)
    print("=== Using MarkItDown ===")
    try:
        cleaned_markitdown = clean_web_content(html_content, use_markitdown=True)
        print(cleaned_markitdown)
    except ImportError:
        print("MarkItDown not available. Install with: pip install markitdown")

    print("\n=== Using BeautifulSoup (fallback) ===")
    # Method 2: Using BeautifulSoup fallback
    cleaned_bs = clean_web_content(html_content, use_markitdown=False)
    print(cleaned_bs)

    print("\n=== Advanced Cleaning ===")
    # Method 3: Advanced cleaning
    cleaned_advanced = clean_web_content_advanced(html_content)
    print(cleaned_advanced)

# To use this script:
# 1. Install required packages:
#    pip install markitdown beautifulsoup4 html5lib
# 
# 2. Use the functions:
#    cleaned_text = clean_web_content(your_html_content)
#    print(cleaned_text)