import httpx
import bs4
from bs4 import BeautifulSoup
import json
from tqdm.auto import tqdm

def extract_text_recursive(element):
    text = []

    # If element is None, return empty text
    if element is None:
        return text

    # Check if the element is a NavigableString (text within an element)
    if isinstance(element, bs4.NavigableString):
        return text

    # Check if the element is a paragraph or header of interest
    if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        t = element.get_text(separator=" ", strip=True)
        if element.name[0] == 'h':
            t = int(element.name[1]) * '#' + ' ' + t
        text.append(t)

    if element.children is None:
        return text
    # Recursively process child elements
    for child in element.children:
        text += extract_text_recursive(child)

    return text

def extract_text_from_wikipedia_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = extract_text_recursive(soup.body)
    return text


def remove_references(text):
    result = []
    skip = 0  # Variable to track whether to skip the next character
    for i, char in enumerate(text):
        if char == '[' and i < len(text) - 1 and text[i + 1].isdigit():
            # If the current character is '[' and the next character is a digit
            skip = 1  # Set skip to 1 to skip the '[' and the following digits
        elif char == ']' and skip:
            skip = 0  # Reset skip to 0 to skip the ']'
        elif not skip:
            # If skip is 0, append the character to the result
            result.append(char)
    return ''.join(result)

def remove_edit_notes(text):
    result = []
    skip = 0  # Variable to track whether to skip the next character
    for i, char in enumerate(text):
        if char == '[' and i < len(text) - 9 and text[i+1:i+8]=='szerkes':
            # If the current character is '[' and the next character is a digit
            skip = 1  # Set skip to 1 to skip the '[' and the following digits
        elif char == ']' and skip:
            skip = 0  # Reset skip to 0 to skip the ']'
        elif not skip:
            # If skip is 0, append the character to the result
            result.append(char)
    return ''.join(result)

def clean_article_text(texts: list[str]) -> str:
    meaningful_text = []
    for element in texts:
        if element == '':
            continue
        if "References" in element or "External links" in element or "Further reading" in element or "See also" in element:
            break
        if "Irodalom " in element or "Kapcsolódó szócikkek" in element or "Külső hivatkozások" in element or "További információk" in element or "Források" in element or "További információk" in element:
            break
        text = remove_references(element)
        text = remove_edit_notes(text)
        meaningful_text.append(text)
    return meaningful_text

important_articles_page = 'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/2'

page = httpx.get(important_articles_page)
bs = BeautifulSoup(page.text, 'html.parser')
sections = bs.find_all('table', class_='col-begin')
articles = []
for section in sections:
    for link in section.find_all('a'):
        href = link.get('href')
        # not articles
        if ':' in href:
            continue
        articles.append(href)

base_url = 'https://en.wikipedia.org'
article_texts = []
for article in tqdm(articles, desc='Downloading articles'):
    article_url = f'{base_url}{article}'
    html = httpx.get(article_url)
    bs = BeautifulSoup(html.text, 'html.parser')
    a = bs.find_all(name='a', attrs={'class':'interlanguage-link-target', 'hreflang':'hu'})
    text_hu = []
    url_hu = None
    if a:
        url_hu = a[0].get('href')
        html_hu = httpx.get(url_hu)
        text_hu = extract_text_from_wikipedia_html(html_hu.text)
    text = extract_text_from_wikipedia_html(html.text)
    article_texts.append({'url':article_url, 'en':text, 'url_hu':url_hu, 'hu':text_hu})

articles = []

for article in tqdm(article_texts, desc='Cleaning articles'):
    cleaned_text = clean_article_text(article['en'])
    cleaned_text_hu = clean_article_text(article['hu'])
    articles.append({'url':article['url'], 'sections_en':cleaned_text, 'sections_hu':cleaned_text_hu, 'url_hu':article['url_hu']})
json.dump(articles, open('important_articles.json', 'w'), indent=2)
