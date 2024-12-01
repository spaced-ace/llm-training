# Scraping and cleaning source material

Wikipedia's _Most vital articles list_ can be scraped with this script. The script collects the english and hungarian articles.

## Purpose

This was used to collect source material, for which quiz questions can be generated via an LLM. This data will be used for finetuning LLMs for quiz question generation.

## Usage

```bash
pip install -r requirements.txt

python scrape_important_articles.py
```
This will create a file named important\_articles.json. Manual cleaning was performed on this file with regexes in vim. During this procedure common remarks like [more information needed] were removed.
The file containing the cleaned content was named important\_articles\_cleaned.json

Following this, the header elements needed to be connected with their paragraphs and trailing headers with no paragraphs need to be dropped. This was done via clean_json.py, which tries to read important\_articles\_cleaned.json.
It can be run with:

```bash
python clean_json.py
```

The output will be a file named important\_articles\_cleaned\_unified.json.
