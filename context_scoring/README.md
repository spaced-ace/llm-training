# Scoring contexts for quality assessment

This script can be used to assign a quality score to the scraped wikipedia paragraphs.

## Purpose

Some of the quiz questions were lower quality and in general, these questions were paired with low quality contexts. To achieve higher quality training data, quality scores were assigned to the contexts, which will later be used to filter out low quality data.

## Usage

The script has a command line interface, and the help message can be printed with the following command:
```bash
pip install -r requirements.txt

python context_scoring.py --help
```
