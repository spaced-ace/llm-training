# Question generation via google's models

Quiz questions were generated for the scraped wikipedia paragraphs. The data can be acquired via the scripts in the ../scraping directory.

## Purpose

The quiz questions will act as training data for llama-3-8b.

## Usage

The script has a command line interface, and the help message can be printed with the follwing command:
```bash
pip install -r requirements.txt

python generate_questions.py --help
```

The script utilizes the json (resulting from the scripts in ../scraping), generates questions to individual jsons in a specified folder.

After the questions have been generated postproces\_data.py can be used to create a huggingface datasets style dataset. For instructions run:
```bash
python postprocess_data.py --help
```

