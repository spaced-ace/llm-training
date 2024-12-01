# Evaluation script

This script can be used to score multiple aspects of quality of the LLM generated quiz questions with an LLM as a judge. (gemini-1.5-flash)

## Purpose

The result of training the LLMs with the previous step (../train-llm directory) has to be evaluated, so they can be compared to the original model and to training runs with different configurations.

## Usage

The scripts have a command line interface, and the help message can be printed with the following command:
```bash
pip install -r requirements.txt

python eval_responses.py --help
```
