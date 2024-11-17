# SpacedAce quiz generation LLM trainig

This repository contains all steps necessary for reproduction of the trained LLM. 

## Steps to reproduce

### 1. Data scraping and preprocessing

The data was scraped from wikipedia's most vital articles list (level 2). These articles are used
to attain context from which questions will be generated.

### 2. Question generation

A larger more capable model's output is used for generating quiz questions, that will serve as a
reference for the model in training.

### 3. Context scoring

The quality of some of the questions was subpar. It was noticed that this is usually the case of
bad contexts. In an attempt to filter out lower quality questions, the quality of the contexts was
evaluated and subsequently the dataset can be filtered.

### 4. LLM training

Lora was used to finetune Llama-3-8B-Instruct for question generation on the generated data.

### 5. LLM response evaluation

Gemini-1.5-flash was tasked with preference scoring the generated questions to evaluate the quality
of the questions on a multidimensional scale.

## Repository structure

Each step of the training, from scraping to response evaluation has its own separate subdirectory.
The subfolders contain further information about their contents. For more information, consult
the README.md in each directory.

1. Data scraping and preprocessing - "scraping" directory
2. Question generation - "question_generation" directory
3. Context scoring - "context_scoring" directory
4. LLM training - "train-llm" directory
5. LLM response evaluation - "eval" directory

