# Training scripts

These scripts can be used to fine-tune an LLM for quiz question generation with LoRA.

## Purpose

- dataset\_to\_chat\_format.py is responsible for converting the generated+scored questions to chat format suitable for training the LLM. It can optionally be used to limit the inclusion of data with lower assigned quality scores.
- train\_llm.py is responsible for running the training with the chat format data. It optionally (on by default) saves the responses given by the LLM to a sqlite database on disk.
- load\_and\_merge.py is responsible for loading model to memory and merging the trained adapter back to the weight matrix. It can be used to push to model to huggingface.

## Usage

The scripts have a command line interface, and the help message can be printed with the following command:
```bash
pip install -r requirements.txt

python dataset_to_chat_format.py --help
# or 
python train_llm.py --help
# or 
python load_and_merge.py --help
```
