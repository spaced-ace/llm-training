import os
import json
import tqdm
import torch
import wandb
import sqlite3
import datasets
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    pipeline,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
)
from trl import SFTTrainer

os.environ['HF_HUB_CACHE'] = '/cache'
os.environ['HF_HOME'] = '/cache'


def parse_response(response: str) -> dict | None:
    output = response.split('<output>')[-1].split('</output>')[0]
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(e)
        return None


def create_result_table(conn) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_responses (
            run_name TEXT NOT NULL,
            conversation_index INTEGER NOT NULL,
            messages TEXT NOT NULL,
            question TEXT,
            PRIMARY KEY (run_name, conversation_index)
        )
    """
    )
    cursor.close()


def save_conversation_to_db(
    conn,
    run_name: str,
    convo_index: int,
    convo: list[dict],
    question: dict | None,
) -> None:
    convo_txt = json.dumps(convo, ensure_ascii=False)
    question_txt = json.dumps(question, ensure_ascii=False)
    cursor = conn.cursor()
    if question is not None:
        cursor.execute(
            """
            INSERT INTO chat_responses(run_name, conversation_index, messages, question)
            VALUES (?, ?, ?, ?)
            """,
            (run_name, convo_index, convo_txt, question_txt),
        )
    else:
        cursor.execute(
            """
            INSERT INTO chat_responses(run_name, conversation_index, messages)
            VALUES (?, ?, ?)
            """,
            (run_name, convo_index, convo_txt),
        )
    cursor.close()


def train_model(
    model,
    tokenizer,
    ds: datasets.DatasetDict,
    model_save_path: str,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_path: str | None = None,
):
    model = prepare_model_for_kbit_training(model)
    if lora_path is not None:
        lora_config = PeftConfig.from_pretrained(lora_path)
        model = PeftModel.from_pretrained(model, lora_path, config=lora_config)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=5,
        save_strategy='steps',
        eval_strategy='steps',
        eval_steps=50,
        save_steps=50,
        load_best_model_at_end=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        dataset_text_field='text',
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    trainer.save_model(os.path.join(args.model_save_path, 'last'))
    tokenizer.save_pretrained(os.path.join(args.model_save_path, 'last'))
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_save_path',
        type=str,
        help='path where the modell will be saved',
    )
    parser.add_argument(
        'test_set_save_path',
        type=str,
        help='path where the model responses will be saved (sqlite file)',
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only use the model for prediciton on the eval set',
    )
    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Only use the model for training on the training set',
    )
    parser.add_argument(
        '--run_name',
        type=str,
        help='name of the run (will be logged to wandb)',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='name of the dataset to load from huggingface',
        default='jazzysnake01/quizgen-chat-sm',
    )
    parser.add_argument(
        '--model_name',
        required=False,
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='the repo id of the model that will be trained',
    )
    parser.add_argument(
        '--lora_path',
        type=str,
        required=False,
        help='Path to a pretrained lora adapter (overrides other lora config options)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of epoch to train',
    )
    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='Lora rank',
    )
    parser.add_argument(
        '--lora_a',
        type=int,
        default=16,
        help='Lora alpha',
    )
    args = parser.parse_args()
    wandb.login()

    ds = datasets.load_dataset(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    for key in ds.keys():
        ds[key] = ds[key].map(
            lambda x: {
                'text': tokenizer.apply_chat_template(
                    x['formatted_conversation'], tokenize=False
                )
            }
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,  #
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
        ),
    )

    run_name = None
    if not args.eval_only:
        if args.run_name is None:
            wandb.init(
                project='diploma',
            )
            run_name = wandb.run.name
        else:
            wandb.init(
                project='diploma',
                name=args.run_name,
            )
            run_name = args.run_name
        model = train_model(
            model,
            tokenizer,
            ds,
            args.model_save_path,
            args.epochs,
            args.lora_r,
            args.lora_a,
            args.lora_path,
        )
        wandb.finish()
    elif args.eval_only and args.lora_path is not None:
        lora_config = PeftConfig.from_pretrained(args.lora_path)
        model = PeftModel.from_pretrained(
            model, args.lora_path, config=lora_config
        )
    if args.train_only:
        exit()

    if run_name is None:
        if args.run_name is None:
            run_name = f'eval_only_{args.model_name}'
        else:
            run_name = args.run_name

    chat_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype='float16',
    )
    chat_pipeline.tokenizer.pad_token_id = model.config.eos_token_id

    def get_data():
        for conversation in ds['test']:
            messages = conversation['formatted_conversation']
            messages_without_last = messages[:-1]
            yield tokenizer.apply_chat_template(
                messages_without_last, tokenize=False
            )

    conn = sqlite3.connect(args.test_set_save_path, isolation_level=None)
    create_result_table(conn)
    for index, response in tqdm.tqdm(
        enumerate(
            chat_pipeline(get_data(), max_new_tokens=500),
        ),
        total=len(ds['test']),
    ):
        res = response[0]['generated_text']
        context = res.split('<context>')[-1]
        context = context.split('</context>')[0]
        parsed_resp = parse_response(res.split('</context>')[-1])
        if parsed_resp is not None:
            parsed_resp['context'] = context
        messages_without_last = ds['test'][index]
        messages_without_last = messages_without_last[
            'formatted_conversation'
        ][:-1]
        new_convo = messages_without_last + [
            {'role': 'assistant', 'content': response}
        ]
        save_conversation_to_db(conn, run_name, index, new_convo, parsed_resp)
        conn.commit()
    conn.close()
