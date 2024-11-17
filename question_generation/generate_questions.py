import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from typing import Annotated
from tqdm.auto import tqdm

import httpx
import pydantic

import dotenv

class MultipleChoice(pydantic.BaseModel):
    question: Annotated[str, pydantic.Field(description='The question text')]
    answers: Annotated[list[str], pydantic.Field(description='The 4 possible answers')]
    solution: Annotated[list[str], pydantic.Field(description='The correct answers (1-4)')]

class SingleChoice(pydantic.BaseModel):
    question: Annotated[str, pydantic.Field(description='The question text')]
    answers: Annotated[list[str], pydantic.Field(description='The 4 possible answers')]
    solution: Annotated[str, pydantic.Field(description='The correct answer')]

class TrueOrFalse(pydantic.BaseModel):
    question: Annotated[str, pydantic.Field(description='The question text')]
    solution: Annotated[bool, pydantic.Field(description='The correct answer')]


harm_categories = [
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    'HARM_CATEGORY_HARASSMENT',
]

TEMP_DIR = 'temp'
QUESTION_TYPES=['single_choice', 'multiple_choice', 'true_or_false']

def pydantic_to_google_json_schema(pydantic_schema: pydantic.BaseModel) -> dict:
    pydantic_schema = pydantic_schema.schema()
    schema = recusively_remove_keys(dict(pydantic_schema), ['title'])
    return schema

def recusively_remove_keys(d: dict, keys: list[str]) -> dict:
    for k in keys:
        d.pop(k, None)
    for k, v in d.items():
        if isinstance(v, dict):
            recusively_remove_keys(v, keys)
    return d

async def make_request(session, api_key, model, text, typ):
    schema = {
        'single_choice': pydantic_to_google_json_schema(SingleChoice),
        'multiple_choice': pydantic_to_google_json_schema(MultipleChoice),
        'true_or_false': pydantic_to_google_json_schema(TrueOrFalse),
    }
    # gemini-1.5-flash-latest, in preview, announced on 05/16/2024
    # 2024-feb is the latest version of the gemini-pro as of now
    #model = 'gemini-1.0-pro-latest'
    if model not in ['gemini-1.0-pro-latest', 'gemini-1.5-flash-latest']:
        exit('Invalid model')
    schema = schema[typ]
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key='
        + api_key
    )
    gen_conf = {
            'temperature':0.6,
            'responseMimeType':'application/json',
            'responseSchema':schema,
        } if 'flash' in model else {'temperature':0.7,}
    headers = {'Content-Type': 'application/json'}
    payload = {
        'safetySettings': [
            {
                'category': cat,
                'threshold': 'BLOCK_NONE',
            }
            for cat in harm_categories
        ],
        'generationConfig':gen_conf,
        'contents': [{'parts': [{'text': text}]}],
        
    }

    response = await session.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def convert_text_to_prompt(text, question_type='single_choice', en=True) -> str:
    if question_type == 'single_choice':
        type_prompt = "Create a single choice question with 4 options."
        type_example = """Magyarország állam Közép-Európában, a Kárpát-medence közepén. 1989 óta parlamentáris köztársaság. Északról Szlovákia, északkeletről Ukrajna, keletről és délkeletről Románia, délről Szerbia, délnyugatról Horvátország és Szlovénia, nyugatról pedig Ausztria határolja.
        Output:
        {"question": "Melyik ország Magyarország északi szomszédja?", "answers": ["Szlovénia", "Szlovákia", "Ukrajna", "Ausztria"], "solution": "B"}
        """ if not en else"""By the early 1990s, however, relational systems dominated in all large-scale data processing applications, and as of 2018 they remain dominant: IBM Db2, Oracle, MySQL, and Microsoft SQL Server are the most searched DBMS.
        Output:
        {"question": "What was the most dominant type of DMBS system in the 1990s?", "answers": ["NoSQL", "Objet-oriented", "Document", "Relational"], "solution": "D"}
        """
    elif question_type == 'multiple_choice':
        type_prompt = "Create a multiple choice question with 4 options. Use A, B, C, D to mark the correct answers and try to create questions with 2-4 correct answers."
        type_example = """By the early 1990s, however, relational systems dominated in all large-scale data processing applications, and as of 2018 they remain dominant: IBM Db2, Oracle, MySQL, and Microsoft SQL Server are the most searched DBMS.
        Output:
        {"question": "What are some of the most dominant RDMBS systems?", "answers": ["IBM Db2", "Oracle", "MySQL", "MongoDB"], "solution": ["A", "B", "C"]}
        """ if en else """A miskolci Városház tér a város közigazgatási központja, itt áll a Városháza és a Megyeháza.
        Output:
        {"question": "Mely közigazgatási épületek találhatóak a miskolci Városház téren?", "answers": ["Városháza", "Földhivatal", "Megyeháza", "Posta"], "solution": ["A", "C"]}
        """
    elif question_type == 'true_or_false':
        type_prompt = "Create a true or false question."
        type_example = """Magyarország állam Közép-Európában, a Kárpát-medence közepén. 1989 óta parlamentáris köztársaság. Északról Szlovákia, északkeletről Ukrajna, keletről és délkeletről Románia, délről Szerbia, délnyugatról Horvátország és Szlovénia, nyugatról pedig Ausztria határolja.
        Output:
        {"question": "Ukrajna Magyarország déli szomszédja.", "solution": false}
        """ if not en else """
        By the early 1990s, however, relational systems dominated in all large-scale data processing applications, and as of 2018 they remain dominant: IBM Db2, Oracle, MySQL, and Microsoft SQL Server are the most searched DBMS.
        Output:
        {"question": "Relational database systems dominated the 1990s, but as of 2018 they have lost their relevance", solution": false}"""
    else:
        raise ValueError('Invalid question type.')
    return f"""You are a helpful teacher. You are preparing a quiz for students.
    You will create questions for snippets of text. {type_prompt} Output the question in json format.
    Here are some examples:
    Snippet:
    {type_example}
    Snippet:
    {text}
    Output:
    """

def tryparse(generated_output:str, question_type:str) -> dict:
    if question_type == 'single_choice':
        return SingleChoice.model_validate_json(generated_output).model_dump()
    if question_type == 'multiple_choice':
        return MultipleChoice.model_validate_json(generated_output).model_dump()
    if question_type == 'true_or_false':
        return TrueOrFalse.model_validate_json(generated_output).model_dump()
    raise ValueError('Invalid question type')

async def generate_question_for_type(session, key: str, model: str, text: str, typ: str, en:bool=True) -> dict | None:
    prompt = convert_text_to_prompt(text, typ, en)
    backoff_base = 4
    max_retries = 5
    tries = 0
    tries_because_of_429 = 0
    successful = False
    while tries <=max_retries and not successful:
        try:
            res = await make_request(session,key, model, prompt, typ)
            if res is not None:
                res = res['candidates'][0]
                if res.get('content') is None:
                    print('NO "content" in res:',res)
                    raise ValueError('No content in response.')
                res = res['content']['parts'][0]['text']
                res = res.removeprefix('```json\n').removesuffix('\n```')
                print(res)
                parsed = tryparse(res, typ)
                successful = True
                parsed['Context'] = text
                return parsed
        except httpx.HTTPStatusError as e:
            tries += 1
            logging.error(f'Failed request #{tries} for reason {e}')
            if e.response.status_code == 429:
                tries_because_of_429 += 1
                backoff_secs = (backoff_base*tries_because_of_429)**2
                await asyncio.sleep(backoff_secs)
            else:
                await asyncio.sleep(backoff_base)
        except Exception as e:
            tries += 1
            logging.error(f'Failed request #{tries} for reason {e}')


def init_progress_dict(articles: dict) -> dict:
    p = {}
    for article_url in articles.keys():
        p[article_url] = []
        for i, _ in enumerate(articles[article_url]):
            for t in QUESTION_TYPES:
                p[article_url].append({'type':t, 'section':i})
    return p

def update_progress_dict(progress: dict, url: str, section_index: int, typ: str, save_path: str) -> None:
    prev: list[dict] | None = progress.get(url)
    if prev is None:
        return
    try:
        prev.remove({'section':section_index, 'type':typ})
        if len(prev) == 0:
            progress.pop(url)
        else:
            progress[url] = prev
        with open(save_path, 'w') as f:
            json.dump(progress, f, indent=2)
    except ValueError:
        print('Progress not updated, section not found.')
        pass

@dataclass
class Job:
    typ: str
    url: str
    section: str
    section_index: int
            
async def generate_all_questions(article_data_path: str, progress_dict_path: str, output_dir: str, key: str, model: str):
    articles = read_article_data(article_data_path)
    progress_so_far = read_progress_dict(progress_dict_path)
    if not progress_so_far:
        progress_so_far = init_progress_dict(articles)
        with open(progress_dict_path, 'w') as f:
            json.dump(progress_so_far, f)
    progress_tracker = {k: v.copy() for k, v in progress_so_far.items()}
    bs = 3
    async with httpx.AsyncClient(timeout=25) as session:
        while sections_left(progress_tracker) > 0:
            for todo_url, todo_sections in tqdm(progress_so_far.items(), desc='Articles left to process'):
                for i in tqdm(range(0,len(todo_sections), bs), desc='Sections left to process', total=len(todo_sections)//bs+1, leave=False, position=1, colour='green'):
                    max_i = min(len(todo_sections)-1, i+bs)
                    batch = [
                        Job(
                            typ=todo_section['type'],
                            url=todo_url, 
                            section=articles[todo_url][todo_section['section']],
                            section_index=todo_section['section']
                        ) for todo_section in todo_sections[i:max_i]]
                    await process_batch(session, key, batch, progress_dict_path, progress_tracker, output_dir, model)

def sections_left(progress: dict) -> int:
    return sum([len(v) for v in progress.values()])

async def process_batch(session, key, batch: list[Job], progress_dict_path:str, progress: dict, output_dir: str, model: str):
    tasks = [generate_question_for_type(session, key, model, job.section, job.typ, 'en.wikipedia' in job.url) for job in batch]
    res = await asyncio.gather(*tasks)
    for j, r in enumerate(res):
        if r is not None:
            original_job = batch[j]
            url_enc = hashlib.sha1(original_job.url.encode()).digest().hex()
            with open(os.path.join(output_dir, f'{url_enc}_{original_job.section_index}_{original_job.typ}.json'), 'w') as f:
                json.dump(r, f)
            update_progress_dict(progress, original_job.url, original_job.section_index, original_job.typ, progress_dict_path)
            try:
                with open(os.path.join(output_dir, f'{url_enc}_{original_job.section_index}_{original_job.typ}.json'), 'w') as f:
                    json.dump(r, f)
                update_progress_dict(progress, original_job.url, original_job.section_index, original_job.typ, progress_dict_path)
            except KeyboardInterrupt:
                with open(os.path.join(output_dir, f'{url_enc}_{original_job.section_index}_{original_job.typ}.json'), 'w') as f:
                    json.dump(r, f)
                update_progress_dict(progress, original_job.url, original_job.section_index, original_job.typ, progress_dict_path)
                exit()


def read_article_data(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def read_progress_dict(file_path: str) -> dict:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Path to the input file.')
    parser.add_argument('output_directory', type=str, help='Path to the output directory for questions.')
    parser.add_argument('--progress_json', type=str,default='progress.json', help='Path to the json file storing the unfinished questions.')
    parser.add_argument('--model', choices=['gemini-1.0-pro-latest', 'gemini-1.5-flash-latest'], default='gemini-1.5-flash-latest', help='The model to use for question generation.')
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    articles = read_article_data(args.input_file)

    dotenv.load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError('API_KEY not found in environment variables.')
    loop = asyncio.new_event_loop()
    loop.run_until_complete(generate_all_questions(args.input_file, args.progress_json, args.output_directory, api_key, args.model))
    loop.close()
