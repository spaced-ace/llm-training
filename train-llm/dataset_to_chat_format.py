import os
import json
import datasets
import argparse
import pandas as pd

SYSTEM_EN = 'You are a helpful assistant to a teacher, who creates test questions for students in json format.'

PROMPT_EN = """
Create a {} question based on the context.
Example:
<context>{}</context>
<output>{}</output>
Task:
<context>{}</context>
"""
PROMPT_HU = """
Írj egy {} kérdést a kontextus alapján.
Példa:
<context>{}</context>
<output>{}</output>
Feladat:
<context>{}</context>
"""

RESPONSE = """<output>{}</output>"""

EXAMPLE_CONTEXT_EN = """
The Nobel Prize in Literature (here meaning for literature; Swedish: Nobelpriset i litteratur) is a Swedish literature prize that is awarded annually, since 1901, to an author from any country who has, in the words of the will of Swedish industrialist Alfred Nobel, "in the field of literature, produced the most outstanding work in an idealistic direction"
"""
MULTI_EXAMPLE_EN = """
{
    "question": "Which of the following statements are true about the Nobel Prize in Literature?",
    "answers": [
        "It is awarded annually.",
        "It is only awarded to Swedish authors.",
        "It has been awarded since 1901.",
        "It is given for outstanding work in the field of literature."
        ],
    "solution": ["A", "C", "D"]
}"""
SINGLE_EXAMPLE_EN = """
{
    "question": "Which of the following statements are true about the Nobel Prize in Literature?",
    "answers": [
        "It is awarded to an author from Sweden.",
        "It is awarded to an author from any country for producing outstanding work in an idealistic direction.",
        "It is awarded to the best-selling author of the year.",
        "It is awarded to an author for writing about Swedish history."
    ],
    "solution": "B"
}"""
BOOLEAN_EXAMPLE_EN = """
{
    "question":"The Nobel Prize in Literature is awarded annually to authors only from Sweden.",
    "solution":false
}"""

EXAMPLE_CONTEXT_HU = """Nobel-díjat a svéd kémikus és feltaláló Alfred Nobel alapította. Nobel 1895 november 27-én kelt végrendeletében rendelkezett úgy, hogy vagyonának kamataiból évről évre részesedjenek a fizika, kémia, fiziológia és orvostudomány, továbbá az irodalom legjobbjai és az a személy, aki a békéért tett erőfeszítéseivel a díjat kiérdemli."""

BOOLEAN_EXAMPLE_HU = """
{
    "question":"Nobel-díjat csak a svéd kémikusok és feltalálók kaphatnak meg.",
    "solution":false
}"""

SINGLE_EXAMPLE_HU = """
{
    "question": "Mi volt Alfred Nobel végrendeletének célja a Nobel-díjjal kapcsolatban?",
    "answers": [
        "Csak svéd tudósoknak adják át.",
        "A fizika, kémia, fiziológia, orvostudomány, irodalom legjobbjait és a békéért küzdő személyt jutalmazzák.",
        "Csak irodalmi teljesítményért ítélik oda.",
        "A legújabb találmányokat jutalmazzák."
    ],
    "solution": "B"
}"""

MULTI_EXAMPLE_HU = """
{
    "question": "Mely állítások igazak a Nobel-díjjal kapcsolatban?",
    "answers": [
    "Alfred Nobel alapította a díjat.",
    "A díjat csak fizikai teljesítményért ítélik oda.",
    "A végrendeletében rendelkezett a díj alapításáról.",
    "A békéért tett erőfeszítéseket is jutalmazzák."
        ],
    "solution": ["A", "C", "D"]
}"""

SYSTEM_HU = 'Segítőkész asszisztens vagy egy tanárnak, aki tesztkérdéseket készít a diákok számára json formátumban.'


def format_question(question: dict) -> list[dict]:
    """Formats a question into a list of conversation turns"""
    question_type = (
        'boolean'
        if 'answers' not in question.keys()
        else 'multiple choice single answer (4 options)'
        if type(question['solution']) == str
        else 'multiple choice multiple answers (4 options)'
    )
    lang = 'hu' if 'hu.wikipedia' in question['url'] else 'en'
    if lang == 'en':
        example_en = (
            BOOLEAN_EXAMPLE_EN
            if question_type == 'boolean'
            else SINGLE_EXAMPLE_EN
            if 'single' in question_type
            else MULTI_EXAMPLE_EN
        )
        prompt = PROMPT_EN.format(
            question_type,
            EXAMPLE_CONTEXT_EN,
            example_en,
            question['context'],
        )
    elif lang == 'hu':
        example_hu = (
            BOOLEAN_EXAMPLE_HU
            if question_type == 'boolean'
            else SINGLE_EXAMPLE_HU
            if 'single' in question_type
            else MULTI_EXAMPLE_HU
        )
        question_type_hu = (
            'igaz/hamis'
            if 'answers' not in question.keys()
            else 'több válaszlehetőséges (4 opciós)'
            if type(question['solution']) == list
            else 'egy válaszlehetőséges (4 opciós)'
        )
        prompt = PROMPT_HU.format(
            question_type_hu,
            EXAMPLE_CONTEXT_HU,
            example_hu,
            question['context'],
        )
    else:
        raise ValueError('Language not supported')
    question_output = {}
    question_output['question'] = question['question']
    if question.get('answers') is not None:
        question_output['answers'] = list(question['answers'])
    question_output['solution'] = (
        question['solution']
        if type(question['solution']) == str
        or type(question['solution']) == bool
        else list(question['solution'])
    )
    question_output = json.dumps(question_output, ensure_ascii=False, indent=4)
    convo = [
        {
            'role': 'system',
            'content': SYSTEM_EN if lang == 'en' else SYSTEM_HU,
        },
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': RESPONSE.format(question_output)},
    ]
    return convo


def format_dataset(ds: datasets.Dataset) -> datasets.Dataset:
    df = ds.to_pandas()
    df['formatted_conversation'] = df.apply(
        lambda q: format_question(dict(q)), axis=1
    )
    return datasets.Dataset.from_pandas(
        df[['formatted_conversation']], preserve_index=False
    )


def train_test_split(
    ds: datasets.DatasetDict,
    test_fraction: float = 0.05,
    validation_fraction: float = -1,
    seed: int = 42,
) -> datasets.DatasetDict:
    df_train = []
    df_test = []
    df_val = []
    for key in list(ds.keys()):
        df = ds[key].to_pandas()
        if validation_fraction > 0:
            train, test = train_test_split_pandas(
                df, test_fraction + validation_fraction, seed
            )
            val_size = int(len(df) * validation_fraction)
            test = test.sample(frac=1, random_state=seed)
            validation = test.head(val_size)
            test = test.tail(len(test) - val_size)
            df_val.append(validation)
        else:
            train, test = train_test_split_pandas(df, test_fraction, seed)
        df_train.append(train)
        df_test.append(test)
    df_train = pd.concat(df_train).sample(frac=1, random_state=seed)
    df_test = pd.concat(df_test).sample(frac=1, random_state=seed)
    if validation_fraction <= 0:
        return datasets.DatasetDict(
            {
                'train': datasets.Dataset.from_pandas(
                    df_train, preserve_index=False
                ),
                'test': datasets.Dataset.from_pandas(
                    df_test, preserve_index=False
                ),
                'validation': datasets.Dataset.from_pandas(
                    df_val, preserve_index=False
                ),
            }
        )
    df_val = pd.concat(df_val).sample(frac=1, random_state=seed)
    return datasets.DatasetDict(
        {
            'train': datasets.Dataset.from_pandas(
                df_train, preserve_index=False
            ),
            'test': datasets.Dataset.from_pandas(
                df_test, preserve_index=False
            ),
            'validation': datasets.Dataset.from_pandas(
                df_val, preserve_index=False
            ),
        }
    )


def train_test_split_pandas(
    df: pd.DataFrame,
    test_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * (1 - test_fraction))
    df_tmp = df.sample(frac=1, random_state=seed)
    return df_tmp.head(train_size), df.tail(len(df) - train_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_directory',
        type=str,
        help='path to the directory containing the dataset',
    )
    parser.add_argument(
        'repo_id',
        type=str,
        help='repo id to push the formatted dataset to',
        default='jazzysnake01/quizgen-chat-md',
    )
    parser.add_argument(
        '--score_minimum',
        type=float,
        default=0.5,
        help='minimum threshold of context score for inclusion in trainig',
    )
    parser.add_argument(
        '--test_frac',
        type=float,
        default=0.05,
        help='fration of the data that will be dedicated to testing',
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.05,
        help='fration of the data that will be dedicated to testing',
    )
    args = parser.parse_args()

    ds = datasets.load_from_disk(args.dataset_directory)
    # only keep MultipleChoice or SingleChoice questions that have 4 options
    ds = ds.filter(
        lambda x: True
        if type(x['solution']) == bool
        else len(x['answers']) <= 4
    )
    ds = ds.filter(lambda x: x['score'] > args.score_minimum)

    ds_f = {}
    for key in ds.keys():
        ds_f[key] = format_dataset(ds[key])
    ds_f = datasets.DatasetDict(ds_f)
    ds_f = train_test_split(
        ds_f, test_fraction=args.test_frac, validation_fraction=args.val_frac
    )
    print(f'RESULTING DATASET:\n{ds_f}')
    ds_f.push_to_hub(args.repo_id, token=os.environ['HF_WRITE'])
