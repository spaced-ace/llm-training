import os
import json
import pandas as pd
import glob
import argparse
import hashlib
import datasets

from generate_questions import MultipleChoice, SingleChoice, TrueOrFalse


def read_json_with_question(path: str) -> dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def parse_single_choices(questions: list[str]) -> list[dict]:
    qs = []
    for q in questions:
        try:
            with open(q, 'r') as f:
                data = json.load(f)
                SingleChoice(**data)
                qs.append(data)
        except Exception as e:
            print(f'error while parsing {q}: {e}')
    return qs


def parse_true_or_falses(questions: list[str]) -> list[dict]:
    qs = []
    for q in questions:
        try:
            with open(q, 'r') as f:
                data = json.load(f)
                TrueOrFalse(**data)
                qs.append(data)
        except Exception as e:
            print(f'error while parsing {q}: {e}')
    return qs


def parse_multiple_choices(questions: list[str]) -> list[dict]:
    qs = []
    for q in questions:
        try:
            with open(q, 'r') as f:
                data = json.load(f)
                MultipleChoice(**data)
                qs.append(data)
        except Exception as e:
            print(f'error while parsing {q}: {e}')
    return qs


def sha1hash(string: str) -> str:
    return hashlib.sha1(string.encode()).digest().hex()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_directory',
        type=str,
        help="the directory containing the generated questions's jsons",
    )
    parser.add_argument(
        'original_input',
        type=str,
        help='the path to the json containing the original scraped content with urls',
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='the path to where the unified dataset will be output',
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_directory):
        print('provided input_directory does not exist')
        exit(1)

    if not os.path.exists(args.original_input):
        print('provided original_input does not exist')
        exit(1)

    try:
        with open(args.original_input, 'r') as f:
            orig = json.load(f)
    except Exception as e:
        print(e)
        exit(1)
    urls = {sha1hash(k): k for k in orig.keys()}

    true_or_falses = glob.glob(
        os.path.join(args.input_directory, '*true_or_false*')
    )
    multiples = glob.glob(
        os.path.join(args.input_directory, '*multiple_choice*')
    )
    singles = glob.glob(os.path.join(args.input_directory, '*single_choice*'))

    singles_parsed = parse_single_choices(singles)
    multiples_parsed = parse_multiple_choices(multiples)
    true_falses_parsed = parse_true_or_falses(true_or_falses)
    singles_matched = [
        (urls.get(singles[i].split('/')[-1].split('_')[0]), singles_parsed[i])
        for i in range(len(singles_parsed))
    ]
    multis_matched = [
        (
            urls.get(multiples[i].split('/')[-1].split('_')[0]),
            multiples_parsed[i],
        )
        for i in range(len(multiples_parsed))
    ]
    tfs_matched = [
        (
            urls.get(true_or_falses[i].split('/')[-1].split('_')[0]),
            true_falses_parsed[i],
        )
        for i in range(len(true_falses_parsed))
    ]
    df_singles = pd.DataFrame(
        data={
            'url': [q[0] for q in singles_matched],
            'context': [q[1]['Context'] for q in singles_matched],
            'question': [q[1]['question'] for q in singles_matched],
            'answers': [q[1]['answers'] for q in singles_matched],
            'solution': [q[1]['solution'] for q in singles_matched],
        }
    )
    df_multiples = pd.DataFrame(
        data={
            'url': [q[0] for q in multis_matched],
            'context': [q[1]['Context'] for q in multis_matched],
            'question': [q[1]['question'] for q in multis_matched],
            'answers': [q[1]['answers'] for q in multis_matched],
            'solution': [q[1]['solution'] for q in multis_matched],
        }
    )
    df_tfs = pd.DataFrame(
        data={
            'url': [q[0] for q in tfs_matched],
            'context': [q[1]['Context'] for q in tfs_matched],
            'question': [q[1]['question'] for q in tfs_matched],
            'solution': [q[1]['solution'] for q in tfs_matched],
        }
    )

    ds = datasets.DatasetDict(
        {
            'boolean': datasets.Dataset.from_pandas(df_tfs),
            'multi': datasets.Dataset.from_pandas(df_multiples),
            'single': datasets.Dataset.from_pandas(df_singles),
        }
    )
    ds.save_to_disk(args.output_path)
