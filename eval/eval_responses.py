import os
import json
import tqdm
import math
import asyncio
import sqlite3
import argparse
import httpx
import pandas as pd
from typing import Any, Generator, Callable
from typing_extensions import TypeVar

T = TypeVar('T')

harm_categories = [
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    'HARM_CATEGORY_HARASSMENT',
]


class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k - 2))


def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def connect_to_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = dict_factory
    conn.create_aggregate('stdev', 1, StdevFunc)
    cursor = conn.cursor()
    cursor.execute('PRAGMA foreign_keys = ON')
    cursor.close()
    return conn


def get_data_to_eval(
    conn: sqlite3.Connection,
    run_name: str,
    evaluator_model_name: str,
    metric: str,
    question_types: list[str] = ['mcsa', 'mcma', 'boolean'],
) -> list[dict]:
    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT
            c.run_name,
            c.conversation_index,
            c.question
        FROM chat_responses c
        INNER JOIN valid_questions v
        ON 1=1
            AND v.conversation_index = c.conversation_index
            AND v.run_name = c.run_name
        INNER JOIN question_metadata qm
        ON 1=1
            AND qm.conversation_index = c.conversation_index
        WHERE 1=1
            AND c.question IS NOT NULL
            AND c.run_name=:run_name
            AND c.conversation_index NOT IN ( --not rated already
                SELECT conversation_index
                FROM question_eval q
                WHERE 1=1
                AND q.metric_name = :metric_name
                AND q.run_name = :run_name
                AND q.evaluator_model = :evaluator_model
            )
            AND (1=0
                    {"OR qm.q_type = 'boolean'" if 'boolean' in question_types else ''}
                    {"OR qm.q_type = 'mcma'" if 'mcma' in question_types else ''}
                    {"OR qm.q_type = 'mcsa'" if 'mcsa' in question_types else ''}
                )
        """,
        {
            'run_name': run_name,
            'evaluator_model': evaluator_model_name,
            'metric_name': metric,
        },
    )
    res = cursor.fetchall()
    cursor.close()
    return res


def create_eval_table(conn: sqlite3.Connection) -> None:
    question_eval_table = """
    CREATE TABLE IF NOT EXISTS question_eval(
        run_name TEXT NOT NULL,
        conversation_index INTEGER NOT NULL,
        evaluator_model TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        FOREIGN KEY (run_name, conversation_index)
            REFERENCES chat_responses(run_name, conversation_index)
        )
    """
    cur = conn.cursor()
    cur.execute(question_eval_table)
    cur.close()


def create_question_metadata_table(conn: sqlite3.Connection) -> None:
    qm_table = """
    CREATE TABLE IF NOT EXISTS question_metadata(
        conversation_index INTEGER PRIMARY KEY NOT NULL,
        lang TEXT NOT NULL,
        q_type TEXT NOT NULL
        );
    """
    qm_table_load = """
        INSERT INTO question_metadata
        SELECT DISTINCT 
            r.conversation_index,
            CASE 
                WHEN r.messages LIKE '%Segítőkész%' THEN 'hu'
                ELSE 'en'
            END AS lang,
            CASE 
                WHEN r.messages LIKE '%multiple choice single answer (4 options)%'
                    OR r.messages LIKE '%egy válaszlehetőséges (4 opciós)%' THEN 'mcsa'
                WHEN r.messages LIKE '%multiple choice multiple answers (4 options)%'
                    OR r.messages LIKE '%több válaszlehetőséges (4 opciós)%' THEN 'mcma'
                ELSE 'boolean'
            END AS q_type
        FROM chat_responses r
        WHERE r.conversation_index NOT IN (
            SELECT conversation_index 
            FROM question_metadata
        );
    """
    cur = conn.cursor()
    cur.execute('BEGIN')
    cur.execute(qm_table)
    cur.execute(qm_table_load)
    cur.execute('COMMIT')
    cur.close()


def create_valid_questions_view(conn: sqlite3.Connection) -> None:
    valid_questions_view = """
    CREATE VIEW IF NOT EXISTS valid_questions AS
    SELECT
        cr.run_name,
        cr.conversation_index,
        cr.messages,
        cr.question
    FROM
        chat_responses cr
    JOIN
        question_metadata qm ON cr.conversation_index = qm.conversation_index
    WHERE
        cr.question IS NOT NULL
        AND json_extract(cr.question, '$.context') IS NOT NULL
        AND json_extract(cr.question, '$.question') IS NOT NULL
        AND (
            (qm.q_type = 'mcsa' AND
                json_extract(cr.question, '$.answers') IS NOT NULL
                AND json_type(cr.question, '$.solution') = 'text')
            OR
            (qm.q_type = 'mcma' AND
                json_extract(cr.question, '$.answers') IS NOT NULL
                AND json_type(cr.question, '$.solution') = 'array')
            OR
            (qm.q_type = 'boolean' AND
                json_type(cr.question, '$.solution') IN ('true', 'false'))
        )
    """
    cur = conn.cursor()
    cur.execute(valid_questions_view)
    cur.close()


def create_aggregate_stats_view(conn: sqlite3.Connection) -> None:
    aggregate_stats_view = """
    CREATE VIEW IF NOT EXISTS aggregate_stats AS 
        SELECT
            evaluator_model,
            run_name,
            metric_name,
            AVG(metric_value) as average,
            STDEV(metric_value) as std,
            MIN(metric_value) as minimum,
            MAX(metric_value) as maximum,
            COUNT(*) as n
        FROM question_eval
        GROUP BY evaluator_model, run_name, metric_name
        ORDER BY evaluator_model, metric_name, average
    """
    cur = conn.cursor()
    cur.execute('BEGIN')
    cur.execute('DROP VIEW IF EXISTS aggregate_stats')
    cur.execute(aggregate_stats_view)
    cur.execute('COMMIT')

    aggregate_stats_lang_view = """
    CREATE VIEW IF NOT EXISTS aggregate_stats_lang AS 
        SELECT
            qe.evaluator_model,
            qe.run_name,
            qe.metric_name,
            qm.lang,
            AVG(qe.metric_value) as average,
            STDEV(qe.metric_value) as std,
            MIN(qe.metric_value) as minimum,
            MAX(qe.metric_value) as maximum,
            COUNT(*) as n
        FROM question_eval qe
        INNER JOIN question_metadata qm
        ON qm.conversation_index = qe.conversation_index
        GROUP BY qe.evaluator_model, qe.run_name, qe.metric_name, qm.lang
        ORDER BY qe.evaluator_model, qe.metric_name, qm.lang, average
    """
    cur = conn.cursor()
    cur.execute('BEGIN')
    cur.execute('DROP VIEW IF EXISTS aggregate_stats_lang')
    cur.execute(aggregate_stats_lang_view)
    cur.execute('COMMIT')
    cur.close()


def get_aggregate_report(
    conn: sqlite3.Connection, group_by_lang: bool = False
) -> list[dict]:
    get_aggregate_query = 'SELECT * FROM aggregate_stats'
    if group_by_lang:
        get_aggregate_query = 'SELECT * FROM aggregate_stats_lang'
    cur = conn.cursor()
    cur.execute(get_aggregate_query)
    res = cur.fetchall()
    cur.close()
    return res


def save_evaluated_questions(
    conn: sqlite3.Connection,
    evaluated_questions: list[dict],
    metric: str,
) -> None:
    save_query = """INSERT INTO question_eval(
        run_name,
        conversation_index,
        evaluator_model,
        metric_name,
        metric_value
    ) VALUES (?,?,?,?,?)
    """
    params = [
        (
            q['run_name'],
            q['conversation_index'],
            q['evaluator_model'],
            metric,
            q[metric],
        )
        for q in evaluated_questions
    ]
    cur = conn.cursor()
    cur.executemany(save_query, params)
    cur.close()


def openai_conversation_to_google(
    convo: list[dict],
) -> tuple[dict | None, list[dict]]:
    if len(convo) == 0:
        return None, []
    convo_cpy = convo.copy()
    system = None
    if convo[0]['role'] == 'system':
        system = {'parts': {'text': convo_cpy[0]['content']}}
        convo_cpy = convo_cpy[1:]
    convo_transformed = [
        {
            'role': 'model' if msg['role'] == 'assistant' else 'user',
            'parts': [{'text': msg['content']}],
        }
        for msg in convo_cpy
    ]
    return system, convo_transformed


async def make_request(
    session: httpx.AsyncClient,
    api_key: str,
    model: str,
    conversation_openai: list[dict],
):
    system, convo = openai_conversation_to_google(conversation_openai)
    # gemini-1.5-flash-latest, 2024-10-31
    if model not in ['gemini-1.5-flash']:
        exit('Invalid model')
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key='
        + api_key
    )
    gen_conf = {
        'temperature': 0.0,
        #'responseMimeType': 'application/json',
    }
    headers = {'Content-Type': 'application/json'}
    payload = {
        'safetySettings': [
            {
                'category': cat,
                'threshold': 'BLOCK_NONE',
            }
            for cat in harm_categories
        ],
        'generationConfig': gen_conf,
        'contents': convo,
    }
    if system is not None:
        payload['system_instruction'] = system

    response = await session.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def question_to_string(q: dict) -> str:
    sol = q.get('solution')
    if type(sol) == list or type(sol) == str:
        return f"""Context: {q['context']}
        Question: {q['question']}
        Answers: {q['answers']}
        Solution: {q['solution']}
        """
    if type(sol) == bool:
        return f"""Context: {q['context']}
        Question: {q['question']}
        Solution: {q['solution']}
        """

    raise ValueError('Unrecognized question type')


def format_question_relevance_prompt(question: dict) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': """
                You serve as an evaluator of quiz questions.

                Evaluate the relevance of the question to the context provided on a 5-point scale using the following words:
                1. 'Irrelevant' - Completely unrelated to the context.
                2. 'Somewhat irrelevant' - Partially related but largely irrelevant to the context.
                3. 'Neutral' - Neither particularly relevant nor irrelevant; moderately related.
                4. 'Relevant' - Clearly related and generally aligns with the context.
                5. 'Highly relevant' - Strongly related, with direct alignment to the context content.

                Based on this scale, assess the following context and question-answer set.
                **Output ONLY A SINGLE CLASS from the five above**

                Examples:
                #1
                <question that asks about info not in the context>
                Output: 'Irrelevant'
                #2
                <question that asks about info that is barely mentioned in the context>
                Output: 'Somewhat Irrelevant'
                #3
                <question that asks about miscellaneous info in the context>
                Output: 'Neutral'
                #4
                <question that asks about important info in the context>
                Output: 'Relevant'
                #5
                <question that asks about the most meaningful info in the context>
                Output: 'Highly relevant'
                """,
        },
        {
            'role': 'user',
            'content': f"""Now assess the following:
                {question_to_string(question)}
                Output: """,
        },
    ]


def parse_question_relevance_response(response: dict) -> int | None:
    try:
        model_response = response['candidates'][0]['content']['parts'][0][
            'text'
        ]
        model_response = model_response.lower()
    except Exception:
        return None
    if 'somewhat irrelevant' in model_response:
        return 2
    if 'irrelevant' in model_response:
        return 1
    if 'neutral' in model_response:
        return 3
    if 'highly relevant' in model_response:
        return 5
    if 'relevant' in model_response:
        return 4


def parse_question_clarity_response(response: dict) -> int | None:
    try:
        model_response = response['candidates'][0]['content']['parts'][0][
            'text'
        ]
        model_response = model_response.lower()

    except Exception:
        return None
    if 'completely unclear' in model_response:
        return 1
    if 'mostly unclear' in model_response:
        return 2
    if 'somewhat clear' in model_response:
        return 3
    if 'mostly clear' in model_response:
        return 4
    if 'completely clear' in model_response:
        return 5


def format_question_clarity_prompt(question: dict) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': """
                You serve as an evaluator of quiz questions.

                Evaluate the clarity of a question on a 5-point scale. When rating the clarity keep
                in mind that the user will not be able to see the context when solving the question
                (but they have read it previously). Use the following classes for rating:

                1. 'Completely Unclear' - The question is very confusing, vague, or nonsensical. It lacks coherent structure, has incorrect grammar, or contains terms that are out of place.
                2. 'Mostly Unclear' - The question is difficult to understand due to ambiguous phrasing or overly technical language. It may contain redundant words or confusing structure.
                3. 'Somewhat Clear' - The question is understandable contains vague or complex wording. It might require the reader to re-read or make assumptions to interpret the meaning accurately.
                4. 'Mostly Clear' - The question is generally clear but might have minor wording that could be simplified. The meaning is still easy to grasp without confusion.
                5. 'Completely Clear' - The question is concise, easy to understand, and unambiguous. It uses straightforward language with no unnecessary information.

                Based on this scale, assess the following context and question-answer set.
                **Output ONLY A SINGLE CLASS from the five above**

                Examples:
                #1
                <incoherent question>
                Output: 'Completely Unclear'
                #2
                <ambigous question>
                Output: 'Mostly Unclear'
                #3
                <overly complex but understandable question>
                Output: 'Somewhat clear'
                #4
                <slightly confusing wording but otherwise clear question>
                Output: 'Mostly clear'
                #5
                <succint, straightforward, easy to understand question>
                Output: 'Completely Clear'
                """,
        },
        {
            'role': 'user',
            'content': f"""Now assess the following:
                {question_to_string(question)}
                Output: """,
        },
    ]


def format_answer_readability_prompt(question: dict) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': """
                You serve as an evaluator of quiz questions.

                Evaluate the readability of the answer choices on a 5-point scale. Use the following classes for rating:

                1. 'Very Difficult' - Answer choices are very complex, with technical terms, convoluted sentence structures, or ambiguous phrasing. They could easily confuse readers.
                2. 'Somewhat Difficult' - Answer choices contain complex phrasing, but the sentece structure is not overly complex. The answer options might be too wordy.
                3. 'Moderately Readable' - Answer choices are somewhat clear, with straightforward sentence structures and only a few complex phrases.
                4. 'Mostly Easy' - Answer choices are clear, with minor complexity that does not hinder understanding.
                5. 'Very Easy' - Answer choices are simple, concise, and easy to understand. They avoid all unnecessary technical terms or jargon.

                Based on this scale, assess the answer choices for readability. Output ONLY A SINGLE CLASS from the five above.

                Examples:
                #1
                <Technical, convoluted answer choices>
                Output: 'Very Difficult'
                #2
                <Answer choices with complex phrasing and unnecessary details>
                Output: 'Somewhat Difficult'
                #3
                <Answer choices with minor complexity and unnecessary technical terms>
                Output: 'Moderately Readable'
                #4
                <Clear answer choices with minor complexity>
                Output: 'Mostly Easy'
                #5
                <Straightforward, simple answer choices>
                Output: 'Very Easy'
            """,
        },
        {
            'role': 'user',
            'content': f"""Now assess the following:
                {question_to_string(question)}
                Output: """,
        },
    ]


def format_distractor_quality_prompt(question: dict) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': """
                You serve as an evaluator of quiz questions.

                Evaluate the quality of the distractors (incorrect answer choices) on a 5-point scale using the following words:

                1. 'Poor' - Distractors are irrelevant or nonsensical, making the correct answer obvious without any background knowledge. They fail to serve as plausible alternatives.
                2. 'Weak' - Distractors are poorly chosen, with one or more options clearly irrelevant or too easily dismissed. The correct answer is apparent with only minimal knowledge of the topic.
                3. 'Moderate' - Distractors have a loose connection to the topic but contain one or two options that are either too obvious or unrelated. Correct options are somewhat easier to identify.
                4. 'Good' - Distractors are mostly plausible and relate to the topic, though one may be slightly less convincing. They still require some knowledge of the subject to be ruled out.
                5. 'Excellent' - Distractors are highly plausible and relate closely to the topic, making them tempting choices. They are clearly incorrect but appear reasonable within the context, requiring understanding of the material to identify the correct answer.

                Based on this scale, assess the following question-answer set for distractor quality. Output ONLY A SINGLE CLASS from the five above.

                Examples:
                #1
                <Answer choices with irrelevant distractors>
                Output: 'Poor'
                #2
                <Answer choices with weak, easily dismissed distractors>
                Output: 'Weak'
                #3
                <Answer choices with moderately related distractors, some weak>
                Output: 'Moderate'
                #4
                <Answer choices with mostly plausible distractors>
                Output: 'Good'
                #5
                <Answer choices with highly plausible, topic-related distractors>
                Output: 'Excellent'
                """,
        },
        {
            'role': 'user',
            'content': f"""Now assess the following:
                {question_to_string(question)}
                Output: """,
        },
    ]


def format_incorrect_option_plausability_prompt(question: dict) -> list[dict]:
    return [
        {
            'role': 'system',
            'content': """
                You serve as an evaluator of quiz questions.

                Evaluate the plausibility of the incorrect option in a boolean (true/false) question on a 5-point scale.
                This metric assesses how believable or reasonable the incorrect option appears within the question context,
                regardless of whether it is technically true or false. Use the following classes for rating:

                1. 'Completely Implausible' - The incorrect option is obviously wrong or nonsensical. It is unlikely to mislead anyone with basic knowledge of the topic.
                2. 'Mostly Implausible' - The incorrect option has some connection to the topic but is still clearly false. It may only be plausible in very specific or unusual interpretations.
                3. 'Somewhat Plausible' - The incorrect option could be believable in certain cases, but generally, it would not mislead someone with a moderate understanding of the context.
                4. 'Mostly Plausible' - The incorrect option is quite believable and could easily be mistaken for the correct answer. It would require a good understanding of the topic to be ruled out.
                5. 'Completely Plausible' - The incorrect option is highly believable and closely resembles the correct answer, making it challenging to distinguish between the two. It requires strong knowledge of the context to identify the true answer confidently.

                Based on this scale, assess the following question-answer set for the plausibility of the incorrect option. Output ONLY A SINGLE CLASS from the five above.

                Examples:
                #1
                <Very unrealistic incorrect option>
                Output: Completely Implausible
                #2
                <Incorrect option that’s loosely related but unlikely>
                Output: Mostly Implausible
                #3
                <Incorrect option that’s somewhat believable but generally clear>
                Output: Somewhat Plausible
                #4
                <Incorrect option that is very close to being correct>
                Output: Mostly Plausible
                #5
                <Incorrect option that seems as credible as the correct one>
                Output: Completely Plausible
                """,
        },
        {
            'role': 'user',
            'content': f"""Now assess the following:
                {question_to_string(question)}
                Output: """,
        },
    ]


def parse_incorrect_option_plausibility_response(response: dict) -> int | None:
    try:
        model_response = response['candidates'][0]['content']['parts'][0][
            'text'
        ]
        model_response = model_response.lower()
    except Exception:
        return None
    if 'completely implausible' in model_response:
        return 1
    if 'mostly implausible' in model_response:
        return 2
    if 'somewhat plausible' in model_response:
        return 3
    if 'mostly plausible' in model_response:
        return 4
    if 'completely plausible' in model_response:
        return 5


def parse_answer_readibility_response(response: dict) -> int | None:
    try:
        model_response = response['candidates'][0]['content']['parts'][0][
            'text'
        ]
        model_response = model_response.lower()

    except Exception:
        return None
    if 'very difficult' in model_response:
        return 1
    if 'somewhat difficult' in model_response:
        return 2
    if 'moderately readable' in model_response:
        return 3
    if 'mostly easy' in model_response:
        return 4
    if 'very easy' in model_response:
        return 5


def parse_distractor_quality_response(response: dict) -> int | None:
    try:
        model_response = response['candidates'][0]['content']['parts'][0][
            'text'
        ]
        model_response = model_response.lower()
    except Exception:
        return None
    if 'poor' in model_response:
        return 1
    if 'weak' in model_response:
        return 2
    if 'moderate' in model_response:
        return 3
    if 'good' in model_response:
        return 4
    if 'excellent' in model_response:
        return 5


def batch_iter(ls: list[T], batch_size: int) -> Generator[list[T], int, None]:
    if batch_size < 1:
        raise ValueError('batch size must be an integer greater than 0')
    for i in range(0, len(ls), batch_size):
        yield ls[i : i + batch_size]


def iserror(obj: Any) -> bool:
    return isinstance(obj, BaseException)


def handle_results(
    questions: list[dict],
    results: list[list[dict] | BaseException],
    parser_func: Callable[[dict], int | None],
    metric: str,
) -> tuple[list[dict], list[dict]]:
    """Parses the responses of the llm and returns a tuple of the result.

    The res_tuple[0] contains the evaluated questions, res_tuple[1] contains the questions and their
    corresponding errors.
    """
    if len(questions) != len(results):
        raise ValueError(
            'The result and question lists must have matching lengths'
        )
    errors = []
    parsed = []
    for q, r in zip(questions, results):
        if iserror(r):
            err = q.copy()
            err['error'] = f'{type(q).__name__}: {str(r)}'
            errors.append(err)
            continue
        p = parser_func(r)
        if p is None:
            err = q.copy()
            err['error'] = f'Unparsable response'
            errors.append(err)
            continue
        succ = q.copy()
        succ[metric] = p
        parsed.append(succ)
    return parsed, errors


async def eval_via_google_api(
    api_key: str,
    model: str,
    questions: list[dict],
    metric: str,
) -> list[dict]:
    if metric == 'question_relevance':
        format_prompt = format_question_relevance_prompt
        parse_response = parse_question_relevance_response
    elif metric == 'question_clarity':
        format_prompt = format_question_clarity_prompt
        parse_response = parse_question_clarity_response
    elif metric == 'answer_readability':
        format_prompt = format_answer_readability_prompt
        parse_response = parse_answer_readibility_response
    elif metric == 'distractor_quality':
        format_prompt = format_distractor_quality_prompt
        parse_response = parse_distractor_quality_response
    elif metric == 'incorrect_option_plausability':
        format_prompt = format_incorrect_option_plausability_prompt
        parse_response = parse_incorrect_option_plausibility_response
    else:
        raise ValueError('Metric not supported')
    batch_size = 3
    rated: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=25) as session:
            for qs in tqdm.tqdm(
                batch_iter(questions, batch_size),
                desc=f"Evaluating questions' {metric}",
                total=math.ceil(len(questions) / batch_size),
            ):
                max_tries = 3
                not_all_complete = True
                tries = 1
                rated_questions: list[dict] = []
                while not_all_complete and tries < max_tries:
                    convos = [format_prompt(q) for q in qs]
                    tasks = [
                        make_request(session, api_key, model, c)
                        for c in convos
                    ]
                    res = await asyncio.gather(*tasks, return_exceptions=True)
                    correct, errors = handle_results(
                        qs, res, parse_response, metric
                    )
                    rated_questions.extend(correct)
                    if len(errors) > 0:
                        for e in errors:
                            if '429' in e['error']:
                                await asyncio.sleep(15)
                        list(map(lambda e: print(e['error']), errors))
                        qs = list(map(lambda e: drop_key(e, 'error'), errors))
                        tries += 1
                    not_all_complete = False
                rated.extend(rated_questions)
    except KeyboardInterrupt:
        print('Received interrupt, stopping gently')
    except Exception as e:
        print(f'Unexpected error: {e}, shutting down safely')
    for r in rated:
        r['evaluator_model'] = model
    return rated


def drop_key(d: dict[str, Any], key: str) -> dict[str, Any]:
    d.pop(key)
    return d


def flatten_question_record(record: dict) -> dict:
    q = json.loads(record['question'])
    res = record.copy()
    res.pop('question')
    for k, v in q.items():
        res[k] = v
    return res


def records_to_df(records) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'db_path',
        type=str,
        help='The path to the sqlite db containing responses',
    )
    parser.add_argument(
        '--run_name',
        required=False,
        type=str,
        default='eval_only_meta-llama/Meta-Llama-3-8B-Instruct',
        help='The name of the run that will be evaluated, default: eval_only_meta-llama/Meta-Llama-3-8B-Instruct',
    )
    parser.add_argument(
        '--evaluator_model',
        type=str,
        default='gemini-1.5-flash',
        help='What model to use for evaluation, default: gemini-1.5-flash',
    )
    parser.add_argument(
        '--model_provider',
        type=str,
        help='What LLM provider to use, default: google',
        default='google',
        choices=['google'],
    )
    parser.add_argument(
        '--metric',
        default='question_relevance',
        choices=[
            'question_relevance',
            'question_clarity',
            'answer_readability',
            'distractor_quality',
            'incorrect_option_plausability',
        ],
        help='the metric by which the questions will be evaluated, default: question_relevance',
        type=str,
    )
    report_args = parser.add_argument_group(
        title='report arguments',
        description='Arguments that control the reporting behaviour of results',
    )
    report_args.add_argument(
        '--report_only',
        action='store_true',
        help='Only print the report of previous eval runs',
    )
    report_args.add_argument(
        '--group_by_lang',
        action='store_true',
        help='additionally group by question language',
    )
    report_args.add_argument(
        '--report_save_path',
        type=str,
        help='if specified, the report will be saved at the given path as csv',
    )
    args = parser.parse_args()
    conn = connect_to_db(args.db_path)
    create_eval_table(conn)
    create_question_metadata_table(conn)
    create_valid_questions_view(conn)
    create_aggregate_stats_view(conn)
    if not args.report_only:
        if args.metric in ['answer_readability', 'distractor_quality']:
            questions = get_data_to_eval(
                conn,
                run_name=args.run_name,
                metric=args.metric,
                evaluator_model_name=args.evaluator_model,
                question_types=['mcma', 'mcsa'],
            )
        elif args.metric == 'incorrect_option_plausability':
            questions = get_data_to_eval(
                conn,
                run_name=args.run_name,
                metric=args.metric,
                evaluator_model_name=args.evaluator_model,
                question_types=['boolean'],
            )
        else:
            questions = get_data_to_eval(
                conn,
                run_name=args.run_name,
                metric=args.metric,
                evaluator_model_name=args.evaluator_model,
            )
        questions = list(map(flatten_question_record, questions))

        if args.model_provider == 'google':
            api_key = os.environ['GOOGLE_AI_TOKEN']
            evaluated = asyncio.run(
                eval_via_google_api(
                    api_key,
                    'gemini-1.5-flash',
                    questions,
                    args.metric,
                )
            )
        else:
            raise ValueError(
                f'model provider: {args.model_provider} is not known'
            )
        # print(*evaluated, sep='\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        save_evaluated_questions(conn, evaluated, args.metric)
    res = get_aggregate_report(conn, args.group_by_lang)
    df = records_to_df(res)
    print(df)
    if args.report_save_path is not None:
        df.to_csv(args.report_save_path, index=False)
