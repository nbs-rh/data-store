import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import generate

@scorer(metrics=[mean()])
def prediction():
    async def score(state, target):
        text = state.output.completion.strip().upper().rstrip('.!')
        if text not in ('TRUE', 'FALSE'):
            raise ValueError(f'Expected binary prediction, got {text!r}')
        return Score(value=int(text == 'TRUE'), answer=text)
    return score

@task
def arithmetic_verification():
    rows = [json.loads(line) for line in Path('/test_data/test-200.jsonl').read_text().splitlines()]
    samples = [Sample(id=r['sample_id'], input=r['input'], target=r['target'], metadata={'label': r['label']}) for r in rows]
    return Task(dataset=samples, solver=generate(), scorer=prediction())
