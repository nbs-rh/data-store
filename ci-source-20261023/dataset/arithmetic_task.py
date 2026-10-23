import json
import re
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig

@scorer(metrics=[mean()])
def prediction():
    async def score(state, target):
        raw = state.output.completion.strip().upper()
        match = re.search(r'\b(TRUE|FALSE)\b', raw)
        if match is None:
            raise ValueError(f'Expected binary prediction, got {raw!r}')
        text = match.group(1)
        return Score(value=int(text == 'TRUE'), answer=text)
    return score

@task
def arithmetic_verification():
    rows = [json.loads(line) for line in Path('/test_data/test-200.jsonl').read_text().splitlines()]
    samples = [Sample(id=r['sample_id'], input=r['input'], target=r['target'], metadata={'label': r['label']}) for r in rows]
    return Task(dataset=samples, solver=generate(), scorer=prediction(), config=GenerateConfig(max_tokens=32, temperature=0))
