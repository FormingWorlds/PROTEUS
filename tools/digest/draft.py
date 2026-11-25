"""Draft the digest entries with the model from the collected JSON only.

Usage::

    python tools/digest/draft.py collected.json draft.json

The model runs as ``claude -p`` with no tools; the instruction and the data both
go on stdin, because a prompt argument makes the CLI ignore piped input. Its
structured output must follow ``entry_schema.json``. The CLI starts in an empty
folder, so no project instruction file of the checkout joins the prompt. A window with no merged
pull request and no release skips the model and gives an empty draft.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from check import check, is_empty
from common import DIGEST_DIR, DigestError, load_settings

PR_FIELDS = ('ref', 'status', 'title', 'body', 'author', 'labels', 'size')
RELEASE_FIELDS = ('ref', 'name', 'body', 'author', 'prerelease')
COMMIT_FIELDS = ('ref', 'title', 'body', 'author', 'size')
FACT_FIELDS = ('id', 'pr', 'kind', 'change', 'key', 'old', 'new')


def model_input(data: dict) -> dict:
    """Return the part of the collected data the model sees: no URLs, no commit ids."""
    return {
        'window': {k: data['window'][k] for k in ('since', 'until')},
        'pull_requests': [{k: p[k] for k in PR_FIELDS} for p in data['prs']],
        'releases': [{k: r[k] for k in RELEASE_FIELDS} for r in data['releases']],
        'commits': [{k: c[k] for k in COMMIT_FIELDS} for c in data.get('commits', [])],
        'facts': [{k: f[k] for k in FACT_FIELDS} for f in data.get('facts', [])],
    }


def build_prompt(data: dict) -> str:
    """Instruction text followed by the model input as JSON."""
    instruction = (DIGEST_DIR / 'prompt.md').read_text()
    return f'{instruction}\n\n```json\n{json.dumps(model_input(data))}\n```\n'


def repair_prompt(prompt: str, draft: dict, fails: list[str]) -> str:
    """The first prompt, the first answer and its check failures, asking for a full corrected answer."""
    answer = json.dumps({'entries': draft['entries']})
    listed = '\n'.join(f'- {f}' for f in fails)
    return (
        f'{prompt}\nYour previous answer was:\n\n```json\n{answer}\n```\n\n'
        f'It failed these checks:\n\n{listed}\n\nReturn the full corrected answer. Cite a fact '
        'only together with the pull request it belongs to, and use only numbers from cited sources.\n'
    )


def run_model(prompt: str, model: str) -> dict:
    """Run the CLI once and return its structured output.

    Raises
    ------
    DigestError
        On a failed call, an error result or a missing structured output.
    """
    schema = (DIGEST_DIR / 'entry_schema.json').read_text()
    cmd = [
        os.environ.get('CLAUDE_BIN', 'claude'),
        '-p',
        '--model',
        model,
        '--tools',
        '',
        '--no-session-persistence',
        '--output-format',
        'json',
        '--json-schema',
        schema,
    ]
    with tempfile.TemporaryDirectory() as empty_dir:
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=1500, cwd=empty_dir
        )
    if proc.returncode != 0:
        raise DigestError(
            f'model call failed with code {proc.returncode}: {json.dumps(proc.stderr[-2000:])}'
        )
    out = json.loads(proc.stdout)
    if out.get('is_error') or not isinstance(out.get('structured_output'), dict):
        raise DigestError(f'model gave no structured output: {json.dumps(proc.stdout[-2000:])}')
    return out['structured_output']


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    src, dst = argv or sys.argv[1:3]
    with open(src) as fh:
        data = json.load(fh)
    if is_empty(data):
        draft = {'entries': [], 'empty': True}
    else:
        model, prompt = load_settings()['model'], build_prompt(data)
        draft = {**run_model(prompt, model), 'empty': False}
        fails = check(data, draft)
        if fails:
            print(f'{len(fails)} check failures; one repair round')
            draft = {**run_model(repair_prompt(prompt, draft, fails), model), 'empty': False}
            draft['first_failures'] = fails
    with open(dst, 'w') as fh:
        json.dump(draft, fh)
    print(f'{len(draft["entries"])} entries')
    return 0


if __name__ == '__main__':
    sys.exit(main())
