# Weekly ecosystem digest

Each week, a GitHub Actions workflow in this repository writes a digest of the PROTEUS ecosystem. The digest lists the pull requests merged since the last digest, the open pull requests that need review, and the new releases. When posting is switched on, it goes to the FormingWorlds Discussions tab in the category Weekly digest, before the Tuesday PROTEUS meeting.

The digest is written automatically: a language model drafts the text from the collected data, and a script checks every entry before anything is posted. The post says so in its first line. Read the linked pull requests before you act on an entry.

This page explains what the workflow does, how to make a dry run, and how to switch posting on or off.

## What the workflow does

The workflow is `.github/workflows/weekly-digest.yml`, with its scripts in `tools/digest/`. It runs every Monday at 22:23 UTC (23:23 in Amsterdam in winter, 00:23 on Tuesday in summer). You can also start it by hand.

It has 3 jobs:

1. **Collect** (`collect.py`, `facts.py`). The job reads the pull requests and releases of every repository in `tools/digest/repos.toml` through the GitHub API. It stops if a repository is private or archived. Links, images, HTML and `@` mentions are removed from the pull-request text, because that text is untrusted input. For each merged pull request that changes a tracked file, the job compares the file before and after the merge. Each change becomes a fact: a config key that was added, removed or given a new default, a new helpfile column, or a new dependency floor.
2. **Draft** (`draft.py`, `prompt.md`). The model reads only the collected data and returns entries in a fixed form (`entry_schema.json`). It runs with no tools and no GitHub token. If the checks below fail, the model gets the failures once and returns a corrected answer.
3. **Check, render and post** (`check.py`, `render.py`, `post.py`). The check fails the run, and posts nothing, if an entry:
    - cites a pull request, release, fact or login that is not in the collected data;
    - holds a link, an `@`, HTML, a `#` reference or a backtick in its text;
    - uses a number that is not in a cited source;
    - names a config key or helpfile column that is not in a cited fact.

    The script then writes the post. The settings, values, links, dates and mentions come from the collected data, not from the model text. The job posts only when posting is switched on, and then sends the link to Discord if the repository has the secret `DISCORD_WEBHOOK_URL`.

Every run uploads the rendered digest, the collected data, the draft and the check result as the artifact `digest`. If a run fails, a last job sends a message to Discord when the webhook secret exists.

The window of a digest starts at the end of the window of the last digest post, which the script reads from a hidden marker in that post. It trusts only a marker in a post by `github-actions`. With no earlier post, the window is the last 7 days.

## Sections of a digest

| Section | What goes in it |
| --- | --- |
| Changes that affect your results | Changes after which a run with the same config gives a different result, with the config changes as facts |
| New capabilities | New physics, modules, options and tools |
| Installing and running | Installation, environments, platforms and the command-line interface |
| Data and licensing | Reference data, downloads and licences |
| Design choices to look at | New interfaces, defaults, coupling choices and dependencies that maintainers can check against their plans |
| Open for review | Open pull requests that need review |

## Mentions

The post names each author as `@login`, which notifies that person. The rules:

- Each person is mentioned once per post; later names are plain text.
- Bots and organisations are never mentioned (the list `bots` in `repos.toml`, and every account that GitHub marks as a bot).
- The author of an open pull request that was in the last digest and is still open is not mentioned again for it.
- A dry run shows the mentions in code spans, so nobody is notified.

## Make a dry run

A dry run writes the digest and posts nothing. To make one for a past window:

1. Open the Actions tab and select **Weekly ecosystem digest**.
2. Click **Run workflow**.
3. Enter `since` and `until` as dates or times in UTC, for example `2026-09-01` and `2026-09-30`.
4. Download the artifact `digest` and read `out/digest.md`.

A run with `since` or `until` set never posts. You can run the same scripts locally with a token that can read public repositories:

```bash
export GITHUB_TOKEN=$(gh auth token)
python tools/digest/collect.py --out collected.json --since 2026-09-01 --until 2026-09-30
python tools/digest/facts.py collected.json
python tools/digest/draft.py collected.json draft.json
python tools/digest/check.py collected.json draft.json
python tools/digest/render.py collected.json draft.json out
```

The draft step needs the `claude` command line and a login or `CLAUDE_CODE_OAUTH_TOKEN`.

## Switch posting on or off

Posting is controlled by `publish` in `tools/digest/repos.toml`:

- `publish = false`: every run is a dry run.
- `publish = true`: the scheduled run posts. A manual run posts only when its `post` box is ticked and no window is given.

Change the line in a pull request. To add or remove a repository, edit `repos` in the same file. Each repository must be public and must not be archived.

## Settings in `repos.toml`

| Setting | Meaning |
| --- | --- |
| `publish` | `true` to post, `false` for dry runs only |
| `category` | Slug of the discussion category; a run that posts fails if it does not exist |
| `model` | Model that drafts the entries |
| `max_bodies` | Number of the largest pull requests (by changed lines) whose text the model reads; the others give their titles only. From 0 to 500 |
| `body_chars` | Characters kept of each pull-request or release text. From 1 to 20000 |
| `bots` | Logins that are never mentioned |
| `repos` | Repositories in the digest, as `owner/name` |
| `tracked` | Files per repository whose changes become facts; `pyproject.toml` and `Project.toml` are always tracked |

The run stops if a setting is missing or has the wrong type or range.

## Limits

- A fact comes from the merge commit of a pull request and the commit before it. For a pull request merged as several rebased commits, only the last commit is compared.
- Only the first 100 changed files of a pull request are read, so a larger pull request can miss a fact.
- Open pull requests count only when they were updated in the window, so an open pull request with no activity drops out of the digest.
- A back-dated window uses the present draft state of a pull request.
