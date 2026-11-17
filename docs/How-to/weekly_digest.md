# Weekly ecosystem digest

The developers of the PROTEUS ecosystem (PROTEUS and its module repositories) read a short digest at the start of their Tuesday meeting. A GitHub Actions workflow in this repository writes it each week. The digest is a selection of the pull requests merged since the last digest, the open pull requests that need review, and the new releases. When posting is switched on, it goes to the FormingWorlds Discussions tab, in the category Weekly digest.

A language model drafts the text from the collected data, and a script checks every entry before anything is posted. The post says so in its first line. Read the linked pull requests before you act on an entry.

This page explains what the workflow does, how to make a dry run (a run that writes the digest and posts nothing), and how to switch posting on or off.

## What the workflow does

The workflow is `.github/workflows/weekly-digest.yml`, with its scripts in `tools/digest/`. It runs every Monday at 22:23 UTC (23:23 in Amsterdam in winter, 00:23 on Tuesday in summer). You can also start it by hand. It has 4 jobs: collect, draft, a job that checks and posts, and a last job that reports a failure.

The window of a digest is the period it covers. It starts at the end of the window of the last digest post, which the script reads from a hidden marker at the end of that post. It trusts only a marker in a post by `github-actions`. With no earlier post, the window is the last 7 days.

### Collect

`collect.py` reads the pull requests and releases of every repository in `tools/digest/repos.toml` through the GitHub API. It stops if a repository is private or archived. Links, images, HTML and `@` mentions are removed from the pull-request text, because that text is untrusted input.

For each merged pull request that changes a tracked file, `facts.py` compares the file before and after the merge. A tracked file is one listed in `repos.toml` (see the settings table below). Each change becomes a fact, which is one of:

- a config key that was added or removed, or whose default, choices, bounds or description changed;
- a helpfile column (a column of the PROTEUS output table) that was added or removed, or whose unit changed;
- a changed dependency version, for example a new floor (the lowest version a package accepts), or a changed module pin;
- a changed value in another tracked TOML file.

### Draft

`draft.py` gives the collected data and `prompt.md` to the model, which returns entries in a fixed form (`entry_schema.json`). The model runs with no tools, no GitHub token and in an empty folder. If the checks below fail, the model gets the failures once and returns a corrected answer.

### Check, render and post

`check.py` fails the run, and nothing is posted, if an entry:

- cites a pull request, release, fact or login that is not in the collected data;
- holds a link, an `@`, markdown, HTML, an HTML entity, a `#` or `GH-` reference, a commit id or a backtick in its text;
- uses a number that is not in a cited source;
- names a config key or helpfile column that is not in a cited fact, or repeats the old value of a cited change.

`render.py` then writes the post. The config keys, their values, the links, the dates and the mentions come from the collected data, not from the model text, and the model text is escaped so that it shows as plain text. `post.py` posts only when posting is switched on. It does not post again for a window that is already posted, so a rerun of a failed job is safe. It then sends the link to Discord if the repository has the secret `DISCORD_WEBHOOK_URL`; a failed Discord message is a warning, because the post stands.

A run that reaches the check uploads the artifact `digest` (a file set that you can download from the run page) with the collected data, the draft and the check result. A run that passes the check adds the rendered digest, `out/digest.md`, and also shows it in the run summary. If a job fails, the last job sends a message to Discord when the webhook secret exists.

## Sections of a digest

| Section | What goes in it |
| --- | --- |
| Changes that affect your results | Changes after which a run with the same config gives a different result, with the config changes as facts |
| New capabilities | New physics, modules, options and tools |
| Installing and running | Installation, environments, platforms and the command-line interface |
| Data and licensing | Reference data, downloads and licences |
| Design choices to look at | New interfaces, defaults, coupling choices and dependencies that maintainers can check against their plans |
| Open for review | Open pull requests that need review (drafts are left out) |

## Mentions

The post names each author as `@login`, which notifies that person. The rules:

- Each person is mentioned once per post; later names are plain text.
- Bots and organisations are never mentioned (the list `bots` in `repos.toml`, and every account that GitHub marks as a bot).
- The author of an open pull request that was in the last digest and is still open is not mentioned again for it.
- A run that does not post shows the mentions in code spans, so nobody is notified.

## Make a dry run

A manual run with all inputs empty and the box **Post the digest** unticked is a dry run of the current window. To make a dry run for a past window:

1. Open the Actions tab and select **Weekly ecosystem digest**.
2. Click **Run workflow**.
3. Enter **Window start** and **Window end** as dates or times in UTC, for example `2026-09-01` and `2026-09-30`. If you set only the end, the window starts 7 days before it.
4. Read the digest in the run summary, or download the artifact `digest` and open `out/digest.md`.

A run with a window start or end never posts. You can run the same scripts locally with a token that can read public repositories:

```bash
export GITHUB_TOKEN=$(gh auth token)
python tools/digest/collect.py --out collected.json --since 2026-09-01 --until 2026-09-30
python tools/digest/facts.py collected.json
python tools/digest/draft.py collected.json draft.json
python tools/digest/check.py collected.json draft.json
python tools/digest/render.py collected.json draft.json out
```

The draft step needs the `claude` command line and a login or `CLAUDE_CODE_OAUTH_TOKEN`. Read the result in `out/digest.md`.

## Switch posting on or off

`publish` in `tools/digest/repos.toml` controls posting:

- `publish = false`: every run is a dry run.
- `publish = true`: the scheduled run posts. A manual run posts only when its box **Post the digest** is ticked and no window is given.

To switch posting on:

1. Check that the repository secret `CLAUDE_CODE_OAUTH_TOKEN` exists. The draft job needs it. The secret `DISCORD_WEBHOOK_URL` is optional.
2. Check that the Discussions category with the slug in `category` exists in the PROTEUS repository. A run that posts fails without it.
3. Set `publish = true` in a pull request.

To add or remove a repository, edit `repos` in the same file. Each repository must be public and must not be archived. The first repository in the list receives the post, so keep `FormingWorlds/PROTEUS` first.

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
- Only pull requests merged into the default branch count as merged; a pull request closed without a merge does not appear.
- Open pull requests count only when they were updated in the window, so an open pull request with no activity drops out of the digest.
- A pull request that is a draft when the digest is made is left out of the open pull requests, even if it was ready for review during a back-dated window.
- At most the 20 newest releases of each repository are read.
