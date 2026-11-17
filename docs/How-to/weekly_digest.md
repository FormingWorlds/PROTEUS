# Weekly ecosystem digest

The developers of the PROTEUS ecosystem (PROTEUS and its module repositories) read a short digest at the start of their Tuesday meeting. A GitHub Actions workflow in this repository writes it each week. The digest is a selection of the pull requests merged since the last digest, the open pull requests that need review, and the new releases. When posting is switched on, it goes to the FormingWorlds Discussions tab, in the category Weekly digest.

A language model drafts the text from the collected data. A script checks every entry before anything is posted: each pull request, release, number and config key in an entry must come from the collected data. The first line of the post says that a workflow wrote it and that it is a selection. The last paragraph says for how many pull requests the model read the text and for how many only the title. Read the linked pull requests before you act on an entry.

This page is for people who run or change the workflow. It explains what the workflow does, how to make a dry run (a run that writes the digest and posts nothing), and how to switch posting on or off.

## What the workflow does

The workflow is `.github/workflows/weekly-digest.yml`, with its scripts in `tools/digest/`. It runs every Monday at 22:23 UTC (23:23 in Amsterdam in winter, 00:23 on Tuesday in summer). You can also start it by hand. It has 4 jobs: collect, draft, a job that checks and posts, and a last job that reports a failure.

A digest covers a window, a period of time. The window starts where the window of the last digest post ended. `collect.py` reads that time from a hidden marker (an HTML comment) at the end of the post. It trusts only a marker in a post by `github-actions`. The window ends at the time of the run. With no earlier post, the window is the last 7 days.

### Collect the data

`collect.py` reads the pull requests and releases of every repository in `tools/digest/repos.toml` through the GitHub API. It stops if a repository is private or archived. Links, images, HTML, URLs and `@` mentions are removed from the titles and texts of pull requests and releases, because that text is untrusted input.

For each merged pull request that changes a tracked file, `facts.py` compares the file before and after the merge. A tracked file is one listed in `repos.toml` (see the settings table below). Each change becomes a fact, which is one of:

- a config key that was added or removed, or whose default, choices, bounds or description changed;
- a helpfile column (a column of the PROTEUS output table) that was added or removed, or whose unit changed;
- a dependency that was added or removed, or whose version changed. This covers a new floor (the lowest version a package accepts), a Julia `[compat]` entry in `Project.toml`, and a changed module pin (a module version under `tool.proteus.modules` in `pyproject.toml`);
- a key that was added, removed or changed in another tracked TOML file.

A file that gains more than 8 keys in one pull request gives one fact for the added keys. A file that loses more than 8 keys gives one fact for the removed keys. Each of these facts holds the number of keys and their names, cut at 600 characters.

### Draft the entries

`draft.py` gives the collected data and `prompt.md` to the model, which returns entries in a fixed form (`entry_schema.json`). The model runs with no tools, no GitHub token and in an empty folder. If the checks below fail, the model gets the failures once and returns a corrected answer. If the corrected answer fails the checks too, the job that checks and posts fails the run.

### Check the draft

`check.py` fails the run, and nothing is posted, if:

- the model returns no entry for a window that has a merged pull request or a release;
- an entry cites a pull request, release or fact that is not in the collected data;
- an entry cites no pull request and no release, or cites a release in the section In review, coming soon;
- an entry cites a fact but not the pull request that the fact comes from;
- an entry names a login that is not an author of a cited item;
- an entry cites an open pull request outside the section In review, coming soon, or a merged pull request inside it;
- an entry in the section Changes that affect your results cites no fact and no release;
- an entry outside the section Installing and running carries the action marker;
- the title or text of an entry holds a URL, an `@`, square brackets, HTML, an HTML entity, a `#` or `GH-` reference, a commit id, a backtick, an invisible character or a special number character (superscripts, fractions);
- the title or text uses a number that is not in the title or text of a cited pull request or release (the values of cited facts do not count);
- the title or text names a config key or helpfile column that is not in a cited fact;
- a sentence says what a setting is now and holds the old value of a cited change but not the new one.

### Render and post

`render.py` writes the post:

- The config keys, their values, the links, the dates and the mentions come from the collected data, not from the model text.
- The model text is escaped, so any markdown in it shows as plain text.

`post.py` posts only when posting is switched on:

- It does not post again when the newest digest post covers the same window or has the same title, so a rerun of a run that posted, failed jobs or all jobs, makes no second post.
- After a new post, it sends the link to Discord if the repository has the secret `DISCORD_WEBHOOK_URL` (a GitHub repository secret that holds the Discord webhook address). A failed Discord message is a warning, because the post stands.

If a job fails, the last job sends a message to Discord when the secret exists.

## Sections of a digest

The sections appear in the order of the table below. The first section is a table with the columns Setting, Change (old value → new value), Merged and Links. A section without entries is left out.

| Section | What goes in it |
| --- | --- |
| Changes that affect your results | Changes after which a run with the same config gives a different result, with the config changes as facts |
| New capabilities | New physics, modules, options and tools |
| Installing and running | Installation, environments, platforms and the command-line interface; an entry that every user must act on starts with **Action for everyone:** |
| Data and licensing | Reference data, downloads and licences |
| Design choices to look at | New interfaces, defaults, coupling choices and dependencies that maintainers can check against their plans |
| In review, coming soon | Open pull requests in review (drafts are left out) |

## Mentions

The post names each author as `@login`, which notifies that person. The rules:

- Each person is mentioned once per post; later names are plain text.
- Bots and organisations are never mentioned (the list `bots` in `repos.toml`, and every account that GitHub marks as a bot).
- The author of an open pull request that was in the last digest and is still open is not mentioned again for it.
- A dry run shows the mentions in code spans, so nobody is notified.

## Make a dry run

A manual run with all inputs empty and the box **Post the digest** unticked is a dry run of the current window. To make a dry run for a past window:

1. Open the Actions tab and select **Weekly ecosystem digest**.
2. Click **Run workflow**.
3. Enter **Window start** and **Window end** as dates or times in UTC, for example `2026-09-01` and `2026-09-30`. If you set only the end, the window starts 7 days before it. If you set only the start, the window ends now.
4. Read the digest in the run summary, or download the artifact `digest` and open `out/digest.md`.

The artifact `digest` (a file set that you can download from the run page) holds `collected.json`, `draft.json` and `check.txt`. If the check passes, it also holds `out/digest.md`, and the digest also appears in the run summary. If the check fails, there is no `out/digest.md`: read `check.txt` or the run summary instead.

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

To add or remove a repository, edit `repos` in the same file. Each repository must be public and must not be archived, and no two repositories may have the same name after the slash. The first repository in the list receives the post, so keep `FormingWorlds/PROTEUS` first.

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
| `tracked` | Files per repository whose changes become facts; `pyproject.toml` and `Project.toml` are always tracked. A listed file must exist on the default branch of its repository, or the run stops |

The run stops if a setting is missing or has the wrong type or range.

## Limits

- A fact comes from the merge commit of a pull request and the commit before it. For a pull request merged as several rebased commits, only the last commit is compared.
- Only the first 100 changed files of a pull request are read, so a larger pull request can miss a fact.
- Only pull requests whose base is the default branch of their repository are read, merged or open. A pull request closed without a merge does not appear.
- Open pull requests count only when they were last updated after the window start, so an open pull request with no activity drops out of the digest.
- A window with no merged pull request and no release gives a post that says Quiet week and lists no open pull requests, even when some are open.
- A pull request that is a draft when the digest is made is left out of the open pull requests, even if it was ready for review during a back-dated window.
- In a back-dated window, a pull request that was created before the window end and merged after it counts as open, unless it is a draft now.
- At most the 20 newest releases of each repository are read.
