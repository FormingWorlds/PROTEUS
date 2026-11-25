# Weekly ecosystem digest

The weekly digest is a short post that tells the developers of the PROTEUS ecosystem (PROTEUS and its module repositories) what changed. It lists the pull requests merged since the last digest, the open pull requests that need review, and the new releases. Developers read it at the start of their weekly meeting on Tuesday, so a change that alters model results, or that every user must act on, does not stay hidden in a long list of pull requests.

A GitHub Actions workflow in this repository writes the digest and, when posting is switched on, posts it to the FormingWorlds Discussions tab (the forum of a GitHub repository), in the category Weekly digest. This page is for people who run or change the workflow. It explains what the workflow does, how to make a dry run (a run that writes the digest and posts nothing), how to switch posting on or off, and what to do when a run is red.

A language model drafts the text from the collected data. A script checks every entry before anything is posted: each pull request, release, number and config key in an entry must come from the collected data.

The post opens with a summary of what the digest covers:

- the collection window;
- the number of merged pull requests, open pull requests and releases;
- the repositories, each a link to its GitHub repository;
- whether the model read the text of every pull request or only of the largest ones (the others by title only);
- a link to the workflow run.

A dry run puts the line **Dry run: not posted.** before this summary. After the summary, the post says that it is a selection and that a workflow wrote it. Readers must read the linked pull requests before they act on an entry.

## What the workflow does

The workflow is `.github/workflows/weekly-digest.yml`, with its scripts in `tools/digest/`. It runs every Monday at 22:23 UTC (23:23 in Amsterdam in winter, 00:23 on Tuesday in summer). You can also start it by hand. It has 3 jobs, shown in the Actions tab as `Collect pull requests, releases and facts`, `Draft the entries` and `Check, render and post`. This page calls them the collect job, the draft job and the check job (its job id in the workflow is `publish`). The check job checks the draft, renders it, and posts it only in a live run. It uploads its files as the artifact `digest`, a file set that you can download from the run page. A scheduled run and a manual run with the box **Post the digest** ticked and no window share one queue: a run waits while another runs, and a new run cancels a run that waits. Every other manual run runs on its own and cannot cancel a waiting scheduled run.

A digest covers a window of time. The window starts where the window of the last digest post ended. `collect.py` reads that time from a hidden state marker (an HTML comment) at the end of the post. The state marker holds the window end and the run id. `collect.py` trusts only a state marker in a post by the login (GitHub user name) `github-actions`, the account the workflow posts as. The window ends at the time of the run. With no earlier post, the window is the last 7 days. A window that you set by hand, with a start or an end, is back-dated, and a back-dated run never posts.

### Collect the data

`collect.py` reads the pull requests and releases of every repository in `tools/digest/repos.toml` through the GitHub API. It stops if a repository is private or archived. Anyone with a GitHub account can write the titles and texts of pull requests and releases, so `collect.py` cleans that text before the model reads it. It reads only the first 20000 characters of each title or text. It removes format and control characters, HTML comments (a comment ends only at `-->`, and an unclosed comment removes the rest of the text), images, link targets, HTML tags, every `<`, `>` and `@`, and every word that holds `://` or `www.`. Link text stays, and so does the text around a `<` or `>` that does not start a tag, as in `T < 300 K`. Text inside markup, such as an attribute value or the content of a `<script>` element, can still reach the model. Of text in the form `[word](...)` only `word` stays, and of `![word](...)` nothing stays, also when the round brackets hold plain prose and not a link target. `check.py` and the link scan of the post step stop links, mentions, markup and numbers without a source in the post; they do not stop wording that such text steers.

An open pull request counts only when GitHub reports its author as an owner, member, collaborator or contributor of the repository (the author association). `collect.py` leaves out the other open pull requests and names each one, with its login and association, in the run log and the job summary. Merged pull requests count whatever the association.

### Find the docs pages of a pull request

For each merged pull request, `collect.py` also lists its docs pages. Each file `docs/<path>.md` of the pull request maps to `<site>/<path>.html`, and a space in the path becomes `%20` in the URL. The docs site of its repository comes from the table `docs` in `repos.toml`. A repository without a docs site in the table gives no pages.

A page counts only when one HEAD request (a request for the status of a page, without its content) to that URL answers 200 when the data is collected. A page that is not served is left out. So is a page of a pull request merged after the last docs deployment. A redirect is not followed, so a page that answers with a redirect is also left out. Each URL is asked once per run.

After a network error on a docs site (no HTTP answer, such as a timeout), the pages of that site that are not asked yet are left out without a request. An HTTP answer such as 404 does not stop the site.

A page name is also left out when it has a character other than an ASCII letter or digit, `_`, a space, `.`, `/` or `-`, or when it has no letter or digit.

A page that the pull request deletes is not a candidate: it gets no request, no link, no log line and no count. The run log names every other page left out, with the reason. The job summary gives the number of these pages for each docs site.

### Read the file changes as facts

For each merged pull request that changes a tracked file, `facts.py` compares the file before and after the merge. A tracked file is one listed in `repos.toml` (see [Settings in `repos.toml`](#settings-in-repostoml)). Each change becomes a fact: a change that the script reads from the files, not from the model. A fact is one of:

- a config key that was added or removed, or whose default, choices, bounds or description changed;
- a helpfile column (a column of the PROTEUS output table) that was added or removed, or whose unit changed;
- a dependency that was added or removed, or whose version changed. This covers a new floor (the lowest version a package accepts), a Julia `[compat]` entry in `Project.toml`, and a changed module pin (a module version under `tool.proteus.modules` in `pyproject.toml`);
- a key that was added, removed or changed in another tracked TOML file.

A file that gains more than 8 keys in one pull request gives one fact for the added keys. A file that loses more than 8 keys gives one fact for the removed keys. Each of these facts holds the number of keys and their names, cut at 600 characters. A date in a TOML file becomes text, and a longer array or table is cut at 600 characters.

A tracked file that does not parse, or is not valid UTF-8, gives no facts. The other tracked files of the same pull request still give their facts. `facts.py` names the pull request, the file and the error in the run log and the job summary, and the run goes on. A pull request with no fact cannot go into the section "Changes that affect your results" unless it cites a release. A tracked file that does not parse on the default branch gives no keys for the key check and is named the same way.

### Draft the entries

`draft.py` gives the collected data and `prompt.md` to the model, which returns entries in a fixed form (`entry_schema.json`). The model runs with no tools, no GitHub token and in an empty folder. If the checks below fail, the model gets the failures once and returns a corrected draft.

Each entry has an action flag, which the model sets when every user must act on the entry. The script then adds the words **Action for everyone:** in front of its text.

### Check the draft

`check.py` checks the corrected draft. The run fails, and nothing is posted, if one of the conditions below is true.

#### Checks on citations and form

- the model returns no entry for a window that has a merged pull request or a release;
- the draft says the window is empty although it has a merged pull request or a release, or says it has content although it has none;
- an entry misses a field, has a field of the wrong type, or names a section that does not exist;
- an entry cites a pull request, release or fact that is not in the collected data;
- an entry cites no pull request and no release, or cites a release in the section "Design choices to look at" or "In review, coming soon";
- an entry cites a fact but not the pull request that the fact comes from;
- an entry names a login that is not an author of a cited item;
- an entry cites an open pull request outside the sections "Design choices to look at" and "In review, coming soon", or a merged pull request inside them;
- an entry in the section "Changes that affect your results" cites no fact and no release;
- an entry outside the section "Installing and running" carries the action flag;
- the title of an entry is longer than 80 characters, or its text is longer than 400 characters.

#### Checks on the words of an entry

The check reads the title and text of an entry as the model wrote them, and as the post shows them (a `|` shows as `/`, a dash as a comma). The run fails when either form has:

- a link or a mention: a URL, an `@` or square brackets;
- markup: HTML, an HTML entity or a backtick;
- a reference: a `#` or `GH-` reference, or a word of 7 or more hex characters that can read as a commit id (`deadbeef`, `1234567`);
- a superscript;
- the words "Action for everyone", which only the script adds.

The run also fails when the text as the model wrote it has:

- an invisible character, a fraction or a letter number such as a Roman numeral;
- a number that is not in a cited pull request or release (cited fact values do not count);
- a config key or helpfile column that is not in a cited fact.

How the check reads numbers:

- It reads `v1.0`, `V100`, `Δ50`, `mode_42`, `No.5` and `.5` as the numbers 1.0, 100, 50, 42, 5 and 0.5.
- It also reads the 2 of `H2`.
- It does not read the digits of a formula of 2 or more element symbols (`H2O`, `CO2`).

#### Warnings

The check gives a warning when an entry states the old value of a cited config key change as the current value. A warning does not stop the run, and the post goes out. A sentence gives a warning when it names the config key and one of these is true:

- an assignment ("is now", "set to", "becomes", "equals", "uses") stands directly before the old value, and the new value is absent or marked as old ("from", "not", "instead of");
- the sentence reads "from" the new value "to" the old value.

The check prints each failure as one line, `FAIL` and a quoted JSON string, so entry text cannot break the line. Each warning is a `WARN` line and an annotation on the run page. The job summary and `check.txt` in the artifact `digest` hold the `FAIL` and `WARN` lines.

### Render the post

`render.py` writes the post:

- The config keys, their values, the links, the dates and the author names come from the collected data, not from the model text.
- Each author is a link to the GitHub profile (see [Authors](#authors)).
- A row of the section "Changes that affect your results" and an entry of the section "New capabilities" link the docs pages of their pull requests, at most 5, then the number of the other pages.
- The model text is escaped, so any markdown in it shows as plain text.
- An empty fact value shows as (empty), a null or missing one as (none), and a value of only `|` or backtick characters as (not shown).
- The title is the date of the Tuesday of the week (Monday to Sunday, Amsterdam time) in which the window ends. The scheduled Monday run takes the next day. A run from Tuesday to Sunday takes the Tuesday that is already past, the meeting of its own week.
- A body longer than 65,536 characters stops the run.

### Post it once

The post step runs only in a live run, a run that is allowed to post (see [Switch posting on or off](#switch-posting-on-or-off)). Before it posts, `post.py` repeats every guard:

- posting is switched on;
- the window is not back-dated;
- the run is scheduled, or its box **Post the digest** is ticked;
- the draft passes `check.py`;
- the body is not a dry run;
- the final body links only to the collected pull requests, releases and docs pages, the run, the repositories and the profiles of the authors, and mentions nobody.

`post.py` makes at most one post per meeting. It reads the newest digest post, and skips posting when that post has the same workflow run id (the number of the Actions run, which a rerun keeps), the same title, or a window end that is the same as or later than the window end of this run:

| Situation | Result |
| --- | --- |
| Rerun of failed jobs or of all jobs of the run that made the newest post, on any day | No second post |
| Second manual run for the same meeting | No second post |
| Rerun of the failed jobs of an older run after a newer post | No post, so the next window does not move back |
| Rerun of all jobs of an older run | It collects a new window from the newest post and posts for the meeting of its week, unless that meeting has a post |
| Manual run on a Monday before the scheduled run | It takes the title of the scheduled run, which then posts nothing; the next window starts at the end of the manual run, so no pull request is left out |

A skipped post gives a notice annotation on the run page, and the run stays green. After `post.py` creates the post, it checks the author in the answer of GitHub (see [If a run is red](#if-a-run-is-red)).

## Make a dry run

A manual run with all inputs empty and the box **Post the digest** unticked is a dry run of the current window. To make a dry run for a past window:

1. Open the Actions tab and select **Weekly ecosystem digest**.
2. Click **Run workflow**.
3. Enter **Window start** and **Window end** as dates or times in UTC, for example `2026-09-01` and `2026-09-30`. If you set only the end, the window starts 7 days before it. If you set only the start, the window ends now.
4. Read the digest in the job summary, or download the artifact `digest` and open `out/digest.md`.

The artifact `digest` holds `collected.json`, `draft.json` and `check.txt`. If the check passes, it also holds `out/digest.md`, and the digest also appears in the job summary. If the check fails, there is no `out/digest.md`: read `check.txt` or the job summary instead.

A run with a window start or end never posts. A dry run does not scan the final body for links: the scan runs only in the post step, on the live body. You can run the same scripts locally with a token that can read public repositories:

```bash
export GITHUB_TOKEN=$(gh auth token)
python tools/digest/collect.py --out collected.json --since 2026-09-01 --until 2026-09-30
python tools/digest/facts.py collected.json
python tools/digest/draft.py collected.json draft.json
python tools/digest/check.py collected.json draft.json
python tools/digest/render.py collected.json draft.json out
```

The draft step needs the `claude` command line and either a signed-in `claude` session or `CLAUDE_CODE_OAUTH_TOKEN`. Read the result in `out/digest.md`.

## Switch posting on or off

`publish` in `tools/digest/repos.toml` controls posting:

- `publish = false`: every run is a dry run.
- `publish = true`: the scheduled run posts. A manual run posts only when its box **Post the digest** is ticked and no window is given.

To switch posting on:

1. Check that the repository secret `CLAUDE_CODE_OAUTH_TOKEN` exists. The draft job needs it.
2. Check that the Discussions category with the slug in `category` exists in the PROTEUS repository. A run that posts fails without it.
3. Set `publish = true` in a pull request.

To add or remove a repository, edit `repos` in the same file. Each repository must be public and must not be archived, and no two repositories may have the same name after the slash. The first repository in the list receives the post, so keep `FormingWorlds/PROTEUS` first.

## If a run is red

If the collect job or the draft job is red, the check job does not run and there is no `check.txt`. Open the log of that job: the collect job names each open pull request that it left out, and the draft job prints the error text of the model call.

If the check job is red before the post step, nothing is posted. Read the check lines in the job summary, or open `check.txt` in the artifact `digest` (see [Make a dry run](#make-a-dry-run)).

If the post step is red with `not posted:`, nothing is posted, and the message names each failed guard. For link problems, the log has one `Link problem:` line for each link or mention in the final body that the collected data does not allow. The check stops such text earlier, so a link problem means a gap in `check.py`: read `out/digest.md` in the artifact `digest`.

A run that fails in the post step after GitHub created the post is red, and the post stays. A run is also red when the author of the post is not `github-actions`: the next run would not find that post, and a rerun of this run posts again. Look at the category before you start the run again.

## Sections of a digest

The sections appear in the order of the table below, and a section without entries is left out. The section "Changes that affect your results" is a table with the columns Setting, Change (the entry text, then old value → new value for each cited fact), Merged and Links (with the docs pages and the authors). An entry of the section "New capabilities" lists its docs pages after its text.

| Section | What goes in it |
| --- | --- |
| Changes that affect your results | Changes after which a run with the same config gives a different result, with the config changes as facts |
| New capabilities | New physics, modules, options and tools |
| Installing and running | Installation, environments, platforms and the command-line interface; an entry that every user must act on starts with **Action for everyone:** |
| Data and licensing | Reference data, downloads and licences |
| Design choices to look at | Open pull requests that change an interface between modules, a default, a coupling choice or a dependency, for maintainers to check against their plans; a merged design change goes to one of the 4 sections above |
| In review, coming soon | The other open pull requests in review by owners, members, collaborators and contributors (draft pull requests are left out) |

## Authors

Each entry ends with the authors of its pull requests and releases, each once:

- A person is a link to the GitHub profile, `[login](https://github.com/login)`. A link notifies nobody, and the post has no `@` mentions.
- A bot or an organisation is a code span: the logins in the list `bots` in `repos.toml`, every login that ends in `[bot]`, and every account that GitHub does not mark as a user.
- A deleted account has no login and shows as the word unknown, with no link and no code span.

## Settings in `repos.toml`

| Setting | Meaning |
| --- | --- |
| `publish` | `true` to post, `false` for dry runs only |
| `category` | Slug of the discussion category; a run that posts fails if it does not exist |
| `model` | Model that drafts the entries |
| `max_bodies` | Number of the largest pull requests (by changed lines) whose text the model reads; the others give their titles only. From 0 to 500 |
| `body_chars` | Characters kept of each pull-request or release text. From 1 to 20000 |
| `bots` | Logins shown as code spans, not as profile links |
| `repos` | Repositories in the digest, as `owner/name` |
| `docs` | Docs site per repository of `repos`, as `owner/name = "https://..."` with no `/` at the end; a page `docs/<path>.md` is served at `<site>/<path>.html` |
| `tracked` | Files per repository whose changes become facts; `pyproject.toml` and `Project.toml` are always tracked. A listed file must exist on the default branch of its repository, or the run stops |

The run stops if a setting is missing or has the wrong type or range.

## Limits

What a digest can miss:

- A fact comes from the merge commit of a pull request and the commit before it. For a pull request merged as several rebased commits, only the last commit is compared.
- For a pull request with more than 100 changed files, every tracked file of its repository is compared, and its docs pages come from its first 100 files only.
- A docs page is checked once, when the data is collected; a page that the docs deployment removes later stays linked.
- Only pull requests whose base is the default branch of their repository are read, merged or open. A pull request closed without a merge does not appear.
- An open pull request counts only when it was updated after the window start.
- A window with no merged pull request and no release gives a post that says Quiet week and lists no open pull requests, even when some are open.
- At most the 20 newest releases of each repository are read.
- The number check does not read digits after 2 or more letters (`step200`, `RK45`).
- The value warning covers only the old value in the wordings above. A third value ("the mixing length is now linear" for a change from nearest to constant), another wording or a sentence that does not name the config key gives no warning. The table next to the entry shows the fact.
- The author association is what the token of the run can see. A member with a private organisation membership can show as a contributor or with no association; an open pull request with no association is left out and named in the job summary.

Windows set by hand (back-dated):

- A pull request that is a draft when the digest is made is left out of the open pull requests, even if it was ready for review during the window.
- A pull request that was created before the window end and merged after it counts as open, unless it is a draft now.
