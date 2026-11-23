# Hacker News & GitHub Trending Daily Digests

Zero-dependency GitHub Actions that send two clean daily emails:

1. **Hacker News Daily Digest (`hn_digest.py`)** — rolling 24h top stories, article TL;DRs, HN comment synthesis, **Spiciest Debate** spotlight, and 7-day dedup via Actions Cache.
2. **GitHub Trending Daily Digest (`gh_trending.py`)** — daily trending repositories, README-grounded TL;DRs, **Breakout Repo** spotlight, language & star velocity stats, and 7-day dedup via Actions Cache.

Both scripts use only the Python standard library, so there is nothing to `pip install`.

## How it works

```
Algolia HN API (24h stories + top comments)
  + Article excerpts (stdlib HTMLParser)
  + Rolling 7-day dedup (Actions Cache)
        │
        ▼
  hn_digest.py ──▶ LLM (structured JSON) ──▶ SMTP ──▶ your inbox
```

## Configuration

All configuration is via environment variables. In GitHub, set **Secrets** for
anything sensitive and **Variables** for the rest
(`Settings ▸ Secrets and variables ▸ Actions`).

### LLM

| Variable        | Where     | Required | Notes                                                        |
| --------------- | --------- | -------- | ------------------------------------------------------------ |
| `LLM_API_KEY`   | Secret    | ✅       | API key for your provider.                                   |
| `LLM_PROVIDER`  | Variable  | optional | `openai` (default), `gemini`, or `custom`.                   |
| `LLM_MODEL`     | Variable  | optional | Defaults: `gpt-4o-mini` (openai), `gemini-3.8-flash` (gemini). |
| `LLM_BASE_URL`  | Variable  | only for `custom` | Any OpenAI-compatible base URL (OpenRouter, Groq, local, …). |

### Digest & Enrichment

| Variable         | Where    | Required | Notes                                                                      |
| ---------------- | -------- | -------- | -------------------------------------------------------------------------- |
| `TOP_N`          | Variable | optional | Number of stories. Default `15`.                                           |
| `FETCH_MODE`     | Variable | optional | `24h` (default, rolling past 24 hours) or `front_page` (current snapshot). |
| `DEDUP_ENABLED`  | Variable | optional | `true` (default) to skip stories sent in the last 7 days via Actions Cache. |
| `STATE_FILE`     | Variable | optional | Path to rolling seen-IDs JSON. Default `state/seen.json`.                  |
| `FETCH_ARTICLES` | Variable | optional | `true` (default) to fetch article excerpts for grounded 1-line TL;DRs.     |
| `COMMENT_TOP_K`  | Variable | optional | Number of top stories (+ spiciest debate) to fetch HN comments for. Default `8` (`0` disables). |

### GitHub Trending (`gh_trending.py`)

| Variable           | Where    | Required | Notes                                                                       |
| ------------------ | -------- | -------- | --------------------------------------------------------------------------- |
| `GH_TOP_N`         | Variable | optional | Number of trending repos to include. Default `10`.                          |
| `GH_LANGUAGE`      | Variable | optional | Optional language slug (e.g. `python`, `rust`, `typescript`). Blank = all.  |
| `GH_SINCE`         | Variable | optional | `daily` (default), `weekly`, or `monthly`.                                  |
| `GH_DEDUP_ENABLED` | Variable | optional | `true` (default) to skip repos sent in the last 7 days via Actions Cache.   |
| `GH_STATE_FILE`    | Variable | optional | Path to rolling seen-repos JSON. Default `state/gh_seen.json`.              |
| `GH_FETCH_READMES` | Variable | optional | `true` (default) to fetch README excerpts for grounded 1-line summaries.    |

### Email (SMTP)

| Variable        | Where    | Required | Notes                                         |
| --------------- | -------- | -------- | --------------------------------------------- |
| `SMTP_HOST`     | Secret   | ✅       | e.g. `smtp.gmail.com`.                        |
| `SMTP_PORT`     | Secret   | optional | Default `587`. Use `465` for implicit SSL.    |
| `SMTP_USER`     | Secret   | optional | SMTP username (omit for unauthenticated).     |
| `SMTP_PASS`     | Secret   | optional | SMTP password / app password.                 |
| `EMAIL_FROM`    | Secret   | ✅       | Sender address.                               |
| `EMAIL_TO`      | Secret   | ✅       | Recipient(s), comma-separated.                |
| `SMTP_STARTTLS` | Variable | optional | `false` to disable STARTTLS. Default `true`.  |

> If SMTP variables are missing, the script prints the digest to stdout instead
> of emailing — handy for testing.

## Provider examples

**OpenAI** — `LLM_PROVIDER=openai`, `LLM_API_KEY=sk-...`
**Gemini** — `LLM_PROVIDER=gemini`, `LLM_API_KEY=...` (uses Google's
OpenAI-compatible endpoint; defaults to `gemini-3.8-flash`).
**Anything else** (OpenRouter, Groq, DeepSeek, Ollama, …) —
`LLM_PROVIDER=custom`, `LLM_BASE_URL=https://...`, `LLM_MODEL=...`

### Email setup: Brevo (recommended)

[Brevo](https://www.brevo.com/) has a free tier (300 emails/day, **no credit
card**) and gives you a disposable, scoped **SMTP key** instead of exposing a
personal account password — safer to store in CI.

1. Sign up at brevo.com.
2. **Senders, Domains & Dedicated IPs → Senders** → add and verify a sender
   address (click the link Brevo emails you). Optionally verify your domain
   (SPF/DKIM) for better deliverability.
3. **SMTP & API → SMTP** → **Generate a new SMTP key** and copy it.
4. Set these GitHub secrets:

   | Secret      | Value                          |
   | ----------- | ------------------------------ |
   | `SMTP_HOST` | `smtp-relay.brevo.com`         |
   | `SMTP_PORT` | `587`                          |
   | `SMTP_USER` | your Brevo login email         |
   | `SMTP_PASS` | the generated SMTP key         |
   | `EMAIL_FROM`| your **verified** sender       |
   | `EMAIL_TO`  | recipient(s), comma-separated  |

> Free-tier emails include a small Brevo footer. Works with the existing code —
> no changes needed.

### Gmail tip

Use an [App Password](https://support.google.com/accounts/answer/185833)
(not your normal password) with `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`.

## Run locally

```bash
cp .env.example .env      # fill in your keys
set -a; source .env; set +a
python hn_digest.py       # Hacker News digest
python gh_trending.py     # GitHub Trending digest
```

Without `LLM_API_KEY` set, it will error on the LLM step; without SMTP it just
prints to the terminal. Run all unit tests with:

```bash
python3 -m unittest -v
```

## Schedule

- **HN Daily Digest** runs daily at **13:00 UTC** (`.github/workflows/hn-digest.yml`).
- **GitHub Trending Daily Digest** runs daily at **13:05 UTC** (`.github/workflows/gh-trending.yml`).

Both can be triggered manually anytime from the **Actions** tab (`workflow_dispatch`).
