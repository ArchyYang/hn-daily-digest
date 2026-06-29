# Hacker News Daily Digest

A zero-dependency GitHub Action that, once a day:

1. **Fetches the top Hacker News stories** (Algolia HN front-page API — free, no key).
2. **Summarizes them with an LLM** of your choice (OpenAI or Gemini, or any
   OpenAI-compatible endpoint via `custom`).
3. **Emails the digest** (HTML + plain text) over SMTP.

The script (`hn_digest.py`) uses only the Python standard library, so there is
nothing to `pip install`.

## How it works

```
Algolia HN API ──▶ hn_digest.py ──▶ LLM (chat/completions) ──▶ SMTP ──▶ your inbox
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
| `LLM_MODEL`     | Variable  | optional | Defaults: `gpt-4o-mini` (openai), `gemini-3-pro-preview` (gemini). |
| `LLM_BASE_URL`  | Variable  | only for `custom` | Any OpenAI-compatible base URL (OpenRouter, Groq, local, …). |

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

### Other

| Variable | Where    | Required | Notes                                |
| -------- | -------- | -------- | ------------------------------------ |
| `TOP_N`  | Variable | optional | Number of stories. Default `15`.     |

## Provider examples

**OpenAI** — `LLM_PROVIDER=openai`, `LLM_API_KEY=sk-...`
**Gemini** — `LLM_PROVIDER=gemini`, `LLM_API_KEY=...` (uses Google's
OpenAI-compatible endpoint). Default model is `gemini-3-pro-preview`, which
requires a **billed** Google AI account. Free-tier keys must use a Flash model
(set `LLM_MODEL=gemini-2.5-flash`).
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
python hn_digest.py
```

Without `LLM_API_KEY` set, it will error on the LLM step; without SMTP it just
prints to the terminal.

## Schedule

The workflow runs daily at **13:00 UTC** and can be triggered manually from the
**Actions** tab (`workflow_dispatch`). Edit the `cron` line in
`.github/workflows/hn-digest.yml` to change the time.
