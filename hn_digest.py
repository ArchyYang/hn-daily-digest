#!/usr/bin/env python3
"""Fetch top Hacker News stories, summarize them with an LLM, and email the result.

Pure standard library — no pip install required.

Configuration is via environment variables (see README.md):
  LLM_PROVIDER   openai | gemini | custom        (default: openai)
  LLM_API_KEY    API key for the chosen provider  (required)
  LLM_MODEL      model name                        (provider default if unset)
  LLM_BASE_URL   override base URL (for provider=custom or self-hosted)

  TOP_N          number of stories to include      (default: 15)

  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS       (required to email)
  EMAIL_FROM, EMAIL_TO                              (required to email)
  SMTP_STARTTLS  true | false                       (default: true)

If SMTP settings are missing, the digest is printed to stdout instead.
"""

import json
import os
import smtplib
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

# --------------------------------------------------------------------------- #
# Hacker News
# --------------------------------------------------------------------------- #

HN_FRONTPAGE = "https://hn.algolia.com/api/v1/search"


def fetch_top_stories(top_n: int) -> list[dict]:
    """Return the current front-page stories, sorted by points (desc)."""
    params = urllib.parse.urlencode({"tags": "front_page", "hitsPerPage": top_n})
    url = f"{HN_FRONTPAGE}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "hn-daily-digest"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    stories = []
    for h in data.get("hits", []):
        stories.append(
            {
                "title": h.get("title") or "(no title)",
                "url": h.get("url")
                or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "points": h.get("points") or 0,
                "comments": h.get("num_comments") or 0,
                "author": h.get("author") or "",
                "hn_url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            }
        )
    stories.sort(key=lambda s: s["points"], reverse=True)
    return stories[:top_n]


# --------------------------------------------------------------------------- #
# LLM (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------- #

PROVIDER_PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        # Google's OpenAI-compatible endpoint.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-3-pro-preview",
    },
    "custom": {
        "base_url": None,  # must be supplied via LLM_BASE_URL
        "default_model": None,
    },
}


def summarize_with_llm(stories: list[dict]) -> str:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    if provider not in PROVIDER_PRESETS:
        raise SystemExit(
            f"Unknown LLM_PROVIDER '{provider}'. Use: {', '.join(PROVIDER_PRESETS)}"
        )

    preset = PROVIDER_PRESETS[provider]
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("LLM_API_KEY is required.")

    base_url = os.environ.get("LLM_BASE_URL") or preset["base_url"]
    if not base_url:
        raise SystemExit(
            "LLM_BASE_URL is required when LLM_PROVIDER=custom."
        )
    model = os.environ.get("LLM_MODEL") or preset["default_model"]
    if not model:
        raise SystemExit("LLM_MODEL is required for this provider.")

    story_lines = "\n".join(
        f"{i + 1}. {s['title']} ({s['points']} pts, {s['comments']} comments)"
        for i, s in enumerate(stories)
    )

    system = (
        "You are a sharp tech-news analyst. You write concise, insightful daily "
        "digests of Hacker News. Identify the dominant themes, surface what the "
        "community is excited or worried about, and note any standout story. "
        "Be specific, not generic. Respond ONLY with JSON."
    )
    user = (
        "Here are today's top Hacker News stories:\n\n"
        f"{story_lines}\n\n"
        "Summarize today's themes. Respond with a JSON object with exactly these "
        "keys:\n"
        '  "overview": a 2-3 sentence string on today\'s overall mood/themes.\n'
        '  "trends": an array of 3-5 short strings, each a key trend.\n'
        "Keep the whole thing under 250 words. Plain text only inside the "
        "strings (no markdown)."
    )

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"LLM request failed ({e.code}): {e.read().decode()[:500]}")

    content = data["choices"][0]["message"]["content"].strip()
    return parse_summary(content)


def parse_summary(content: str) -> dict:
    """Parse the LLM response into {overview, trends[]}; tolerate stray text."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        # Fallback: treat the whole thing as the overview with no bullets.
        return {"overview": content, "trends": []}
    overview = str(obj.get("overview", "")).strip()
    trends = [str(t).strip() for t in obj.get("trends", []) if str(t).strip()]
    return {"overview": overview, "trends": trends}


# --------------------------------------------------------------------------- #
# Rendering & email
# --------------------------------------------------------------------------- #

def render_html(summary: dict, stories: list[dict], date_str: str) -> str:
    rows = []
    for i, s in enumerate(stories, 1):
        rows.append(
            f"""<tr>
              <td style="padding:6px 10px;color:#888;">{i}</td>
              <td style="padding:6px 10px;">
                <a href="{escape(s['url'])}" style="color:#1a73e8;text-decoration:none;">{escape(s['title'])}</a>
                <div style="font-size:12px;color:#888;">
                  {s['points']} points &middot; {s['comments']} comments &middot;
                  <a href="{escape(s['hn_url'])}" style="color:#888;">discuss</a>
                </div>
              </td>
            </tr>"""
        )
    overview_html = escape(summary.get("overview", "")).replace("\n", "<br>")
    trends = summary.get("trends", [])
    trends_html = ""
    if trends:
        items = "".join(f"<li style=\"margin:4px 0;\">{escape(t)}</li>" for t in trends)
        trends_html = (
            '<div style="padding:0 20px 4px;font-size:13px;font-weight:bold;color:#555;'
            'text-transform:uppercase;letter-spacing:.5px;">Key Trends</div>'
            f'<ul style="margin:0 20px 16px;padding-left:20px;font-size:15px;'
            f'line-height:1.5;color:#222;">{items}</ul>'
        )
    return f"""<!DOCTYPE html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f6f6;margin:0;padding:20px;">
  <div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;border:1px solid #eee;">
    <div style="background:#ff6600;color:#fff;padding:16px 20px;font-size:20px;font-weight:bold;">
      Hacker News Daily Digest
      <div style="font-size:13px;font-weight:normal;opacity:.9;">{escape(date_str)}</div>
    </div>
    <div style="padding:20px;font-size:15px;line-height:1.55;color:#222;">
      {overview_html}
    </div>
    {trends_html}
    <div style="padding:0 20px 8px;font-size:13px;font-weight:bold;color:#555;text-transform:uppercase;letter-spacing:.5px;">Top Stories</div>
    <table style="width:100%;border-collapse:collapse;font-size:15px;">
      {''.join(rows)}
    </table>
    <div style="padding:16px 20px;font-size:12px;color:#aaa;border-top:1px solid #eee;">
      Generated automatically &middot; data from the Algolia HN API
    </div>
  </div>
</body></html>"""


def render_text(summary: dict, stories: list[dict], date_str: str) -> str:
    lines = [f"HACKER NEWS DAILY DIGEST — {date_str}", "", summary.get("overview", "")]
    trends = summary.get("trends", [])
    if trends:
        lines += ["", "KEY TRENDS:"] + [f"- {t}" for t in trends]
    lines += ["", "TOP STORIES:"]
    for i, s in enumerate(stories, 1):
        lines.append(f"{i}. {s['title']} ({s['points']} pts, {s['comments']} comments)")
        lines.append(f"   {s['url']}")
    return "\n".join(lines)


def send_email(subject: str, html: str, text: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    sender = os.environ.get("EMAIL_FROM")
    recipients = os.environ.get("EMAIL_TO")
    if not (host and sender and recipients):
        return False

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    use_starttls = os.environ.get("SMTP_STARTTLS", "true").lower() != "false"
    to_list = [r.strip() for r in recipients.split(",") if r.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_list)
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            if user:
                server.login(user, password)
            server.sendmail(sender, to_list, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if use_starttls:
                server.starttls(context=context)
            if user:
                server.login(user, password)
            server.sendmail(sender, to_list, msg.as_string())
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    top_n = int(os.environ.get("TOP_N", "15"))
    date_str = datetime.now(timezone.utc).strftime("%A, %d %B %Y (UTC)")

    print(f"Fetching top {top_n} Hacker News stories...", file=sys.stderr)
    stories = fetch_top_stories(top_n)
    if not stories:
        raise SystemExit("No stories returned from Hacker News.")

    print(f"Summarizing {len(stories)} stories with the LLM...", file=sys.stderr)
    summary = summarize_with_llm(stories)

    html = render_html(summary, stories, date_str)
    text = render_text(summary, stories, date_str)
    subject = f"HN Daily Digest — {datetime.now(timezone.utc):%Y-%m-%d}"

    if send_email(subject, html, text):
        print("Digest emailed successfully.", file=sys.stderr)
    else:
        print("SMTP not configured — printing digest below.\n", file=sys.stderr)
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
