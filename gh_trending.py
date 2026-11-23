#!/usr/bin/env python3
"""Fetch daily GitHub Trending repositories, enrich with README excerpts,
summarize with an LLM, and send as a dedicated digest email.

Pure standard library — no pip install required.
Reuses LLM, SMTP, and state helpers from hn_digest.py.

Configuration is via environment variables (see README.md):
  LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_STARTTLS
  EMAIL_FROM, EMAIL_TO

  GH_TOP_N          number of trending repos to include (default: 10)
  GH_LANGUAGE       optional language slug, e.g. python (default: all)
  GH_SINCE          daily | weekly | monthly            (default: daily)
  GH_DEDUP_ENABLED  true | false                        (default: true)
  GH_STATE_FILE     rolling 7-day seen repos JSON       (default: state/gh_seen.json)
  GH_FETCH_READMES  true | false                        (default: true)

If SMTP settings are missing, the digest is printed to stdout instead.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from hn_digest import (
    USER_AGENT,
    _collapse_ws,
    _env_bool,
    _env_int,
    call_llm,
    load_seen_ids,
    save_seen_ids,
    send_email,
    strip_html,
)

GH_TRENDING_BASE = "https://github.com/trending"
GH_RAW_BASE = "https://raw.githubusercontent.com"
MAX_README_BYTES = 64 * 1024
MAX_README_CHARS = 1200


# --------------------------------------------------------------------------- #
# GitHub Trending HTML parser (pure stdlib)
# --------------------------------------------------------------------------- #

def _parse_int(text: str) -> int:
    m = re.search(r"([\d,]+)", text or "")
    if not m:
        return 0
    return int(m.group(1).replace(",", ""))


def _extract_bg_color(style: str) -> str:
    m = re.search(r"background-color\s*:\s*(#[0-9a-fA-F]{3,8})", style or "")
    return m.group(1) if m else "#8b949e"


class _GitHubTrendingParser(HTMLParser):
    """Parse <article class="Box-row"> entries from github.com/trending."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.repos: list[dict] = []
        self._in_article = False
        self._article_depth = 0
        self._current: dict | None = None

        self._in_h2 = False
        self._in_p = False
        self._p_parts: list[str] = []

        self._in_lang = False
        self._lang_parts: list[str] = []

        self._link_target: str | None = None  # "stars" | "forks"
        self._link_parts: list[str] = []

        self._in_stars_today = False
        self._stars_today_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        classes = set(attr_map.get("class", "").split())

        if t == "article" and "Box-row" in classes:
            self._in_article = True
            self._article_depth = 1
            self._current = {
                "full_name": "",
                "url": "",
                "description": "",
                "language": "",
                "language_color": "#8b949e",
                "stars": 0,
                "forks": 0,
                "stars_today": 0,
                "readme_excerpt": "",
            }
            return

        if not self._in_article or self._current is None:
            return

        if t == "article":
            self._article_depth += 1
        elif t == "h2":
            self._in_h2 = True
        elif t == "a":
            href = attr_map.get("href", "").strip()
            if self._in_h2 and not self._current["full_name"] and href.startswith("/"):
                path = href.strip("/")
                if path.count("/") == 1:
                    self._current["full_name"] = path
                    self._current["url"] = f"https://github.com/{path}"
            elif href.endswith("/stargazers"):
                self._link_target = "stars"
                self._link_parts = []
            elif href.endswith("/forks"):
                self._link_target = "forks"
                self._link_parts = []
        elif t == "p" and not self._current["description"]:
            self._in_p = True
            self._p_parts = []
        elif t == "span":
            if "repo-language-color" in classes:
                self._current["language_color"] = _extract_bg_color(
                    attr_map.get("style", "")
                )
            elif attr_map.get("itemprop") == "programmingLanguage":
                self._in_lang = True
                self._lang_parts = []
            elif "float-sm-right" in classes:
                self._in_stars_today = True
                self._stars_today_parts = []

    def handle_endtag(self, tag: str) -> None:
        if not self._in_article or self._current is None:
            return
        t = tag.lower()

        if t == "article":
            self._article_depth -= 1
            if self._article_depth <= 0:
                if self._current["full_name"]:
                    self.repos.append(self._current)
                self._current = None
                self._in_article = False
        elif t == "h2":
            self._in_h2 = False
        elif t == "p" and self._in_p:
            self._current["description"] = _collapse_ws("".join(self._p_parts))
            self._in_p = False
        elif t == "a" and self._link_target:
            val = _parse_int("".join(self._link_parts))
            if self._link_target == "stars":
                self._current["stars"] = val
            elif self._link_target == "forks":
                self._current["forks"] = val
            self._link_target = None
        elif t == "span":
            if self._in_lang:
                self._current["language"] = _collapse_ws("".join(self._lang_parts))
                self._in_lang = False
            if self._in_stars_today:
                raw = _collapse_ws("".join(self._stars_today_parts))
                if "star" in raw.lower():
                    self._current["stars_today"] = _parse_int(raw)
                self._in_stars_today = False

    def handle_data(self, data: str) -> None:
        if not self._in_article or self._current is None:
            return
        if self._in_p:
            self._p_parts.append(data)
        if self._in_lang:
            self._lang_parts.append(data)
        if self._link_target:
            self._link_parts.append(data)
        if self._in_stars_today:
            self._stars_today_parts.append(data)


def parse_trending_html(html_content: str) -> list[dict]:
    """Parse GitHub Trending HTML into a list of repo dicts."""
    parser = _GitHubTrendingParser()
    parser.feed(html_content)
    parser.close()
    return parser.repos


def fetch_trending_repos(
    top_n: int,
    language: str = "",
    since: str = "daily",
    seen_ids: set[str] | None = None,
) -> list[dict]:
    """Fetch trending repositories from github.com/trending."""
    since_clean = (since or "daily").strip().lower()
    if since_clean not in {"daily", "weekly", "monthly"}:
        raise SystemExit(
            f"Unknown GH_SINCE '{since_clean}'. Use: daily, weekly, monthly"
        )

    lang_clean = (language or "").strip().lower()
    base = (
        f"{GH_TRENDING_BASE}/{urllib.parse.quote(lang_clean)}"
        if lang_clean
        else GH_TRENDING_BASE
    )
    url = f"{base}?{urllib.parse.urlencode({'since': since_clean})}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html_content = resp.read().decode("utf-8", errors="replace")

    all_repos = parse_trending_html(html_content)
    if seen_ids:
        unseen = [r for r in all_repos if r["full_name"] not in seen_ids]
        if unseen:
            return unseen[:top_n]
        print(
            "All trending repos were already in seen state; falling back to unfiltered list.",
            file=sys.stderr,
        )
    return all_repos[:top_n]


# --------------------------------------------------------------------------- #
# README enrichment & breakout detection
# --------------------------------------------------------------------------- #

def clean_readme_markdown(raw_md: str, max_chars: int = MAX_README_CHARS) -> str:
    """Strip badges, images, code fences, and HTML from a README into clean prose."""
    if not raw_md:
        return ""
    text = raw_md
    # Drop fenced code blocks (often large install blocks or ASCII art).
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Drop markdown images / linked badges: [![alt](img)](url) and ![alt](img).
    text = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # Replace markdown links [label](url) with label.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Strip inline HTML tags (e.g. centered logo banners, badge tables).
    text = strip_html(text)
    # Strip markdown heading markers, emphasis, and horizontal rules.
    text = re.sub(r"[#*`_~>|]+", " ", text)
    text = _collapse_ws(text)
    if len(text) < 40:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def fetch_readme_excerpt(full_name: str, max_chars: int = MAX_README_CHARS) -> str:
    """Fetch a repo's README from raw.githubusercontent.com (fail-soft)."""
    if not full_name or "/" not in full_name:
        return ""
    for filename in ("README.md", "readme.md", "README.rst"):
        url = f"{GH_RAW_BASE}/{full_name}/HEAD/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw_bytes = resp.read(MAX_README_BYTES)
                decoded = raw_bytes.decode("utf-8", errors="replace")
            cleaned = clean_readme_markdown(decoded, max_chars=max_chars)
            if cleaned:
                return cleaned
        except Exception:
            continue
    return ""


def enrich_repos(repos: list[dict], fetch_readmes: bool = True) -> None:
    """Concurrently enrich trending repos with cleaned README excerpts."""
    for r in repos:
        r.setdefault("readme_excerpt", "")
    if not fetch_readmes or not repos:
        return

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_readme_excerpt, r["full_name"]): r for r in repos}
        for fut, repo in futures.items():
            try:
                repo["readme_excerpt"] = fut.result()
            except Exception:
                repo["readme_excerpt"] = ""

    ok = sum(1 for r in repos if r.get("readme_excerpt"))
    print(f"Enriched {ok}/{len(repos)} repos with README excerpts.", file=sys.stderr)


def find_breakout_repo(repos: list[dict]) -> dict | None:
    """Identify the fastest-breaking repo using stars_today^2 / max(stars, 1)."""
    eligible = [r for r in repos if r.get("stars_today", 0) >= 50]
    if eligible:
        return max(
            eligible,
            key=lambda r: (
                (r["stars_today"] ** 2) / max(r.get("stars", 1), 1),
                r["stars_today"],
            ),
        )
    if repos:
        return max(repos, key=lambda r: r.get("stars_today", 0))
    return None


# --------------------------------------------------------------------------- #
# LLM prompt & parsing
# --------------------------------------------------------------------------- #

def build_llm_prompt(
    repos: list[dict],
    breakout: dict | None = None,
    since: str = "daily",
) -> tuple[str, str]:
    breakout_name = breakout.get("full_name") if breakout else None
    period_label = {"daily": "today", "weekly": "this week", "monthly": "this month"}.get(
        since, "today"
    )
    blocks: list[str] = []

    for i, r in enumerate(repos, 1):
        lang_part = f" [{r['language']}]" if r.get("language") else ""
        breakout_tag = (
            " [BREAKOUT REPO]"
            if breakout_name and r.get("full_name") == breakout_name
            else ""
        )
        lines = [
            f"[REPO:{r['full_name']}] #{i}. {r['full_name']}{lang_part} — "
            f"{r.get('stars', 0):,} stars (+{r.get('stars_today', 0):,} {period_label}), "
            f"{r.get('forks', 0):,} forks{breakout_tag}"
        ]
        if r.get("description"):
            lines.append(f"Tagline: {r['description']}")
        if r.get("readme_excerpt"):
            lines.append(f"README excerpt: {r['readme_excerpt']}")
        else:
            lines.append(
                "README excerpt: (unavailable — stick strictly to the tagline; do not invent features)"
            )
        blocks.append("\n".join(lines))

    system = (
        "You are a sharp open-source software analyst writing a concise, high-signal "
        "daily digest of GitHub Trending repositories. Cut through hype: explain "
        "clearly what each project actually does, what problem it solves, and who "
        "should care. Never invent capabilities not present in the tagline or README. "
        "Respond ONLY with valid JSON."
    )

    breakout_instruction = (
        f'  "breakout_repo": {{"full_name": "{breakout_name}", "summary": "1-2 sentences explaining what the flagged breakout repo is and why developers are starring it so fast."}},\n'
        if breakout_name
        else '  "breakout_repo": null,\n'
    )

    user = (
        f"Here are GitHub's trending repositories ({period_label}) with taglines and README excerpts:\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with a JSON object with exactly these keys:\n"
        '  "overview": a 2-3 sentence read on today\'s open-source momentum and dominant themes.\n'
        '  "themes": an array of 3-5 short strings capturing today\'s biggest developer trends.\n'
        + breakout_instruction
        + '  "build_ideas": a single concise paragraph (3-4 sentences) brainstorming 1-2 concrete, original side-project, tool, or startup ideas that combine or build on top of today\'s trending repos (name the specific repos you\'re building on).\n'
        + '  "repos": an array with one object per repo above, each containing:\n'
        '      "full_name": the exact owner/repo string from [REPO:...],\n'
        '      "tldr": 1 crisp sentence explaining what the repo actually builds or does,\n'
        '      "why_notable": 1 short sentence on what makes it interesting or who should check it out.\n'
        "Plain text only inside all strings (no markdown, no bullet symbols)."
    )
    return system, user


def parse_summary(content: str) -> dict:
    """Parse structured JSON response from the LLM; tolerate stray text."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "overview": content,
            "themes": [],
            "breakout_repo": None,
            "build_ideas": "",
            "repos": {},
        }

    overview = str(obj.get("overview", "")).strip()
    themes = [str(t).strip() for t in obj.get("themes", []) if str(t).strip()]
    build_ideas = str(obj.get("build_ideas", "")).strip()

    breakout_repo = None
    raw_breakout = obj.get("breakout_repo")
    if isinstance(raw_breakout, dict):
        b_name = str(raw_breakout.get("full_name", "")).strip()
        b_summary = str(raw_breakout.get("summary", "")).strip()
        if b_name and b_summary:
            breakout_repo = {"full_name": b_name, "summary": b_summary}

    repo_notes: dict[str, dict[str, str]] = {}
    raw_repos = obj.get("repos")
    if isinstance(raw_repos, list):
        for entry in raw_repos:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("full_name", "")).strip()
            if not name:
                continue
            repo_notes[name] = {
                "tldr": str(entry.get("tldr", "")).strip(),
                "why_notable": str(entry.get("why_notable", "")).strip(),
            }
    elif isinstance(raw_repos, dict):
        for name, entry in raw_repos.items():
            if not isinstance(entry, dict):
                continue
            repo_notes[str(name).strip()] = {
                "tldr": str(entry.get("tldr", "")).strip(),
                "why_notable": str(entry.get("why_notable", "")).strip(),
            }

    return {
        "overview": overview,
        "themes": themes,
        "breakout_repo": breakout_repo,
        "build_ideas": build_ideas,
        "repos": repo_notes,
    }


def summarize_with_llm(
    repos: list[dict],
    breakout: dict | None = None,
    since: str = "daily",
) -> dict:
    system, user = build_llm_prompt(repos, breakout=breakout, since=since)
    content = call_llm(system, user)
    return parse_summary(content)


# --------------------------------------------------------------------------- #
# Rendering (GitHub-native styling, distinct from HN orange)
# --------------------------------------------------------------------------- #

def render_html(
    summary: dict,
    repos: list[dict],
    date_str: str,
    language: str = "",
    since: str = "daily",
) -> str:
    repo_notes = summary.get("repos") or {}
    repo_by_name = {r["full_name"]: r for r in repos}
    period_label = {"daily": "today", "weekly": "this week", "monthly": "this month"}.get(
        since, "today"
    )

    breakout_info = summary.get("breakout_repo")
    breakout_name = (
        str(breakout_info.get("full_name", ""))
        if isinstance(breakout_info, dict)
        else ""
    )
    breakout_repo = repo_by_name.get(breakout_name) if breakout_name else None

    breakout_html = ""
    if breakout_repo and breakout_info.get("summary"):
        b_lang = ""
        if breakout_repo.get("language"):
            b_color = escape(breakout_repo.get("language_color") or "#8b949e")
            b_lang = (
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                f'background:{b_color};margin-right:4px;vertical-align:middle;"></span>'
                f'{escape(breakout_repo["language"])} &middot; '
            )
        breakout_html = (
            '<div style="margin:0 20px 18px;padding:14px 16px;background:#f0f6ff;'
            'border-left:3px solid #0969da;border-radius:6px;">'
            '<div style="font-size:12px;font-weight:bold;color:#0969da;'
            'text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">'
            "Breakout Repo"
            "</div>"
            f'<div style="font-size:15px;font-weight:600;margin-bottom:4px;">'
            f'<a href="{escape(breakout_repo["url"])}" style="color:#1f2328;text-decoration:none;">'
            f'{escape(breakout_repo["full_name"])}</a></div>'
            f'<div style="font-size:14px;line-height:1.5;color:#333;">'
            f'{escape(breakout_info["summary"])}</div>'
            f'<div style="font-size:12px;color:#59636e;margin-top:6px;">'
            f'{b_lang}&#9733; {breakout_repo["stars"]:,} '
            f'<span style="color:#1a7f37;font-weight:600;">'
            f'(+{breakout_repo["stars_today"]:,} {period_label})</span>'
            "</div></div>"
        )

    rows = []
    for i, r in enumerate(repos, 1):
        name = r["full_name"]
        notes = repo_notes.get(name) or {}
        tldr = notes.get("tldr") or r.get("description", "")
        why_notable = notes.get("why_notable", "")

        breakout_pill = (
            ' <span style="display:inline-block;font-size:11px;font-weight:600;'
            'color:#0969da;background:#ddf4ff;padding:1px 6px;border-radius:10px;'
            'vertical-align:middle;">breakout</span>'
            if breakout_name and name == breakout_name
            else ""
        )
        tldr_html = (
            f'<div style="font-size:13.5px;line-height:1.45;color:#24292f;margin-top:4px;">'
            f"{escape(tldr)}</div>"
            if tldr
            else ""
        )
        why_html = (
            f'<div style="font-size:13px;line-height:1.45;color:#57606a;margin-top:4px;'
            f'padding-left:8px;border-left:2px solid #d0d7de;">'
            f'<strong>Why notable:</strong> {escape(why_notable)}</div>'
            if why_notable
            else ""
        )

        meta_parts = []
        if r.get("language"):
            color = escape(r.get("language_color") or "#8b949e")
            meta_parts.append(
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                f'background:{color};margin-right:4px;vertical-align:middle;"></span>'
                f'{escape(r["language"])}'
            )
        meta_parts.append(f'&#9733; {r.get("stars", 0):,}')
        if r.get("forks"):
            meta_parts.append(f'Forks {r["forks"]:,}')
        if r.get("stars_today"):
            meta_parts.append(
                f'<span style="color:#1a7f37;font-weight:600;">'
                f'+{r["stars_today"]:,} {period_label}</span>'
            )

        rows.append(
            f"""<tr>
              <td style="padding:10px 10px 10px 16px;color:#6e7781;vertical-align:top;width:24px;">{i}</td>
              <td style="padding:10px 16px 10px 4px;border-bottom:1px solid #eaeef2;">
                <a href="{escape(r['url'])}" style="color:#0969da;text-decoration:none;font-weight:600;">{escape(name)}</a>{breakout_pill}
                {tldr_html}
                {why_html}
                <div style="font-size:12px;color:#57606a;margin-top:6px;">
                  {' &middot; '.join(meta_parts)}
                </div>
              </td>
            </tr>"""
        )

    overview_html = escape(summary.get("overview", "")).replace("\n", "<br>")
    themes = summary.get("themes", [])
    themes_html = ""
    if themes:
        items = "".join(f'<li style="margin:4px 0;">{escape(t)}</li>' for t in themes)
        themes_html = (
            '<div style="padding:0 20px 4px;font-size:13px;font-weight:bold;color:#57606a;'
            'text-transform:uppercase;letter-spacing:.5px;">Key Themes</div>'
            f'<ul style="margin:0 20px 16px;padding-left:20px;font-size:15px;'
            f'line-height:1.5;color:#1f2328;">{items}</ul>'
        )

    build_ideas = (summary.get("build_ideas") or "").strip()
    ideas_html = ""
    if build_ideas:
        ideas_html = (
            '<div style="margin:18px 20px 20px;padding:14px 16px;background:#f6f3ff;'
            'border-left:3px solid #8250df;border-radius:6px;">'
            '<div style="font-size:12px;font-weight:bold;color:#6639ba;'
            'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            "Idea Spark — What You Could Build"
            "</div>"
            f'<div style="font-size:14px;line-height:1.55;color:#1f2328;">'
            f'{escape(build_ideas).replace(chr(10), "<br>")}</div>'
            "</div>"
        )

    subtitle = date_str
    if language:
        subtitle = f"{date_str} &middot; Language: {escape(language)}"

    return f"""<!DOCTYPE html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f8fa;margin:0;padding:20px;">
  <div style="max-width:640px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;border:1px solid #d0d7de;">
    <div style="background:#24292f;color:#fff;padding:16px 20px;font-size:20px;font-weight:bold;">
      GitHub Trending Daily Digest
      <div style="font-size:13px;font-weight:normal;color:#c9d1d9;margin-top:2px;">{subtitle}</div>
    </div>
    <div style="padding:20px;font-size:15px;line-height:1.55;color:#1f2328;">
      {overview_html}
    </div>
    {themes_html}
    {breakout_html}
    <div style="padding:0 20px 8px;font-size:13px;font-weight:bold;color:#57606a;text-transform:uppercase;letter-spacing:.5px;">Trending Repositories</div>
    <table style="width:100%;border-collapse:collapse;font-size:15px;">
      {''.join(rows)}
    </table>
    {ideas_html}
    <div style="padding:16px 20px;font-size:12px;color:#8c959f;border-top:1px solid #eaeef2;">
      Generated automatically &middot; data from GitHub Trending
    </div>
  </div>
</body></html>"""


def render_text(
    summary: dict,
    repos: list[dict],
    date_str: str,
    language: str = "",
    since: str = "daily",
) -> str:
    repo_notes = summary.get("repos") or {}
    repo_by_name = {r["full_name"]: r for r in repos}
    period_label = {"daily": "today", "weekly": "this week", "monthly": "this month"}.get(
        since, "today"
    )

    header = f"GITHUB TRENDING DAILY DIGEST — {date_str}"
    if language:
        header += f" ({language})"
    lines = [header, "", summary.get("overview", "")]

    themes = summary.get("themes", [])
    if themes:
        lines += ["", "KEY THEMES:"] + [f"- {t}" for t in themes]

    breakout_info = summary.get("breakout_repo")
    if isinstance(breakout_info, dict):
        b_repo = repo_by_name.get(str(breakout_info.get("full_name", "")))
        if b_repo and breakout_info.get("summary"):
            lines += [
                "",
                "BREAKOUT REPO:",
                f"  {b_repo['full_name']} ({b_repo['stars']:,} stars, +{b_repo['stars_today']:,} {period_label})",
                f"  {breakout_info['summary']}",
                f"  {b_repo['url']}",
            ]

    lines += ["", "TRENDING REPOSITORIES:"]
    for i, r in enumerate(repos, 1):
        name = r["full_name"]
        notes = repo_notes.get(name) or {}
        lang_suffix = f" [{r['language']}]" if r.get("language") else ""
        lines.append(
            f"{i}. {name}{lang_suffix} ({r.get('stars', 0):,} stars, +{r.get('stars_today', 0):,} {period_label})"
        )
        tldr = notes.get("tldr") or r.get("description", "")
        if tldr:
            lines.append(f"   TL;DR: {tldr}")
        if notes.get("why_notable"):
            lines.append(f"   Why notable: {notes['why_notable']}")
        lines.append(f"   {r['url']}")

    build_ideas = (summary.get("build_ideas") or "").strip()
    if build_ideas:
        lines += ["", "IDEA SPARK — WHAT YOU COULD BUILD:", build_ideas]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    top_n = _env_int("GH_TOP_N", 10)
    language = (os.environ.get("GH_LANGUAGE") or "").strip().lower()
    since = (os.environ.get("GH_SINCE") or "daily").strip().lower()
    dedup_enabled = _env_bool("GH_DEDUP_ENABLED", True)
    state_file = (os.environ.get("GH_STATE_FILE") or "state/gh_seen.json").strip()
    fetch_readmes = _env_bool("GH_FETCH_READMES", True)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%A, %d %B %Y (UTC)")

    seen_map = load_seen_ids(state_file, today=now.date()) if dedup_enabled else {}
    if seen_map:
        print(f"Loaded {len(seen_map)} recently seen GitHub repos.", file=sys.stderr)

    print(
        f"Fetching top {top_n} GitHub Trending repos (since={since}, language={language or 'all'})...",
        file=sys.stderr,
    )
    repos = fetch_trending_repos(
        top_n,
        language=language,
        since=since,
        seen_ids=set(seen_map.keys()) if dedup_enabled else None,
    )
    if not repos:
        raise SystemExit("No trending repositories returned from GitHub.")

    breakout = find_breakout_repo(repos)
    if breakout:
        print(
            f"Breakout repo: {breakout['full_name']} "
            f"(+{breakout['stars_today']:,} / {breakout['stars']:,} total stars)",
            file=sys.stderr,
        )

    enrich_repos(repos, fetch_readmes=fetch_readmes)

    print(f"Summarizing {len(repos)} repos with the LLM...", file=sys.stderr)
    summary = summarize_with_llm(repos, breakout=breakout, since=since)

    html = render_html(summary, repos, date_str, language=language, since=since)
    text = render_text(summary, repos, date_str, language=language, since=since)
    subject = f"GitHub Trending Digest — {now:%Y-%m-%d}"

    if send_email(subject, html, text):
        print("GitHub Trending digest emailed successfully.", file=sys.stderr)
    else:
        print("SMTP not configured — printing digest below.\n", file=sys.stderr)
        print(text)

    if dedup_enabled and state_file:
        save_seen_ids(
            state_file,
            seen_map,
            [r["full_name"] for r in repos if r.get("full_name")],
            today=now.date(),
        )
        print(f"Updated seen state in {state_file}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
