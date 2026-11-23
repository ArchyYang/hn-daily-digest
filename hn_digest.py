#!/usr/bin/env python3
"""Fetch top Hacker News stories, enrich them with article text and top comments,
summarize them with an LLM, and email the result.

Pure standard library — no pip install required.

Configuration is via environment variables (see README.md):
  LLM_PROVIDER    openai | gemini | custom        (default: openai)
  LLM_API_KEY     API key for the chosen provider  (required)
  LLM_MODEL       model name                        (provider default if unset)
  LLM_BASE_URL    override base URL (for provider=custom or self-hosted)

  TOP_N           number of stories to include      (default: 15)
  FETCH_MODE      24h | front_page                  (default: 24h)
  DEDUP_ENABLED   true | false                      (default: true)
  STATE_FILE      path to rolling seen-stories JSON (default: state/seen.json)
  FETCH_ARTICLES  true | false                      (default: true)
  COMMENT_TOP_K   stories to fetch comments for     (default: 8)

  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS        (required to email)
  EMAIL_FROM, EMAIL_TO                               (required to email)
  SMTP_STARTTLS   true | false                       (default: true)

If SMTP settings are missing, the digest is printed to stdout instead.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape, unescape
from html.parser import HTMLParser
import json
import os
import re
import smtplib
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #

USER_AGENT = "hn-daily-digest/2.0 (+https://github.com/ArchyYang/hn-daily-digest)"

HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"
HN_ITEM_API = "https://hn.algolia.com/api/v1/items"

SEEN_TTL_DAYS = 7
MAX_ARTICLE_BYTES = 256 * 1024
MAX_ARTICLE_CHARS = 1200
MAX_COMMENT_CHARS = 350
MAX_REPLY_CHARS = 250

BINARY_EXTENSIONS = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".mov",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".dmg",
    ".exe",
    ".whl",
)


def _env_bool(name: str, default: bool) -> bool:
    val = (os.environ.get(name) or "").strip().lower()
    if not val:
        return default
    return val not in {"false", "0", "no", "off"}


def _env_int(name: str, default: int) -> int:
    val = (os.environ.get(name) or "").strip()
    return int(val) if val else default


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# HTML stripping & article extraction (pure stdlib)
# --------------------------------------------------------------------------- #

class _HTMLStripper(HTMLParser):
    """Strip HTML tags into clean plain text (used for HN comments & story text)."""

    BLOCK_TAGS = {"p", "br", "div", "li", "blockquote", "pre", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return _collapse_ws(unescape("".join(self._parts)))


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    parser = _HTMLStripper()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        return _collapse_ws(unescape(re.sub(r"<[^>]+>", " ", raw)))
    return parser.get_text()


class ArticleTextExtractor(HTMLParser):
    """Extract readable body text from an HTML page, preferring <article>/<main>."""

    SKIP_TAGS = {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "svg",
        "form",
        "iframe",
        "button",
        "select",
        "option",
    }
    BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "br", "div"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._main_depth = 0
        self._main_parts: list[str] = []
        self._all_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if t in {"article", "main"}:
            self._main_depth += 1
        if self._skip_depth == 0 and t in self.BLOCK_TAGS:
            self._all_parts.append(" ")
            if self._main_depth > 0:
                self._main_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self.SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if t in {"article", "main"} and self._main_depth > 0:
            self._main_depth -= 1
        if self._skip_depth == 0 and t in self.BLOCK_TAGS:
            self._all_parts.append(" ")
            if self._main_depth > 0:
                self._main_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._all_parts.append(data)
        if self._main_depth > 0:
            self._main_parts.append(data)

    def get_text(self) -> str:
        main_text = _collapse_ws("".join(self._main_parts))
        if len(main_text) >= 200:
            return main_text
        return _collapse_ws("".join(self._all_parts))


def extract_article_text(html_content: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Extract readable text from raw HTML and truncate to max_chars."""
    parser = ArticleTextExtractor()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception:
        return ""
    text = parser.get_text()
    if len(text) < 80:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def fetch_article_text(url: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    """Fetch an external URL and return a short readable excerpt (fail-soft)."""
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    host = parsed.netloc.lower().split(":", 1)[0].removeprefix("www.")
    if not host or host == "news.ycombinator.com":
        return ""
    if parsed.path.lower().endswith(BINARY_EXTENSIONS):
        return ""

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype and "text/plain" not in ctype:
                return ""
            raw_bytes = resp.read(MAX_ARTICLE_BYTES)
            charset = resp.headers.get_content_charset() or "utf-8"
            decoded = raw_bytes.decode(charset, errors="replace")
    except Exception:
        return ""

    if "text/plain" in ctype and "text/html" not in ctype:
        text = _collapse_ws(decoded)
        if len(text) < 80:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        return text

    return extract_article_text(decoded, max_chars=max_chars)


# --------------------------------------------------------------------------- #
# Deduplication state (rolling 7-day seen story IDs)
# --------------------------------------------------------------------------- #

def _prune_seen(seen: dict[str, str], today: date) -> dict[str, str]:
    cutoff = today - timedelta(days=SEEN_TTL_DAYS)
    pruned: dict[str, str] = {}
    for story_id, date_str in seen.items():
        try:
            d = date.fromisoformat(str(date_str))
        except ValueError:
            continue
        if d >= cutoff:
            pruned[str(story_id)] = d.isoformat()
    return pruned


def load_seen_ids(path: str, today: date | None = None) -> dict[str, str]:
    """Load rolling seen story IDs from JSON file; fail-soft if missing/corrupt."""
    if not path or not os.path.exists(path):
        return {}
    today = today or datetime.now(timezone.utc).date()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return _prune_seen(data, today)
    except Exception as e:
        print(f"Warning: could not read state file '{path}': {e}", file=sys.stderr)
    return {}


def save_seen_ids(
    path: str,
    seen: dict[str, str],
    new_ids: list[str],
    today: date | None = None,
) -> None:
    """Merge new story IDs, prune old entries, and persist to JSON file."""
    if not path:
        return
    today = today or datetime.now(timezone.utc).date()
    merged = _prune_seen(seen, today)
    today_iso = today.isoformat()
    for sid in new_ids:
        if sid:
            merged[str(sid)] = today_iso
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")


# --------------------------------------------------------------------------- #
# Hacker News fetching, comments & debate detection
# --------------------------------------------------------------------------- #

def extract_domain(url: str | None) -> str:
    """Extract a clean hostname (without 'www.') from an external story URL."""
    if not url:
        return ""
    host = urllib.parse.urlparse(url).netloc.lower().split(":", 1)[0]
    host = host.removeprefix("www.")
    if host == "news.ycombinator.com":
        return ""
    return host


def fetch_top_stories(
    top_n: int,
    fetch_mode: str = "24h",
    seen_ids: set[str] | None = None,
    now_ts: int | None = None,
) -> list[dict]:
    """Return top HN stories sorted by points (desc).

    fetch_mode:
      - '24h': stories created within the last 24 hours (default)
      - 'front_page': current Algolia front_page snapshot
    """
    mode = (fetch_mode or "24h").strip().lower()
    if mode not in {"24h", "front_page"}:
        raise SystemExit(
            f"Unknown FETCH_MODE '{mode}'. Use: 24h, front_page"
        )

    if mode == "front_page":
        hits_per_page = min(100, max(30, top_n + len(seen_ids or ())))
        params = urllib.parse.urlencode(
            {"tags": "front_page", "hitsPerPage": hits_per_page}
        )
    else:
        cutoff = (now_ts if now_ts is not None else int(time.time())) - 86400
        hits_per_page = min(100, max(60, top_n * 3))
        params = urllib.parse.urlencode(
            {
                "tags": "story",
                "numericFilters": f"created_at_i>{cutoff}",
                "hitsPerPage": hits_per_page,
            }
        )

    url = f"{HN_SEARCH_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    all_stories: list[dict] = []
    for h in data.get("hits", []):
        object_id = str(h.get("objectID") or "")
        raw_url = h.get("url")
        hn_url = f"https://news.ycombinator.com/item?id={object_id}"
        points = h.get("points") or 0
        comments = h.get("num_comments") or 0
        all_stories.append(
            {
                "object_id": object_id,
                "title": h.get("title") or "(no title)",
                "url": raw_url or hn_url,
                "domain": extract_domain(raw_url),
                "points": points,
                "comments": comments,
                "debate_ratio": round(comments / max(points, 1), 2),
                "author": h.get("author") or "",
                "hn_url": hn_url,
                "story_text": strip_html(h.get("story_text")),
                "article_excerpt": "",
                "top_comments": [],
            }
        )

    all_stories.sort(key=lambda s: s["points"], reverse=True)
    if seen_ids:
        unseen = [s for s in all_stories if s["object_id"] not in seen_ids]
        if unseen:
            return unseen[:top_n]
        print(
            "All fetched stories were already in seen state; falling back to unfiltered list.",
            file=sys.stderr,
        )
    return all_stories[:top_n]


def find_spiciest_debate(stories: list[dict]) -> dict | None:
    """Identify the most contentious/debated story by comments-to-points ratio."""
    eligible = [s for s in stories if s["points"] >= 50 and s["comments"] >= 30]
    if eligible:
        return max(
            eligible,
            key=lambda s: (s["comments"] / max(s["points"], 1), s["comments"]),
        )
    fallback = [s for s in stories if s["comments"] >= 15]
    if fallback:
        return max(fallback, key=lambda s: s["comments"])
    return None


def extract_comments_from_item(
    item: dict,
    max_top: int = 6,
    max_replies: int = 2,
    max_chars: int = MAX_COMMENT_CHARS,
) -> list[str]:
    """Extract clean top-level comments (and select replies) from an Algolia item."""
    comments: list[str] = []
    skip_markers = {"[dead]", "[flagged]", "[deleted]"}

    for child in item.get("children") or []:
        if len(comments) >= max_top:
            break
        raw_text = child.get("text")
        if not raw_text:
            continue
        clean = strip_html(raw_text)
        if not clean or clean.lower() in skip_markers or len(clean) < 25:
            continue
        if len(clean) > max_chars:
            clean = clean[:max_chars].rsplit(" ", 1)[0] + "..."
        author = child.get("author") or "anon"
        entry = f"{author}: {clean}"

        # Include up to max_replies direct replies on the first two top comments
        # to capture pushback or counterarguments.
        if len(comments) < 2 and max_replies > 0:
            replies: list[str] = []
            for rep in child.get("children") or []:
                if len(replies) >= max_replies:
                    break
                rep_clean = strip_html(rep.get("text"))
                if (
                    not rep_clean
                    or rep_clean.lower() in skip_markers
                    or len(rep_clean) < 25
                ):
                    continue
                if len(rep_clean) > MAX_REPLY_CHARS:
                    rep_clean = rep_clean[:MAX_REPLY_CHARS].rsplit(" ", 1)[0] + "..."
                rep_author = rep.get("author") or "anon"
                replies.append(f"  -> {rep_author}: {rep_clean}")
            if replies:
                entry = entry + "\n" + "\n".join(replies)

        comments.append(entry)
    return comments


def fetch_story_comments(object_id: str) -> list[str]:
    """Fetch top comments for a story from the Algolia item API (fail-soft)."""
    if not object_id:
        return []
    url = f"{HN_ITEM_API}/{urllib.parse.quote(object_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            item = json.load(resp)
    except Exception:
        return []
    return extract_comments_from_item(item)


def enrich_stories(
    stories: list[dict],
    spiciest: dict | None = None,
    fetch_articles: bool = True,
    comment_top_k: int = 8,
) -> None:
    """Concurrently enrich stories with article excerpts and top HN comments."""
    comment_ids = {s["object_id"] for s in stories[: max(0, comment_top_k)]}
    if spiciest and spiciest.get("object_id") and comment_top_k > 0:
        comment_ids.add(spiciest["object_id"])

    article_futures = {}
    comment_futures = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for s in stories:
            s.setdefault("article_excerpt", "")
            s.setdefault("top_comments", [])
            if s.get("story_text"):
                text = s["story_text"]
                if len(text) > MAX_ARTICLE_CHARS:
                    text = text[:MAX_ARTICLE_CHARS].rsplit(" ", 1)[0] + "..."
                s["article_excerpt"] = text
            elif fetch_articles:
                article_futures[pool.submit(fetch_article_text, s["url"])] = s

            if s["object_id"] in comment_ids:
                comment_futures[pool.submit(fetch_story_comments, s["object_id"])] = s

        for fut, story in article_futures.items():
            try:
                story["article_excerpt"] = fut.result()
            except Exception:
                story["article_excerpt"] = ""

        for fut, story in comment_futures.items():
            try:
                story["top_comments"] = fut.result()
            except Exception:
                story["top_comments"] = []

    articles_ok = sum(1 for s in stories if s.get("article_excerpt"))
    comments_ok = sum(1 for s in stories if s.get("top_comments"))
    print(
        f"Enriched {articles_ok}/{len(stories)} articles and "
        f"{comments_ok}/{len(comment_ids)} comment threads.",
        file=sys.stderr,
    )


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


def build_llm_prompt(stories: list[dict], spiciest: dict | None = None) -> tuple[str, str]:
    """Construct the (system, user) prompts with enriched article and comment context."""
    spiciest_id = spiciest.get("object_id") if spiciest else None
    blocks: list[str] = []

    for i, s in enumerate(stories, 1):
        domain_part = f" ({s['domain']})" if s.get("domain") else ""
        ratio = s.get("debate_ratio", round(s["comments"] / max(s["points"], 1), 2))
        spicy_tag = (
            f" [SPICIEST DEBATE: {ratio} comments/pt]"
            if spiciest_id and s.get("object_id") == spiciest_id
            else ""
        )
        lines = [
            f"[ID:{s['object_id']}] #{i}. {s['title']}{domain_part} "
            f"— {s['points']} pts, {s['comments']} comments{spicy_tag}"
        ]
        if s.get("article_excerpt"):
            lines.append(f"Article excerpt: {s['article_excerpt']}")
        else:
            lines.append(
                "Article excerpt: (unavailable — stick strictly to title/comments; do not invent facts)"
            )
        if s.get("top_comments"):
            lines.append("Top HN comments:")
            for c in s["top_comments"]:
                lines.append(f"  - {c}")
        blocks.append("\n".join(lines))

    system = (
        "You are a sharp tech-news analyst writing a concise, high-signal daily "
        "digest of Hacker News. Go beyond headlines: synthesize what the linked "
        "articles actually say and what the HN comment section actually thinks "
        "(consensus, technical nuances, or sharp skepticism). Never invent article "
        "details when an excerpt is unavailable. Respond ONLY with valid JSON."
    )

    spiciest_instruction = (
        f'  "spiciest_debate": {{"story_id": "{spiciest_id}", "summary": "1-2 sentences capturing both sides of the debate in the flagged spiciest thread."}},\n'
        if spiciest_id
        else '  "spiciest_debate": null,\n'
    )

    user = (
        "Here are today's top Hacker News stories with article excerpts and top comments:\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with a JSON object with exactly these keys:\n"
        '  "overview": a 2-3 sentence string on today\'s overall mood and dominant themes.\n'
        '  "trends": an array of 3-5 short strings, each a specific trend from today.\n'
        + spiciest_instruction
        + '  "stories": an array with one object per story above, each containing:\n'
        '      "story_id": the exact ID string from [ID:...],\n'
        '      "tldr": 1 crisp sentence explaining what the link actually is or argues,\n'
        '      "hn_take": 1 crisp sentence summarizing comment consensus or top pushback ("" if no comments were provided).\n'
        "Plain text only inside all strings (no markdown, no bullet symbols)."
    )
    return system, user


def summarize_with_llm(stories: list[dict], spiciest: dict | None = None) -> dict:
    provider = (os.environ.get("LLM_PROVIDER") or "openai").lower()
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

    system, user = build_llm_prompt(stories, spiciest=spiciest)

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

    # Retry transient errors (rate limit / server overload) with backoff.
    transient = {429, 500, 502, 503, 504}
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            if e.code in transient and attempt < max_attempts:
                wait = min(2 ** attempt, 30)
                print(
                    f"LLM {e.code} (attempt {attempt}/{max_attempts}); "
                    f"retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise SystemExit(f"LLM request failed ({e.code}): {body}")
        except urllib.error.URLError as e:
            if attempt < max_attempts:
                wait = min(2 ** attempt, 30)
                print(
                    f"LLM connection error ({e.reason}); retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise SystemExit(f"LLM request failed: {e.reason}")

    content = data["choices"][0]["message"]["content"].strip()
    return parse_summary(content)


def parse_summary(content: str) -> dict:
    """Parse the LLM response into structured summary fields; tolerate stray text."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "overview": content,
            "trends": [],
            "spiciest_debate": None,
            "stories": {},
        }

    overview = str(obj.get("overview", "")).strip()
    trends = [str(t).strip() for t in obj.get("trends", []) if str(t).strip()]

    spiciest_debate = None
    raw_debate = obj.get("spiciest_debate")
    if isinstance(raw_debate, dict):
        debate_id = str(raw_debate.get("story_id", "")).strip()
        debate_summary = str(raw_debate.get("summary", "")).strip()
        if debate_id and debate_summary:
            spiciest_debate = {"story_id": debate_id, "summary": debate_summary}

    story_notes: dict[str, dict[str, str]] = {}
    raw_stories = obj.get("stories")
    if isinstance(raw_stories, list):
        for entry in raw_stories:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("story_id", "")).strip()
            if not sid:
                continue
            story_notes[sid] = {
                "tldr": str(entry.get("tldr", "")).strip(),
                "hn_take": str(entry.get("hn_take", "")).strip(),
            }
    elif isinstance(raw_stories, dict):
        for sid, entry in raw_stories.items():
            if not isinstance(entry, dict):
                continue
            story_notes[str(sid).strip()] = {
                "tldr": str(entry.get("tldr", "")).strip(),
                "hn_take": str(entry.get("hn_take", "")).strip(),
            }

    return {
        "overview": overview,
        "trends": trends,
        "spiciest_debate": spiciest_debate,
        "stories": story_notes,
    }


# --------------------------------------------------------------------------- #
# Rendering & email
# --------------------------------------------------------------------------- #

def render_html(summary: dict, stories: list[dict], date_str: str) -> str:
    story_notes = summary.get("stories") or {}
    story_by_id = {str(s.get("object_id", "")): s for s in stories}

    debate_info = summary.get("spiciest_debate")
    debate_story_id = (
        str(debate_info.get("story_id", "")) if isinstance(debate_info, dict) else ""
    )
    debate_story = story_by_id.get(debate_story_id) if debate_story_id else None

    debate_html = ""
    if debate_story and debate_info.get("summary"):
        debate_html = (
            '<div style="margin:0 20px 18px;padding:14px 16px;background:#fff8f3;'
            'border-left:3px solid #ff6600;border-radius:6px;">'
            '<div style="font-size:12px;font-weight:bold;color:#cc5200;'
            'text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;">'
            "Spiciest Debate"
            "</div>"
            f'<div style="font-size:15px;font-weight:600;margin-bottom:4px;">'
            f'<a href="{escape(debate_story["hn_url"])}" style="color:#222;text-decoration:none;">'
            f'{escape(debate_story["title"])}</a></div>'
            f'<div style="font-size:14px;line-height:1.5;color:#333;">'
            f'{escape(debate_info["summary"])}</div>'
            f'<div style="font-size:12px;color:#888;margin-top:6px;">'
            f'{debate_story["points"]} points &middot; {debate_story["comments"]} comments &middot; '
            f'<a href="{escape(debate_story["hn_url"])}" style="color:#cc5200;">join thread</a>'
            "</div></div>"
        )

    rows = []
    for i, s in enumerate(stories, 1):
        sid = str(s.get("object_id", ""))
        notes = story_notes.get(sid) or {}
        tldr = notes.get("tldr", "")
        hn_take = notes.get("hn_take", "")

        domain_html = (
            f' <span style="font-size:12px;color:#888;">({escape(s["domain"])})</span>'
            if s.get("domain")
            else ""
        )
        heated_badge = (
            ' <span style="display:inline-block;font-size:11px;font-weight:600;'
            'color:#cc5200;background:#ffefe5;padding:1px 6px;border-radius:10px;'
            'vertical-align:middle;">heated</span>'
            if debate_story_id and sid == debate_story_id
            else ""
        )
        tldr_html = (
            f'<div style="font-size:13.5px;line-height:1.45;color:#333;margin-top:4px;">'
            f"{escape(tldr)}</div>"
            if tldr
            else ""
        )
        take_html = (
            f'<div style="font-size:13px;line-height:1.45;color:#555;margin-top:4px;'
            f'padding-left:8px;border-left:2px solid #e0e0e0;">'
            f'<strong>HN take:</strong> {escape(hn_take)}</div>'
            if hn_take
            else ""
        )
        rows.append(
            f"""<tr>
              <td style="padding:10px 10px 10px 16px;color:#888;vertical-align:top;width:24px;">{i}</td>
              <td style="padding:10px 16px 10px 4px;border-bottom:1px solid #f2f2f2;">
                <a href="{escape(s['url'])}" style="color:#1a73e8;text-decoration:none;font-weight:500;">{escape(s['title'])}</a>{domain_html}{heated_badge}
                {tldr_html}
                {take_html}
                <div style="font-size:12px;color:#888;margin-top:5px;">
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
        items = "".join(f'<li style="margin:4px 0;">{escape(t)}</li>' for t in trends)
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
    {debate_html}
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
    story_notes = summary.get("stories") or {}
    story_by_id = {str(s.get("object_id", "")): s for s in stories}

    lines = [f"HACKER NEWS DAILY DIGEST — {date_str}", "", summary.get("overview", "")]
    trends = summary.get("trends", [])
    if trends:
        lines += ["", "KEY TRENDS:"] + [f"- {t}" for t in trends]

    debate_info = summary.get("spiciest_debate")
    if isinstance(debate_info, dict):
        debate_story = story_by_id.get(str(debate_info.get("story_id", "")))
        if debate_story and debate_info.get("summary"):
            lines += [
                "",
                "SPICIEST DEBATE:",
                f"  {debate_story['title']} ({debate_story['points']} pts, {debate_story['comments']} comments)",
                f"  {debate_info['summary']}",
                f"  {debate_story['hn_url']}",
            ]

    lines += ["", "TOP STORIES:"]
    for i, s in enumerate(stories, 1):
        sid = str(s.get("object_id", ""))
        notes = story_notes.get(sid) or {}
        domain_suffix = f" [{s['domain']}]" if s.get("domain") else ""
        lines.append(
            f"{i}. {s['title']}{domain_suffix} ({s['points']} pts, {s['comments']} comments)"
        )
        if notes.get("tldr"):
            lines.append(f"   TL;DR: {notes['tldr']}")
        if notes.get("hn_take"):
            lines.append(f"   HN take: {notes['hn_take']}")
        lines.append(f"   {s['url']}")
        lines.append(f"   Discuss: {s['hn_url']}")
    return "\n".join(lines)


def send_email(subject: str, html: str, text: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    sender = os.environ.get("EMAIL_FROM")
    recipients = os.environ.get("EMAIL_TO")
    if not (host and sender and recipients):
        return False

    port = _env_int("SMTP_PORT", 587)
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    use_starttls = _env_bool("SMTP_STARTTLS", True)
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
    top_n = _env_int("TOP_N", 15)
    fetch_mode = (os.environ.get("FETCH_MODE") or "24h").strip().lower()
    dedup_enabled = _env_bool("DEDUP_ENABLED", True)
    state_file = (os.environ.get("STATE_FILE") or "state/seen.json").strip()
    fetch_articles = _env_bool("FETCH_ARTICLES", True)
    comment_top_k = _env_int("COMMENT_TOP_K", 8)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%A, %d %B %Y (UTC)")

    seen_map = load_seen_ids(state_file, today=now.date()) if dedup_enabled else {}
    if seen_map:
        print(f"Loaded {len(seen_map)} recently seen story IDs.", file=sys.stderr)

    print(
        f"Fetching top {top_n} Hacker News stories (mode={fetch_mode})...",
        file=sys.stderr,
    )
    stories = fetch_top_stories(
        top_n,
        fetch_mode=fetch_mode,
        seen_ids=set(seen_map.keys()) if dedup_enabled else None,
    )
    if not stories:
        raise SystemExit("No stories returned from Hacker News.")

    spiciest = find_spiciest_debate(stories)
    if spiciest:
        print(
            f"Spiciest debate: '{spiciest['title']}' "
            f"({spiciest['comments']} comments / {spiciest['points']} pts)",
            file=sys.stderr,
        )

    enrich_stories(
        stories,
        spiciest=spiciest,
        fetch_articles=fetch_articles,
        comment_top_k=comment_top_k,
    )

    print(f"Summarizing {len(stories)} stories with the LLM...", file=sys.stderr)
    summary = summarize_with_llm(stories, spiciest=spiciest)

    html = render_html(summary, stories, date_str)
    text = render_text(summary, stories, date_str)
    subject = f"HN Daily Digest — {now:%Y-%m-%d}"

    emailed = send_email(subject, html, text)
    if emailed:
        print("Digest emailed successfully.", file=sys.stderr)
    else:
        print("SMTP not configured — printing digest below.\n", file=sys.stderr)
        print(text)

    if dedup_enabled and state_file:
        save_seen_ids(
            state_file,
            seen_map,
            [s["object_id"] for s in stories if s.get("object_id")],
            today=now.date(),
        )
        print(f"Updated seen state in {state_file}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
