#!/usr/bin/env python3
"""Weekly Compounding Signal & Synthesis Pipeline.

Combines:
  1. Accumulated daily HN archives from the `data` branch (with automatic 7-day
     Algolia bootstrap when fewer than 3 daily archives exist yet), plus prior
     weekly syntheses for longitudinal "persisted vs. faded" tracking.
  2. Lobste.rs hottest discussions (free JSON API) + HN x Lobste.rs overlap.
  3. High-reaction open GitHub issues from bellwether repos in `watchlist.json`.

Synthesizes with Gemini / OpenAI into an editorial weekly email (and optional
Notion page) containing:
  - The Week in One Take
  - Persisted vs. Faded themes
  - 2-3 "Connect-the-Dots" Article Pitches (thesis, evidence, counterargument)
  - 3-4 Validated Pain Points & focused build angles
  - Cross-Platform Signals

Pure standard library — no pip install required.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from html import escape
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from hn_digest import (
    HN_SEARCH_API,
    USER_AGENT,
    _collapse_ws,
    _env_int,
    call_llm,
    extract_domain,
    fetch_ask_show_stories,
    fetch_story_comments,
    send_email,
    strip_html,
)

LOBSTERS_HOTTEST = "https://lobste.rs/hottest.json"
GH_ISSUE_SEARCH = "https://api.github.com/search/issues"
NOTION_PAGES_API = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"

DEFAULT_WATCHLIST = [
    "anthropics/claude-code",
    "ollama/ollama",
    "vllm-project/vllm",
    "modelcontextprotocol/servers",
    "zed-industries/zed",
    "duckdb/duckdb",
    "astral-sh/uv",
    "oven-sh/bun",
    "vercel/ai",
    "supabase/supabase",
]


# --------------------------------------------------------------------------- #
# Archive loading & saving (orphan `data` branch workspace)
# --------------------------------------------------------------------------- #

def load_daily_archives(
    archive_dir: str,
    days: int = 7,
    today: date | None = None,
) -> list[dict]:
    """Load daily HN JSON archives from the last `days` days."""
    if not archive_dir:
        return []
    hn_dir = os.path.join(archive_dir, "hn")
    if not os.path.isdir(hn_dir):
        return []
    today = today or datetime.now(timezone.utc).date()
    cutoff = (today - timedelta(days=days)).isoformat()
    today_iso = today.isoformat()

    loaded: list[dict] = []
    for fn in sorted(os.listdir(hn_dir)):
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        if cutoff <= stem <= today_iso:
            try:
                with open(os.path.join(hn_dir, fn), encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    loaded.append(obj)
            except Exception as e:
                print(f"Warning: could not read archive '{fn}': {e}", file=sys.stderr)
    return loaded


def load_prior_weekly_archives(
    archive_dir: str,
    limit: int = 3,
    before_week: str | None = None,
) -> list[dict]:
    """Load up to `limit` most recent prior weekly summaries for longitudinal memory."""
    if not archive_dir:
        return []
    w_dir = os.path.join(archive_dir, "weekly")
    if not os.path.isdir(w_dir):
        return []
    candidates = sorted(
        (
            fn
            for fn in os.listdir(w_dir)
            if fn.endswith(".json") and (not before_week or fn[:-5] < before_week)
        ),
        reverse=True,
    )
    loaded: list[dict] = []
    for fn in candidates[:limit]:
        try:
            with open(os.path.join(w_dir, fn), encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                loaded.append(obj)
        except Exception:
            continue
    return list(reversed(loaded))


def save_weekly_archive(
    archive_dir: str,
    week_id: str,
    summary: dict,
    meta: dict,
) -> str:
    """Persist weekly synthesis JSON to `{archive_dir}/weekly/{week_id}.json`."""
    if not archive_dir:
        return ""
    w_dir = os.path.join(archive_dir, "weekly")
    os.makedirs(w_dir, exist_ok=True)
    out_path = os.path.join(w_dir, f"{week_id}.json")
    payload = {
        "week": week_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        **summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out_path


# --------------------------------------------------------------------------- #
# Day-1 Bootstrap (7-day Algolia HN fallback when archive has <3 days)
# --------------------------------------------------------------------------- #

def bootstrap_hn_week(
    top_n: int = 15,
    ask_show_n: int = 6,
    now_ts: int | None = None,
) -> dict:
    """Fetch past 7 days of HN top stories + Ask/Show HN + comments live."""
    now_s = now_ts if now_ts is not None else int(time.time())
    cutoff = now_s - 7 * 86400
    params = urllib.parse.urlencode(
        {
            "tags": "story",
            "numericFilters": f"created_at_i>{cutoff},points>=80",
            "hitsPerPage": max(40, top_n * 2),
        }
    )
    url = f"{HN_SEARCH_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    stories: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for h in data.get("hits", []):
            oid = str(h.get("objectID") or "")
            raw_url = h.get("url")
            hn_url = f"https://news.ycombinator.com/item?id={oid}"
            stories.append(
                {
                    "object_id": oid,
                    "title": h.get("title") or "(no title)",
                    "domain": extract_domain(raw_url),
                    "url": raw_url or hn_url,
                    "hn_url": hn_url,
                    "points": h.get("points") or 0,
                    "comments": h.get("num_comments") or 0,
                    "top_comments": [],
                }
            )
    except Exception as e:
        print(f"Warning: HN 7-day bootstrap failed: {e}", file=sys.stderr)

    stories.sort(key=lambda s: s["points"], reverse=True)
    stories = stories[:top_n]
    story_ids = {s["object_id"] for s in stories}

    ask_show = fetch_ask_show_stories(
        limit=ask_show_n,
        exclude_ids=story_ids,
        lookback_seconds=7 * 86400,
        now_ts=now_s,
    )

    # Fetch top comments concurrently for top 8 stories + all Ask/Show threads.
    comment_targets = stories[:8] + ask_show
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_story_comments, s["object_id"]): s for s in comment_targets if s.get("object_id")}
        for fut, item in futs.items():
            try:
                item["top_comments"] = fut.result()[:3]
            except Exception:
                item["top_comments"] = []

    return {"stories": stories, "ask_show": ask_show}


# --------------------------------------------------------------------------- #
# Lobste.rs & Bellwether GitHub Issues
# --------------------------------------------------------------------------- #

def fetch_lobsters_hottest(limit: int = 15) -> list[dict]:
    """Fetch top discussions from Lobste.rs hottest JSON feed (fail-soft)."""
    req = urllib.request.Request(
        LOBSTERS_HOTTEST,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            items = json.load(resp)
    except Exception as e:
        print(f"Warning: Lobste.rs fetch failed: {e}", file=sys.stderr)
        return []

    results: list[dict] = []
    for it in items if isinstance(items, list) else []:
        url = (it.get("url") or it.get("comments_url") or "").strip()
        results.append(
            {
                "title": (it.get("title") or "").strip(),
                "url": url,
                "comments_url": (it.get("comments_url") or url).strip(),
                "score": int(it.get("score") or 0),
                "comments": int(it.get("comment_count") or 0),
                "tags": [str(t) for t in (it.get("tags") or [])],
            }
        )
    results.sort(key=lambda x: (x["comments"] + x["score"]), reverse=True)
    return results[:limit]


def load_watchlist(path: str = "watchlist.json") -> list[str]:
    """Load bellwether owner/repo list from watchlist.json (with safe defaults)."""
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            repos = data.get("repos") if isinstance(data, dict) else data
            if isinstance(repos, list):
                cleaned = [str(r).strip() for r in repos if "/" in str(r)]
                if cleaned:
                    return cleaned
        except Exception as e:
            print(f"Warning: could not read watchlist '{path}': {e}", file=sys.stderr)
    return list(DEFAULT_WATCHLIST)


def _clean_issue_body(body: str | None, max_chars: int = 300) -> str:
    if not body:
        return ""
    text = re.sub(r"```[\s\S]*?```", " ", body)
    text = strip_html(text)
    text = re.sub(r"[#*`_~>|]+", " ", text)
    text = _collapse_ws(text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def fetch_bellwether_issues(
    repos: list[str],
    limit: int = 12,
    lookback_days: int = 30,
    today: date | None = None,
    max_per_repo: int = 2,
) -> list[dict]:
    """Fetch most +1-reacted recently active open GitHub issues across watchlist repos."""
    if not repos or limit <= 0:
        return []
    today = today or datetime.now(timezone.utc).date()
    since_iso = (today - timedelta(days=lookback_days)).isoformat()
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Batch 5 repos per Search API call to stay well within query-length limits.
    batches = [repos[i : i + 5] for i in range(0, len(repos), 5)]

    def _query_batch(batch: list[str]) -> list[dict]:
        repo_qualifiers = " ".join(f"repo:{r}" for r in batch)
        q = f"{repo_qualifiers} is:issue is:open updated:>={since_iso}"
        params = urllib.parse.urlencode(
            {"q": q, "sort": "reactions-+1", "order": "desc", "per_page": 15}
        )
        req = urllib.request.Request(f"{GH_ISSUE_SEARCH}?{params}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp)
        except Exception as e:
            print(f"Warning: GitHub issue search failed for {batch}: {e}", file=sys.stderr)
            return []

        found: list[dict] = []
        for it in data.get("items") or []:
            if "pull_request" in it:
                continue
            repo_url = str(it.get("repository_url") or "")
            repo_name = "/".join(repo_url.rstrip("/").split("/")[-2:])
            reactions = it.get("reactions") or {}
            plus1 = int(reactions.get("+1") or 0)
            total_reactions = int(reactions.get("total_count") or plus1)
            comments = int(it.get("comments") or 0)
            if plus1 < 5 and comments < 10:
                continue
            found.append(
                {
                    "repo": repo_name,
                    "number": int(it.get("number") or 0),
                    "title": (it.get("title") or "").strip(),
                    "url": (it.get("html_url") or "").strip(),
                    "plus1": plus1,
                    "total_reactions": total_reactions,
                    "comments": comments,
                    "created_at": str(it.get("created_at") or "")[:10],
                    "updated_at": str(it.get("updated_at") or "")[:10],
                    "snippet": _clean_issue_body(it.get("body")),
                }
            )
        return found

    all_issues: list[dict] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch_items in pool.map(_query_batch, batches):
            all_issues.extend(batch_items)

    # Deduplicate by URL, sort by +1 reactions then comments, and cap per repo
    # so one mega-popular repo doesn't crowd out the rest of the watchlist.
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for issue in all_issues:
        if issue["url"] and issue["url"] not in seen_urls:
            seen_urls.add(issue["url"])
            unique.append(issue)
    unique.sort(key=lambda x: (x["plus1"], x["comments"]), reverse=True)
    balanced: list[dict] = []
    repo_counts: dict[str, int] = {}
    for issue in unique:
        repo = issue["repo"]
        if max_per_repo > 0 and repo_counts.get(repo, 0) >= max_per_repo:
            continue
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        balanced.append(issue)
        if len(balanced) >= limit:
            break
    return balanced


# --------------------------------------------------------------------------- #
# Cross-platform convergence detector
# --------------------------------------------------------------------------- #

def normalize_url(url: str) -> str:
    if not url:
        return ""
    p = urllib.parse.urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return f"{host}{path}"


def detect_cross_platform(
    hn_stories: list[dict],
    lobsters_items: list[dict],
) -> list[dict]:
    """Find URLs discussed on both Hacker News and Lobste.rs."""
    hn_by_url: dict[str, dict] = {}
    for s in hn_stories:
        norm = normalize_url(s.get("url", ""))
        if norm and "news.ycombinator.com" not in norm:
            hn_by_url[norm] = s

    overlaps: list[dict] = []
    seen_norm: set[str] = set()
    for lob in lobsters_items:
        norm = normalize_url(lob.get("url", ""))
        if norm and norm in hn_by_url and norm not in seen_norm:
            seen_norm.add(norm)
            hn = hn_by_url[norm]
            overlaps.append(
                {
                    "title": hn.get("title") or lob.get("title"),
                    "url": hn.get("url") or lob.get("url"),
                    "hn_url": hn.get("hn_url", ""),
                    "lobsters_url": lob.get("comments_url", ""),
                    "hn_points": hn.get("points", 0),
                    "lobsters_score": lob.get("score", 0),
                }
            )
    return overlaps


# --------------------------------------------------------------------------- #
# Weekly LLM synthesis prompt & parsing
# --------------------------------------------------------------------------- #

def build_weekly_prompt(
    week_id: str,
    daily_archives: list[dict],
    bootstrap_data: dict | None,
    prior_weeks: list[dict],
    lobsters: list[dict],
    issues: list[dict],
    overlaps: list[dict],
) -> tuple[str, str]:
    sections: list[str] = []

    if prior_weeks:
        pw_lines = ["PRIOR WEEKS' SYNTHESES (compare against these to detect what persisted vs. faded):"]
        for pw in prior_weeks:
            w = pw.get("week", "")
            take = (pw.get("week_in_one_take") or "").strip()
            themes = ", ".join(
                f"{p.get('theme')} ({p.get('status')})"
                for p in (pw.get("persisted_vs_faded") or [])[:4]
                if isinstance(p, dict)
            )
            pw_lines.append(f"- {w}: {take}" + (f" | Themes: {themes}" if themes else ""))
        sections.append("\n".join(pw_lines))

    if daily_archives:
        day_lines = [f"ACCUMULATED DAILY HN SIGNALS ({len(daily_archives)} days archived):"]
        for d in daily_archives:
            day_lines.append(f"\n[{d.get('date')}] Overview: {d.get('overview', '')}")
            if d.get("trends"):
                day_lines.append("  Trends: " + " | ".join(d["trends"][:4]))
            for sig in (d.get("signals") or [])[:5]:
                day_lines.append(
                    f"  Signal ({sig.get('type')}, {sig.get('topic')}): {sig.get('summary')} "
                    f"[HN: https://news.ycombinator.com/item?id={sig.get('story_id', '')}]"
                )
            for st in (d.get("stories") or [])[:5]:
                line = (
                    f"  Story: {st.get('title')} ({st.get('points')} pts, {st.get('comments')} cmts) "
                    f"[{st.get('hn_url')}]"
                )
                if st.get("hn_take"):
                    line += f" — HN take: {st['hn_take']}"
                day_lines.append(line)
            for ex in (d.get("ask_show") or [])[:3]:
                day_lines.append(
                    f"  Ask/Show HN: {ex.get('title')} ({ex.get('points')} pts, {ex.get('comments')} cmts) "
                    f"[{ex.get('hn_url')}]"
                )
                for c in (ex.get("top_comments") or [])[:2]:
                    day_lines.append(f"    * {c[:240]}")
        sections.append("\n".join(day_lines))
    elif bootstrap_data:
        b_lines = ["PAST 7 DAYS HACKER NEWS (live Algolia bootstrap):"]
        for st in (bootstrap_data.get("stories") or [])[:15]:
            b_lines.append(
                f"- {st['title']} ({st['points']} pts, {st['comments']} cmts) [{st['hn_url']}]"
            )
            for c in (st.get("top_comments") or [])[:2]:
                b_lines.append(f"    * {c[:240]}")
        if bootstrap_data.get("ask_show"):
            b_lines.append("\nTOP ASK HN / SHOW HN PAIN & FEEDBACK THREADS:")
            for ex in bootstrap_data["ask_show"]:
                b_lines.append(
                    f"- {ex['title']} ({ex['points']} pts, {ex['comments']} cmts) [{ex['hn_url']}]"
                )
                for c in (ex.get("top_comments") or [])[:3]:
                    b_lines.append(f"    * {c[:260]}")
        sections.append("\n".join(b_lines))

    if lobsters:
        lob_lines = ["LOBSTE.RS HOTTEST ENGINEERING DISCUSSIONS:"]
        for it in lobsters[:12]:
            tags = ", ".join(it.get("tags") or [])
            lob_lines.append(
                f"- {it['title']} ({it['score']} pts, {it['comments']} cmts; tags: {tags}) [{it['comments_url']}]"
            )
        sections.append("\n".join(lob_lines))

    if issues:
        iss_lines = ["BELLWETHER OPEN GITHUB ISSUES (quantified unresolved pain points):"]
        for it in issues[:12]:
            iss_lines.append(
                f"- [{it['repo']}#{it['number']}] {it['title']} "
                f"(+{it['plus1']} thumbs-up, {it['comments']} cmts, opened {it['created_at']}) [{it['url']}]"
            )
            if it.get("snippet"):
                iss_lines.append(f"    Context: {it['snippet']}")
        sections.append("\n".join(iss_lines))

    if overlaps:
        ov_lines = ["EXACT HN x LOBSTE.RS CROSS-PLATFORM OVERLAPS:"]
        for ov in overlaps:
            ov_lines.append(
                f"- {ov['title']} (HN {ov['hn_points']} pts, Lobsters {ov['lobsters_score']} pts) "
                f"[{ov['url']}]"
            )
        sections.append("\n".join(ov_lines))

    system = (
        "You are a senior engineering editor and research producer preparing a "
        "weekly compounding intelligence brief for a software engineer who writes "
        "deep technical essays and builds sharp developer tools. Your job is NOT "
        "to list links — it is to CONNECT THE DOTS across Hacker News, Ask/Show HN "
        "comments, Lobste.rs, and high-reaction GitHub issues. Filter out 1-day hype, "
        "surface genuine multi-thread patterns, steelman counterarguments from real "
        "comments, and ground every pain point in concrete evidence/URLs provided "
        "below. Never fabricate URLs or stats. Respond ONLY with valid JSON."
    )

    user = (
        f"Week: {week_id}\n\n"
        + "\n\n".join(sections)
        + "\n\nRespond with a JSON object with exactly these keys:\n"
        '  "week_in_one_take": 3-4 crisp sentences summarizing what actually mattered this week vs. short-lived noise.\n'
        '  "persisted_vs_faded": array of 3-5 objects: {"theme": "short name", "status": "persisted"|"emerging"|"faded", "detail": "1-2 sentences explaining why"}.\n'
        '  "article_pitches": array of 2-3 high-conviction "connect-the-dots" tech essay pitches, each:\n'
        '      {"working_title": "sharp non-clickbait title", "core_thesis": "2 sentences articulating the non-obvious pattern linking 3+ threads/issues", "evidence": [{"label": "short source title", "url": "exact URL from input"}], "counterargument": "1-2 sentences capturing the sharpest skeptic pushback"}.\n'
        '  "validated_pain_points": array of 3-4 concrete, recurring engineering pain points, each:\n'
        '      {"problem": "1 crisp sentence stating the friction", "proof": "specific citation of Ask/Show HN comments or GitHub issue +1 count", "url": "exact URL from input", "build_angle": "1-2 sentences on a focused weekend tool, plugin, or benchmark that addresses it"}.\n'
        '  "cross_platform_highlights": array of up to 4 notable items converging across HN, Lobste.rs, or GitHub, each:\n'
        '      {"title": "...", "url": "exact URL from input", "why_it_mattered": "1 crisp sentence"}.\n'
        "Plain text only inside all strings (no markdown)."
    )
    return system, user


def parse_weekly_summary(content: str) -> dict:
    """Parse weekly structured JSON safely with graceful fallbacks."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "week_in_one_take": content,
            "persisted_vs_faded": [],
            "article_pitches": [],
            "validated_pain_points": [],
            "cross_platform_highlights": [],
        }

    pvf = []
    for entry in obj.get("persisted_vs_faded") or []:
        if not isinstance(entry, dict):
            continue
        theme = str(entry.get("theme", "")).strip()
        detail = str(entry.get("detail", "")).strip()
        status = str(entry.get("status", "emerging")).strip().lower()
        if status not in {"persisted", "emerging", "faded"}:
            status = "emerging"
        if theme and detail:
            pvf.append({"theme": theme, "status": status, "detail": detail})

    pitches = []
    for entry in obj.get("article_pitches") or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("working_title", "")).strip()
        thesis = str(entry.get("core_thesis", "")).strip()
        if not (title and thesis):
            continue
        evidence = []
        for ev in entry.get("evidence") or []:
            if isinstance(ev, dict) and ev.get("url"):
                evidence.append(
                    {
                        "label": str(ev.get("label") or ev.get("url")).strip(),
                        "url": str(ev.get("url")).strip(),
                    }
                )
        pitches.append(
            {
                "working_title": title,
                "core_thesis": thesis,
                "evidence": evidence,
                "counterargument": str(entry.get("counterargument", "")).strip(),
            }
        )

    pains = []
    for entry in obj.get("validated_pain_points") or []:
        if not isinstance(entry, dict):
            continue
        problem = str(entry.get("problem", "")).strip()
        if not problem:
            continue
        pains.append(
            {
                "problem": problem,
                "proof": str(entry.get("proof", "")).strip(),
                "url": str(entry.get("url", "")).strip(),
                "build_angle": str(entry.get("build_angle", "")).strip(),
            }
        )

    highlights = []
    for entry in obj.get("cross_platform_highlights") or []:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title", "")).strip()
        if not title:
            continue
        highlights.append(
            {
                "title": title,
                "url": str(entry.get("url", "")).strip(),
                "why_it_mattered": str(entry.get("why_it_mattered", "")).strip(),
            }
        )

    return {
        "week_in_one_take": str(obj.get("week_in_one_take", "")).strip(),
        "persisted_vs_faded": pvf,
        "article_pitches": pitches,
        "validated_pain_points": pains,
        "cross_platform_highlights": highlights,
    }


def summarize_weekly(
    week_id: str,
    daily_archives: list[dict],
    bootstrap_data: dict | None,
    prior_weeks: list[dict],
    lobsters: list[dict],
    issues: list[dict],
    overlaps: list[dict],
) -> dict:
    system, user = build_weekly_prompt(
        week_id,
        daily_archives=daily_archives,
        bootstrap_data=bootstrap_data,
        prior_weeks=prior_weeks,
        lobsters=lobsters,
        issues=issues,
        overlaps=overlaps,
    )
    # Deeper synthesis benefits from medium thinking and a generous timeout.
    weekly_timeout = _env_int("WEEKLY_LLM_TIMEOUT", 240)
    weekly_effort = (os.environ.get("WEEKLY_REASONING_EFFORT") or "medium").strip().lower()
    content = call_llm(
        system,
        user,
        reasoning_effort=weekly_effort,
        timeout=weekly_timeout,
    )
    return parse_weekly_summary(content)


# --------------------------------------------------------------------------- #
# Rendering (Editorial Indigo Theme — distinct from HN orange & GitHub dark)
# --------------------------------------------------------------------------- #

_STATUS_STYLES = {
    "persisted": ("#166534", "#dcfce7", "PERSISTED"),
    "emerging": ("#1d4ed8", "#dbeafe", "EMERGING"),
    "faded": ("#4b5563", "#f3f4f6", "FADED"),
}


def render_weekly_html(summary: dict, week_id: str, date_str: str, meta: dict) -> str:
    take_html = escape(summary.get("week_in_one_take", "")).replace("\n", "<br>")

    # 1. Persisted vs Faded
    pvf_html = ""
    pvf = summary.get("persisted_vs_faded") or []
    if pvf:
        items = []
        for entry in pvf:
            fg, bg, label = _STATUS_STYLES.get(entry["status"], _STATUS_STYLES["emerging"])
            items.append(
                f'<div style="padding:8px 0;border-bottom:1px solid #f1f5f9;">'
                f'<span style="display:inline-block;font-size:10px;font-weight:700;'
                f'letter-spacing:.5px;color:{fg};background:{bg};padding:2px 7px;'
                f'border-radius:10px;margin-right:8px;vertical-align:middle;">{label}</span>'
                f'<strong style="color:#0f172a;font-size:14.5px;">{escape(entry["theme"])}</strong>'
                f'<div style="font-size:13.5px;line-height:1.5;color:#334155;margin-top:3px;">'
                f'{escape(entry["detail"])}</div></div>'
            )
        pvf_html = (
            '<div style="padding:0 20px 18px;">'
            '<div style="font-size:12px;font-weight:700;color:#4f46e5;text-transform:uppercase;'
            'letter-spacing:.6px;margin-bottom:6px;">Signal Trajectory — Persisted vs. Faded</div>'
            + "".join(items)
            + "</div>"
        )

    # 2. Connect-the-Dots Article Pitches
    pitches_html = ""
    pitches = summary.get("article_pitches") or []
    if pitches:
        cards = []
        for idx, p in enumerate(pitches, 1):
            ev_links = " &middot; ".join(
                f'<a href="{escape(e["url"])}" style="color:#4f46e5;text-decoration:none;">'
                f'{escape(e["label"])}</a>'
                for e in p.get("evidence") or []
            )
            ev_block = (
                f'<div style="font-size:12.5px;color:#64748b;margin-top:8px;">'
                f'<strong>Evidence:</strong> {ev_links}</div>'
                if ev_links
                else ""
            )
            counter_block = (
                f'<div style="margin-top:8px;padding:8px 10px;background:#fffbeb;'
                f'border-left:2px solid #d97706;border-radius:4px;font-size:13px;'
                f'line-height:1.45;color:#78350f;">'
                f'<strong>Strongest counterargument:</strong> {escape(p["counterargument"])}</div>'
                if p.get("counterargument")
                else ""
            )
            cards.append(
                f'<div style="margin-bottom:14px;padding:14px 16px;background:#f8fafc;'
                f'border:1px solid #e2e8f0;border-left:3px solid #4f46e5;border-radius:8px;">'
                f'<div style="font-size:11px;font-weight:700;color:#4f46e5;text-transform:uppercase;'
                f'letter-spacing:.5px;">Pitch #{idx}</div>'
                f'<div style="font-size:16px;font-weight:700;color:#0f172a;margin:3px 0 6px;">'
                f'{escape(p["working_title"])}</div>'
                f'<div style="font-size:14px;line-height:1.55;color:#1e293b;">'
                f'{escape(p["core_thesis"])}</div>'
                f'{ev_block}{counter_block}</div>'
            )
        pitches_html = (
            '<div style="padding:0 20px 10px;">'
            '<div style="font-size:12px;font-weight:700;color:#4f46e5;text-transform:uppercase;'
            'letter-spacing:.6px;margin-bottom:10px;">Connect-the-Dots Article Pitches</div>'
            + "".join(cards)
            + "</div>"
        )

    # 3. Validated Pain Points
    pains_html = ""
    pains = summary.get("validated_pain_points") or []
    if pains:
        cards = []
        for p in pains:
            proof_text = escape(p.get("proof", ""))
            if p.get("url"):
                proof_html = (
                    f'<a href="{escape(p["url"])}" style="color:#0f766e;text-decoration:none;">'
                    f'{proof_text or escape(p["url"])}</a>'
                )
            else:
                proof_html = proof_text
            angle_html = (
                f'<div style="margin-top:6px;padding-left:8px;border-left:2px solid #0d9488;'
                f'font-size:13px;line-height:1.45;color:#134e4a;">'
                f'<strong>Build angle:</strong> {escape(p["build_angle"])}</div>'
                if p.get("build_angle")
                else ""
            )
            cards.append(
                f'<div style="margin-bottom:12px;padding:12px 14px;background:#f0fdfa;'
                f'border:1px solid #ccfbf1;border-radius:8px;">'
                f'<div style="font-size:14.5px;font-weight:600;color:#0f172a;">'
                f'{escape(p["problem"])}</div>'
                f'<div style="font-size:12.5px;color:#475569;margin-top:4px;">'
                f'<strong>Proof:</strong> {proof_html}</div>'
                f'{angle_html}</div>'
            )
        pains_html = (
            '<div style="padding:0 20px 12px;">'
            '<div style="font-size:12px;font-weight:700;color:#0f766e;text-transform:uppercase;'
            'letter-spacing:.6px;margin-bottom:10px;">Validated Pain Points &amp; Build Angles</div>'
            + "".join(cards)
            + "</div>"
        )

    # 4. Cross-Platform Highlights
    cross_html = ""
    highlights = summary.get("cross_platform_highlights") or []
    if highlights:
        items = []
        for h in highlights:
            title_el = (
                f'<a href="{escape(h["url"])}" style="color:#1d4ed8;text-decoration:none;font-weight:600;">'
                f'{escape(h["title"])}</a>'
                if h.get("url")
                else f'<strong>{escape(h["title"])}</strong>'
            )
            items.append(
                f'<li style="margin:6px 0;font-size:14px;line-height:1.5;color:#1e293b;">'
                f'{title_el} — {escape(h.get("why_it_mattered", ""))}</li>'
            )
        cross_html = (
            '<div style="padding:0 20px 18px;">'
            '<div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;'
            'letter-spacing:.6px;margin-bottom:6px;">Cross-Platform Convergence</div>'
            f'<ul style="margin:0;padding-left:18px;">{"".join(items)}</ul></div>'
        )

    source_summary = (
        f'{meta.get("hn_days", 0)} daily HN archives'
        if meta.get("hn_days")
        else "7-day HN bootstrap"
    )
    footer_meta = (
        f'{source_summary} &middot; {meta.get("lobsters_count", 0)} Lobste.rs threads &middot; '
        f'{meta.get("issues_count", 0)} bellwether GitHub issues'
    )

    return f"""<!DOCTYPE html>
<html><body style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f1f5f9;margin:0;padding:20px;">
  <div style="max-width:660px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;">
    <div style="background:#312e81;color:#fff;padding:18px 22px;">
      <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#a5b4fc;">
        Weekly Compounding Brief &middot; {escape(week_id)}
      </div>
      <div style="font-size:21px;font-weight:700;margin-top:3px;">Tech Signal &amp; Angles</div>
      <div style="font-size:13px;color:#c7d2fe;margin-top:2px;">{escape(date_str)}</div>
    </div>
    <div style="padding:20px;font-size:15px;line-height:1.6;color:#0f172a;">
      {take_html}
    </div>
    {pvf_html}
    {pitches_html}
    {pains_html}
    {cross_html}
    <div style="padding:14px 20px;font-size:12px;color:#64748b;border-top:1px solid #e2e8f0;background:#f8fafc;">
      Synthesized from {footer_meta}
    </div>
  </div>
</body></html>"""


def render_weekly_text(summary: dict, week_id: str, date_str: str) -> str:
    lines = [
        f"WEEKLY TECH SIGNAL & ANGLES ({week_id}) — {date_str}",
        "",
        summary.get("week_in_one_take", ""),
    ]

    pvf = summary.get("persisted_vs_faded") or []
    if pvf:
        lines += ["", "PERSISTED VS. FADED:"]
        for entry in pvf:
            lines.append(
                f"- [{entry['status'].upper()}] {entry['theme']}: {entry['detail']}"
            )

    pitches = summary.get("article_pitches") or []
    if pitches:
        lines += ["", "CONNECT-THE-DOTS ARTICLE PITCHES:"]
        for idx, p in enumerate(pitches, 1):
            lines.append(f"{idx}. {p['working_title']}")
            lines.append(f"   Thesis: {p['core_thesis']}")
            for ev in p.get("evidence") or []:
                lines.append(f"   Evidence: {ev['label']} ({ev['url']})")
            if p.get("counterargument"):
                lines.append(f"   Counterargument: {p['counterargument']}")

    pains = summary.get("validated_pain_points") or []
    if pains:
        lines += ["", "VALIDATED PAIN POINTS & BUILD ANGLES:"]
        for p in pains:
            lines.append(f"- {p['problem']}")
            if p.get("proof"):
                lines.append(f"  Proof: {p['proof']}" + (f" ({p['url']})" if p.get("url") else ""))
            if p.get("build_angle"):
                lines.append(f"  Build angle: {p['build_angle']}")

    highlights = summary.get("cross_platform_highlights") or []
    if highlights:
        lines += ["", "CROSS-PLATFORM CONVERGENCE:"]
        for h in highlights:
            lines.append(f"- {h['title']}: {h.get('why_it_mattered', '')}")
            if h.get("url"):
                lines.append(f"  {h['url']}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional Notion export (zero external deps)
# --------------------------------------------------------------------------- #

def build_notion_payload(database_id: str, week_id: str, summary: dict) -> dict:
    """Build a Notion API /v1/pages payload for a weekly brief."""
    def _rt(text: str, *, bold: bool = False, url: str | None = None) -> dict:
        obj: dict = {"type": "text", "text": {"content": text[:1900]}}
        if url:
            obj["text"]["link"] = {"url": url}
        if bold:
            obj["annotations"] = {"bold": True}
        return obj

    def _h2(text: str) -> dict:
        return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [_rt(text)]}}

    def _p(rich_text: list[dict]) -> dict:
        return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text}}

    def _bullet(rich_text: list[dict]) -> dict:
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": rich_text},
        }

    children: list[dict] = []
    if summary.get("week_in_one_take"):
        children.append(_h2("The Week in One Take"))
        children.append(_p([_rt(summary["week_in_one_take"])]))

    if summary.get("persisted_vs_faded"):
        children.append(_h2("Persisted vs. Faded"))
        for entry in summary["persisted_vs_faded"]:
            children.append(
                _bullet([
                    _rt(f"[{entry['status'].upper()}] {entry['theme']}: ", bold=True),
                    _rt(entry["detail"]),
                ])
            )

    if summary.get("article_pitches"):
        children.append(_h2("Connect-the-Dots Article Pitches"))
        for p in summary["article_pitches"]:
            children.append(_p([_rt(p["working_title"], bold=True)]))
            children.append(_p([_rt(p["core_thesis"])]))
            if p.get("counterargument"):
                children.append(
                    _bullet([_rt("Counterargument: ", bold=True), _rt(p["counterargument"])])
                )
            for ev in p.get("evidence") or []:
                children.append(_bullet([_rt(ev["label"], url=ev["url"])]))

    if summary.get("validated_pain_points"):
        children.append(_h2("Validated Pain Points & Build Angles"))
        for p in summary["validated_pain_points"]:
            rt = [_rt(p["problem"], bold=True), _rt(f" — {p.get('proof', '')}")]
            if p.get("url"):
                rt.append(_rt(" "))
                rt.append(_rt("[link]", url=p["url"]))
            children.append(_bullet(rt))
            if p.get("build_angle"):
                children.append(_p([_rt("Build angle: ", bold=True), _rt(p["build_angle"])]))

    return {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {
                "title": [{"text": {"content": f"Weekly Tech Signal — {week_id}"}}]
            }
        },
        "children": children[:95],  # Notion caps initial children at 100 blocks
    }


def push_to_notion(week_id: str, summary: dict) -> bool:
    """Optionally publish the weekly brief to a Notion database (fail-soft)."""
    api_key = (os.environ.get("NOTION_API_KEY") or "").strip()
    database_id = (os.environ.get("NOTION_DATABASE_ID") or "").strip()
    if not (api_key and database_id):
        return False

    payload = json.dumps(build_notion_payload(database_id, week_id, summary)).encode()
    req = urllib.request.Request(
        NOTION_PAGES_API,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"Warning: Notion export failed: {e}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    archive_dir = (os.environ.get("ARCHIVE_DIR") or ".data").strip()
    watchlist_file = (os.environ.get("WATCHLIST_FILE") or "watchlist.json").strip()

    now = datetime.now(timezone.utc)
    iso_year, iso_week, _ = now.isocalendar()
    week_id = f"{iso_year}-W{iso_week:02d}"
    date_str = now.strftime("%A, %d %B %Y (UTC)")

    daily_archives = load_daily_archives(archive_dir, days=7, today=now.date())
    prior_weeks = load_prior_weekly_archives(archive_dir, limit=3, before_week=week_id)

    bootstrap_data = None
    if len(daily_archives) < 3:
        print(
            f"Found {len(daily_archives)} daily archive(s) (<3); running 7-day Algolia HN bootstrap...",
            file=sys.stderr,
        )
        bootstrap_data = bootstrap_hn_week()
    else:
        print(
            f"Loaded {len(daily_archives)} daily HN archives and {len(prior_weeks)} prior weekly brief(s).",
            file=sys.stderr,
        )

    repos = load_watchlist(watchlist_file)
    print(
        f"Fetching Lobste.rs hottest and bellwether issues across {len(repos)} repos...",
        file=sys.stderr,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        lob_fut = pool.submit(fetch_lobsters_hottest, 15)
        iss_fut = pool.submit(fetch_bellwether_issues, repos, 12, 30, now.date())
        lobsters = lob_fut.result()
        issues = iss_fut.result()

    # Collect HN stories across archives or bootstrap for overlap detection.
    all_hn_stories: list[dict] = []
    if daily_archives:
        for d in daily_archives:
            all_hn_stories.extend(d.get("stories") or [])
    elif bootstrap_data:
        all_hn_stories.extend(bootstrap_data.get("stories") or [])
    overlaps = detect_cross_platform(all_hn_stories, lobsters)

    print(
        f"Gathered {len(lobsters)} Lobste.rs threads, {len(issues)} bellwether issues, "
        f"{len(overlaps)} exact HN x Lobste.rs overlaps.",
        file=sys.stderr,
    )

    print("Synthesizing weekly brief with the LLM...", file=sys.stderr)
    summary = summarize_weekly(
        week_id,
        daily_archives=daily_archives,
        bootstrap_data=bootstrap_data,
        prior_weeks=prior_weeks,
        lobsters=lobsters,
        issues=issues,
        overlaps=overlaps,
    )

    meta = {
        "hn_days": len(daily_archives),
        "bootstrapped": bootstrap_data is not None,
        "lobsters_count": len(lobsters),
        "issues_count": len(issues),
        "overlaps_count": len(overlaps),
    }

    html = render_weekly_html(summary, week_id, date_str, meta)
    text = render_weekly_text(summary, week_id, date_str)
    subject = f"Weekly Tech Signal & Angles — {week_id}"

    if send_email(subject, html, text):
        print("Weekly synthesis emailed successfully.", file=sys.stderr)
    else:
        print("SMTP not configured — printing weekly synthesis below.\n", file=sys.stderr)
        print(text)

    if push_to_notion(week_id, summary):
        print("Weekly synthesis published to Notion.", file=sys.stderr)

    if archive_dir:
        out_path = save_weekly_archive(archive_dir, week_id, summary, meta)
        print(f"Saved weekly archive to {out_path}.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
