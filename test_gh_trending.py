#!/usr/bin/env python3
"""Unit tests for gh_trending.py (standard library only)."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import gh_trending


SAMPLE_TRENDING_HTML = """<!DOCTYPE html>
<html><body>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/acme/fast-db" class="Link">
      <span class="text-normal">acme /</span>
      fast-db
    </a>
  </h2>
  <p class="col-9 color-fg-muted my-1">
    An embedded columnar &amp; vector engine written in Rust.
  </p>
  <div class="f6 color-fg-muted mt-2">
    <span>
      <span class="repo-language-color" style="background-color: #dea584"></span>
      <span itemprop="programmingLanguage">Rust</span>
    </span>
    <a href="/acme/fast-db/stargazers">1,850</a>
    <a href="/acme/fast-db/forks">142</a>
    <span class="d-inline-block float-sm-right">
      1,240 stars today
    </span>
  </div>
</article>
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a href="/bigcorp/mega-framework" class="Link">bigcorp / mega-framework</a>
  </h2>
  <p class="col-9 color-fg-muted my-1">Production web framework.</p>
  <div class="f6 color-fg-muted mt-2">
    <span>
      <span class="repo-language-color" style="background-color: #3572A5"></span>
      <span itemprop="programmingLanguage">Python</span>
    </span>
    <a href="/bigcorp/mega-framework/stargazers">48,200</a>
    <a href="/bigcorp/mega-framework/forks">6,110</a>
    <span class="d-inline-block float-sm-right">
      620 stars today
    </span>
  </div>
</article>
</body></html>
"""


def _mock_http_response(body: bytes):
    resp = MagicMock()
    buf = io.BytesIO(body)
    resp.read.side_effect = buf.read
    cm = MagicMock()
    cm.__enter__.return_value = resp
    return cm


class TestParseTrendingHtml(unittest.TestCase):
    def test_parses_real_github_box_row_structure(self):
        repos = gh_trending.parse_trending_html(SAMPLE_TRENDING_HTML)
        self.assertEqual(len(repos), 2)

        first = repos[0]
        self.assertEqual(first["full_name"], "acme/fast-db")
        self.assertEqual(first["url"], "https://github.com/acme/fast-db")
        self.assertEqual(
            first["description"], "An embedded columnar & vector engine written in Rust."
        )
        self.assertEqual(first["language"], "Rust")
        self.assertEqual(first["language_color"], "#dea584")
        self.assertEqual(first["stars"], 1850)
        self.assertEqual(first["forks"], 142)
        self.assertEqual(first["stars_today"], 1240)

        second = repos[1]
        self.assertEqual(second["full_name"], "bigcorp/mega-framework")
        self.assertEqual(second["stars"], 48200)
        self.assertEqual(second["stars_today"], 620)


class TestFetchAndDedup(unittest.TestCase):
    def test_fetch_trending_dedups_and_supports_language(self):
        captured = []

        def fake_urlopen(req, timeout=30):
            captured.append(req.full_url)
            return _mock_http_response(SAMPLE_TRENDING_HTML.encode("utf-8"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            repos = gh_trending.fetch_trending_repos(
                5, language="rust", since="daily", seen_ids={"acme/fast-db"}
            )

        self.assertEqual(captured[0], "https://github.com/trending/rust?since=daily")
        self.assertEqual([r["full_name"] for r in repos], ["bigcorp/mega-framework"])

    def test_fetch_trending_falls_back_if_all_seen(self):
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_http_response(SAMPLE_TRENDING_HTML.encode("utf-8")),
        ):
            repos = gh_trending.fetch_trending_repos(
                5, seen_ids={"acme/fast-db", "bigcorp/mega-framework"}
            )
        self.assertEqual(len(repos), 2)


class TestReadmeAndBreakout(unittest.TestCase):
    def test_clean_readme_strips_badges_code_and_html(self):
        md = """
        <p align="center"><img src="logo.png" /></p>
        [![CI](https://img.shields.io/badge/ci-pass-green)](https://ci.example.com)
        # FastDB
        FastDB is a zero-copy [columnar engine](https://example.com) built for local analytics on NVMe drives with sub-millisecond scans.
        ```bash
        cargo install fast-db
        ```
        """
        cleaned = gh_trending.clean_readme_markdown(md)
        self.assertIn("FastDB is a zero-copy columnar engine", cleaned)
        self.assertNotIn("shields.io", cleaned)
        self.assertNotIn("cargo install", cleaned)

    def test_find_breakout_repo_prefers_high_velocity_relative_to_size(self):
        repos = gh_trending.parse_trending_html(SAMPLE_TRENDING_HTML)
        winner = gh_trending.find_breakout_repo(repos)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["full_name"], "acme/fast-db")


class TestParsingAndRendering(unittest.TestCase):
    def test_parse_summary_and_render_html_and_text(self):
        repos = gh_trending.parse_trending_html(SAMPLE_TRENDING_HTML)
        raw = json.dumps(
            {
                "overview": "Rust storage engines and Python frameworks lead today.",
                "themes": ["Embedded Rust databases", "Developer ergonomics"],
                "breakout_repo": {
                    "full_name": "acme/fast-db",
                    "summary": "A new Rust columnar store gaining 1,200+ stars in a single day.",
                },
                "build_ideas": (
                    "Pair acme/fast-db with bigcorp/mega-framework to ship a zero-ops local "
                    "analytics dashboard for indie SaaS teams."
                ),
                "repos": [
                    {
                        "full_name": "acme/fast-db",
                        "tldr": "Embedded Rust columnar and vector database for single-node analytics.",
                        "why_notable": "Combines DuckDB-style scans with HNSW vector search in one binary.",
                    },
                    {
                        "full_name": "bigcorp/mega-framework",
                        "tldr": "Full-stack Python web framework.",
                        "why_notable": "New v5 release adds native async streaming.",
                    },
                ],
            }
        )
        summary = gh_trending.parse_summary(raw)
        html = gh_trending.render_html(summary, repos, "Sunday, 22 November 2026 (UTC)")
        text = gh_trending.render_text(summary, repos, "Sunday, 22 November 2026 (UTC)")

        self.assertIn("GitHub Trending Daily Digest", html)
        self.assertIn("Breakout Repo", html)
        self.assertIn("breakout</span>", html)
        self.assertIn("#dea584", html)
        self.assertIn("+1,240 today", html)
        self.assertIn("<strong>Why notable:</strong> Combines DuckDB-style scans", html)
        self.assertIn("Idea Spark", html)
        self.assertIn("zero-ops local analytics dashboard", html)

        self.assertIn("GITHUB TRENDING DAILY DIGEST", text)
        self.assertIn("BREAKOUT REPO:", text)
        self.assertIn("TL;DR: Embedded Rust columnar", text)
        self.assertIn("Why notable: Combines DuckDB-style scans", text)
        self.assertIn("IDEA SPARK", text)
        self.assertIn("zero-ops local analytics dashboard", text)

    def test_summarize_always_pins_low_reasoning_effort(self):
        repos = gh_trending.parse_trending_html(SAMPLE_TRENDING_HTML)
        with patch.object(
            gh_trending,
            "call_llm",
            return_value=json.dumps({"overview": "ok", "themes": [], "repos": []}),
        ) as mocked:
            gh_trending.summarize_with_llm(repos)
        self.assertEqual(mocked.call_args.kwargs.get("reasoning_effort"), "low")


if __name__ == "__main__":
    unittest.main()
