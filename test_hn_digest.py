#!/usr/bin/env python3
"""Unit tests for hn_digest.py (standard library only)."""

from datetime import date
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import hn_digest


def _mock_response(body: bytes, content_type: str = "application/json"):
    resp = MagicMock()
    resp.read.return_value = body
    resp.headers = MagicMock()
    resp.headers.get.side_effect = lambda k, default=None: (
        content_type if k.lower() == "content-type" else default
    )
    resp.headers.get_content_charset.return_value = "utf-8"
    cm = MagicMock()
    cm.__enter__.return_value = resp
    # Make json.load(resp) work too by attaching a BytesIO read when read() called without args
    buf = io.BytesIO(body)
    resp.read.side_effect = buf.read
    return cm


class TestExtractDomain(unittest.TestCase):
    def test_strips_www_and_port(self):
        self.assertEqual(
            hn_digest.extract_domain("https://www.github.com/org/repo"), "github.com"
        )
        self.assertEqual(
            hn_digest.extract_domain("https://arxiv.org:443/abs/1234"), "arxiv.org"
        )

    def test_omits_hn_self_posts_and_empty(self):
        self.assertEqual(
            hn_digest.extract_domain("https://news.ycombinator.com/item?id=123"), ""
        )
        self.assertEqual(hn_digest.extract_domain(None), "")
        self.assertEqual(hn_digest.extract_domain(""), "")


class TestSeenState(unittest.TestCase):
    def test_save_load_and_prune_old_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "state", "seen.json")
            today = date(2026, 11, 22)
            initial = {
                "old_story": "2026-11-10",  # > 7 days old -> pruned
                "recent_story": "2026-11-20",  # within 7 days -> kept
                "bad_date": "not-a-date",  # invalid -> pruned
            }
            hn_digest.save_seen_ids(state_path, initial, ["new_1", "new_2"], today=today)
            loaded = hn_digest.load_seen_ids(state_path, today=today)

            self.assertEqual(
                loaded,
                {
                    "recent_story": "2026-11-20",
                    "new_1": "2026-11-22",
                    "new_2": "2026-11-22",
                },
            )

    def test_load_missing_and_corrupt_fail_soft(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "does_not_exist.json")
            self.assertEqual(hn_digest.load_seen_ids(missing), {})

            corrupt = os.path.join(tmp, "corrupt.json")
            with open(corrupt, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            self.assertEqual(hn_digest.load_seen_ids(corrupt), {})


class TestFetchTopStories(unittest.TestCase):
    def test_front_page_fetches_full_page_before_sorting_and_slicing(self):
        fake_hits = [
            {
                "objectID": str(i),
                "title": f"Story {i}",
                "url": f"https://www.site{i}.com/p" if i % 2 == 0 else None,
                "points": i * 10,
                "num_comments": i,
                "author": f"u{i}",
            }
            for i in range(1, 31)
        ]
        captured_urls = []

        def fake_urlopen(req, timeout=30):
            captured_urls.append(req.full_url)
            return _mock_response(json.dumps({"hits": fake_hits}).encode())

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            top5 = hn_digest.fetch_top_stories(5, fetch_mode="front_page")

        self.assertIn("hitsPerPage=30", captured_urls[0])
        self.assertEqual(len(top5), 5)
        self.assertEqual([s["points"] for s in top5], [300, 290, 280, 270, 260])
        self.assertEqual(top5[0]["domain"], "site30.com")
        self.assertEqual(top5[1]["domain"], "")

    def test_24h_mode_and_deduplication(self):
        fake_hits = [
            {"objectID": "101", "title": "Already Seen", "url": "https://a.com", "points": 500, "num_comments": 100},
            {"objectID": "102", "title": "Fresh Top", "url": "https://b.com", "points": 320, "num_comments": 80},
            {"objectID": "103", "title": "Fresh Second", "url": "https://c.com", "points": 210, "num_comments": 40},
        ]
        captured_urls = []

        def fake_urlopen(req, timeout=30):
            captured_urls.append(req.full_url)
            return _mock_response(json.dumps({"hits": fake_hits}).encode())

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            stories = hn_digest.fetch_top_stories(
                2, fetch_mode="24h", seen_ids={"101"}, now_ts=1_700_086_400
            )

        self.assertIn("tags=story", captured_urls[0])
        self.assertIn("created_at_i%3E1700000000", captured_urls[0])
        self.assertEqual([s["object_id"] for s in stories], ["102", "103"])

    def test_dedup_falls_back_if_all_stories_seen(self):
        fake_hits = [
            {"objectID": "1", "title": "Only Story", "url": "https://a.com", "points": 150, "num_comments": 20},
        ]
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(json.dumps({"hits": fake_hits}).encode()),
        ):
            stories = hn_digest.fetch_top_stories(5, fetch_mode="24h", seen_ids={"1"})
        self.assertEqual(len(stories), 1)
        self.assertEqual(stories[0]["object_id"], "1")


class TestSpiciestDebate(unittest.TestCase):
    def test_picks_highest_comments_to_points_ratio_above_floor(self):
        stories = [
            {"object_id": "1", "points": 500, "comments": 100},  # 0.20
            {"object_id": "2", "points": 120, "comments": 240},  # 2.00 -> winner
            {"object_id": "3", "points": 10, "comments": 80},    # below points floor
        ]
        winner = hn_digest.find_spiciest_debate(stories)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["object_id"], "2")

    def test_fallback_when_no_story_meets_floor(self):
        stories = [
            {"object_id": "1", "points": 30, "comments": 10},
            {"object_id": "2", "points": 40, "comments": 25},
        ]
        winner = hn_digest.find_spiciest_debate(stories)
        self.assertIsNotNone(winner)
        self.assertEqual(winner["object_id"], "2")


class TestHtmlAndArticleExtraction(unittest.TestCase):
    def test_strip_html_cleans_hn_comment_markup(self):
        raw = "<p>We tried Rust &amp; it&#x27;s <i>fast</i>.<p>See <a href='https://x.com'>docs</a>."
        self.assertEqual(
            hn_digest.strip_html(raw),
            "We tried Rust & it's fast. See docs.",
        )

    def test_extract_article_text_strips_boilerplate_and_prefers_article(self):
        html = """
        <html>
          <head><style>body { color: red; }</style><script>var x = 1;</script></head>
          <body>
            <nav>Home About Pricing Login</nav>
            <article>
              <h1>Building a Faster Compiler</h1>
              <p>We rewrote the backend IR pass to eliminate intermediate heap allocations across the hot compilation loop, reducing build times by forty percent on large production codebases while preserving deterministic debug symbols.</p>
            </article>
            <footer>Copyright 2026</footer>
          </body>
        </html>
        """
        text = hn_digest.extract_article_text(html, max_chars=200)
        self.assertIn("Building a Faster Compiler", text)
        self.assertNotIn("Pricing Login", text)
        self.assertNotIn("var x = 1", text)
        self.assertNotIn("Copyright 2026", text)
        self.assertLessEqual(len(text), 204)

    def test_fetch_article_text_skips_binaries_and_non_html(self):
        self.assertEqual(hn_digest.fetch_article_text("https://example.com/paper.pdf"), "")
        self.assertEqual(
            hn_digest.fetch_article_text("https://news.ycombinator.com/item?id=99"), ""
        )
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_response(b"%PDF-1.7 ...", content_type="application/pdf"),
        ):
            self.assertEqual(hn_digest.fetch_article_text("https://example.com/doc"), "")


class TestCommentsAndEnrichment(unittest.TestCase):
    def test_extract_comments_filters_dead_and_includes_replies(self):
        item = {
            "children": [
                {"author": "bad", "text": "[dead]"},
                {"author": "short", "text": "lol"},
                {
                    "author": "alice",
                    "text": "<p>Great benchmark, though the memory numbers exclude GC pauses under sustained load.</p>",
                    "children": [
                        {
                            "author": "bob",
                            "text": "<p>Good catch — section 4 actually measures p99 latency with GC enabled.</p>",
                        }
                    ],
                },
                {
                    "author": "carol",
                    "text": "<p>We deployed something similar in production last quarter and cut tail latency in half.</p>",
                    "children": [],
                },
            ]
        }
        comments = hn_digest.extract_comments_from_item(item, max_top=5, max_replies=1)
        self.assertEqual(len(comments), 2)
        self.assertIn("alice: Great benchmark", comments[0])
        self.assertIn("-> bob: Good catch", comments[0])
        self.assertIn("carol: We deployed something similar", comments[1])

    def test_enrich_stories_uses_story_text_and_includes_spiciest_comments(self):
        stories = [
            {
                "object_id": "1",
                "title": "Ask HN: Best books?",
                "url": "https://news.ycombinator.com/item?id=1",
                "story_text": "Looking for distributed systems recommendations.",
                "points": 300,
                "comments": 90,
            },
            {
                "object_id": "2",
                "title": "External Article",
                "url": "https://example.com/post",
                "story_text": "",
                "points": 200,
                "comments": 40,
            },
            {
                "object_id": "3",
                "title": "Spicy Thread Outside Top K",
                "url": "https://example.com/spicy",
                "story_text": "",
                "points": 100,
                "comments": 250,
            },
        ]
        fetched_articles = []
        fetched_comments = []

        def fake_article(url):
            fetched_articles.append(url)
            return f"Excerpt for {url}"

        def fake_comments(oid):
            fetched_comments.append(oid)
            return [f"Comment on {oid}"]

        with (
            patch.object(hn_digest, "fetch_article_text", side_effect=fake_article),
            patch.object(hn_digest, "fetch_story_comments", side_effect=fake_comments),
        ):
            hn_digest.enrich_stories(
                stories,
                spiciest=stories[2],
                fetch_articles=True,
                comment_top_k=1,
            )

        # Story 1 had story_text so it shouldn't hit fetch_article_text
        self.assertEqual(
            stories[0]["article_excerpt"],
            "Looking for distributed systems recommendations.",
        )
        self.assertEqual(set(fetched_articles), {"https://example.com/post", "https://example.com/spicy"})
        # Top K=1 fetched story 1, plus spiciest debate story 3
        self.assertEqual(set(fetched_comments), {"1", "3"})
        self.assertEqual(stories[1]["top_comments"], [])
        self.assertEqual(stories[2]["top_comments"], ["Comment on 3"])


class TestSummaryParsingAndRendering(unittest.TestCase):
    def test_parse_summary_full_structured_json(self):
        raw = json.dumps(
            {
                "overview": "Systems programming and AI tooling dominate.",
                "trends": ["Local inference gains", "Compiler speedups"],
                "spiciest_debate": {
                    "story_id": "2",
                    "summary": "Fans praise simplicity while critics question scalability.",
                },
                "stories": [
                    {
                        "story_id": "1",
                        "tldr": "A new zero-copy parser written in C.",
                        "hn_take": "Commenters admire benchmarks but want fuzzing results.",
                    },
                    {
                        "story_id": "2",
                        "tldr": "An essay arguing against microservices for small teams.",
                        "hn_take": "Heated split between monolith advocates and scale veterans.",
                    },
                ],
            }
        )
        parsed = hn_digest.parse_summary(raw)
        self.assertEqual(parsed["overview"], "Systems programming and AI tooling dominate.")
        self.assertEqual(len(parsed["trends"]), 2)
        self.assertEqual(parsed["spiciest_debate"]["story_id"], "2")
        self.assertEqual(
            parsed["stories"]["1"]["tldr"], "A new zero-copy parser written in C."
        )

    def test_parse_summary_fallback_on_non_json(self):
        parsed = hn_digest.parse_summary("Plain fallback summary.")
        self.assertEqual(parsed["overview"], "Plain fallback summary.")
        self.assertEqual(parsed["trends"], [])
        self.assertIsNone(parsed["spiciest_debate"])
        self.assertEqual(parsed["stories"], {})

    def test_render_html_and_text_include_enrichment_and_debate(self):
        stories = [
            {
                "object_id": "1",
                "title": "External Story <v2>",
                "url": "https://github.com/example/repo",
                "domain": "github.com",
                "points": 250,
                "comments": 42,
                "hn_url": "https://news.ycombinator.com/item?id=1",
            },
            {
                "object_id": "2",
                "title": "Ask HN: Monoliths vs Microservices?",
                "url": "https://news.ycombinator.com/item?id=2",
                "domain": "",
                "points": 120,
                "comments": 310,
                "hn_url": "https://news.ycombinator.com/item?id=2",
            },
        ]
        summary = {
            "overview": "Test overview.",
            "trends": ["Trend A"],
            "spiciest_debate": {
                "story_id": "2",
                "summary": "Monolith fans clash with distributed-systems engineers.",
            },
            "stories": {
                "1": {
                    "tldr": "Releases v2 with 3x faster indexing.",
                    "hn_take": "Commenters love the speed & ask about Windows support.",
                },
                "2": {
                    "tldr": "Community thread comparing architecture tradeoffs.",
                    "hn_take": "Broad agreement to start simple until org scale demands splits.",
                },
            },
        }
        html = hn_digest.render_html(summary, stories, "Sunday, 22 November 2026 (UTC)")
        text = hn_digest.render_text(summary, stories, "Sunday, 22 November 2026 (UTC)")

        # HTML checks
        self.assertIn("(github.com)", html)
        self.assertNotIn("()", html)
        self.assertIn("External Story &lt;v2&gt;", html)
        self.assertIn("Spiciest Debate", html)
        self.assertIn("Monolith fans clash with distributed-systems engineers.", html)
        self.assertIn("Releases v2 with 3x faster indexing.", html)
        self.assertIn("<strong>HN take:</strong> Commenters love the speed &amp; ask about Windows support.", html)
        self.assertIn("heated</span>", html)

        # Text checks
        self.assertIn("[github.com]", text)
        self.assertNotIn("[]", text)
        self.assertIn("SPICIEST DEBATE:", text)
        self.assertIn("TL;DR: Releases v2 with 3x faster indexing.", text)
        self.assertIn("HN take: Commenters love the speed & ask about Windows support.", text)

    def test_summarize_return_annotation(self):
        self.assertIs(hn_digest.summarize_with_llm.__annotations__["return"], dict)


if __name__ == "__main__":
    unittest.main()
