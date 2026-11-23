#!/usr/bin/env python3
"""Unit tests for hn_digest.py (standard library only)."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import hn_digest


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


class TestFetchTopStories(unittest.TestCase):
    def test_fetches_full_front_page_before_sorting_and_slicing(self):
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
            cm = MagicMock()
            cm.__enter__.return_value = io.BytesIO(
                json.dumps({"hits": fake_hits}).encode()
            )
            return cm

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            top5 = hn_digest.fetch_top_stories(5)

        self.assertIn("hitsPerPage=30", captured_urls[0])
        self.assertEqual(len(top5), 5)
        self.assertEqual([s["points"] for s in top5], [300, 290, 280, 270, 260])
        self.assertEqual(top5[0]["domain"], "site30.com")
        self.assertEqual(top5[1]["domain"], "")


class TestRenderingAndTypes(unittest.TestCase):
    def test_render_includes_domain_only_when_present(self):
        stories = [
            {
                "title": "External Story",
                "url": "https://github.com/example/repo",
                "domain": "github.com",
                "points": 250,
                "comments": 42,
                "hn_url": "https://news.ycombinator.com/item?id=1",
            },
            {
                "title": "Ask HN: Question",
                "url": "https://news.ycombinator.com/item?id=2",
                "domain": "",
                "points": 120,
                "comments": 80,
                "hn_url": "https://news.ycombinator.com/item?id=2",
            },
        ]
        summary = {"overview": "Test overview.", "trends": ["Trend A"]}
        html = hn_digest.render_html(summary, stories, "Sunday, 22 November 2026 (UTC)")
        text = hn_digest.render_text(summary, stories, "Sunday, 22 November 2026 (UTC)")

        self.assertIn("(github.com)", html)
        self.assertNotIn("()", html)
        self.assertIn("[github.com]", text)
        self.assertNotIn("[]", text)

    def test_summarize_return_annotation(self):
        self.assertIs(hn_digest.summarize_with_llm.__annotations__["return"], dict)


if __name__ == "__main__":
    unittest.main()
