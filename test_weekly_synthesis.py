#!/usr/bin/env python3
"""Unit tests for weekly_synthesis.py (standard library only)."""

from datetime import date
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import hn_digest
import weekly_synthesis


def _mock_json_response(obj):
    body = json.dumps(obj).encode("utf-8")
    resp = MagicMock()
    buf = io.BytesIO(body)
    resp.read.side_effect = buf.read
    cm = MagicMock()
    cm.__enter__.return_value = resp
    return cm


class TestArchivesAndWatchlist(unittest.TestCase):
    def test_load_daily_and_prior_weekly_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "hn"), exist_ok=True)
            for d in ("2026-11-10", "2026-11-20", "2026-11-24"):
                with open(os.path.join(tmp, "hn", f"{d}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": d, "overview": f"ov {d}", "stories": []}, f)

            daily = weekly_synthesis.load_daily_archives(tmp, days=7, today=date(2026, 11, 25))
            self.assertEqual([x["date"] for x in daily], ["2026-11-20", "2026-11-24"])

            weekly_synthesis.save_weekly_archive(
                tmp, "2026-W46", {"week_in_one_take": "Week 46 take."}, {"hn_days": 5}
            )
            weekly_synthesis.save_weekly_archive(
                tmp, "2026-W47", {"week_in_one_take": "Week 47 take."}, {"hn_days": 6}
            )
            prior = weekly_synthesis.load_prior_weekly_archives(tmp, limit=2, before_week="2026-W48")
            self.assertEqual([w["week"] for w in prior], ["2026-W46", "2026-W47"])

    def test_load_watchlist_reads_json_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = os.path.join(tmp, "watchlist.json")
            with open(wl, "w", encoding="utf-8") as f:
                json.dump({"repos": ["duckdb/duckdb", "invalid", "astral-sh/uv"]}, f)
            self.assertEqual(weekly_synthesis.load_watchlist(wl), ["duckdb/duckdb", "astral-sh/uv"])
            self.assertEqual(
                weekly_synthesis.load_watchlist(os.path.join(tmp, "missing.json")),
                weekly_synthesis.DEFAULT_WATCHLIST,
            )


class TestSourcesAndOverlap(unittest.TestCase):
    def test_fetch_lobsters_hottest(self):
        fake = [
            {
                "title": "Deep Rust Kernel Dive",
                "url": "https://example.com/rust-kernel/",
                "comments_url": "https://lobste.rs/s/abc123",
                "score": 45,
                "comment_count": 22,
                "tags": ["rust", "linux"],
            }
        ]
        with patch("urllib.request.urlopen", return_value=_mock_json_response(fake)):
            res = weekly_synthesis.fetch_lobsters_hottest(limit=5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["title"], "Deep Rust Kernel Dive")
        self.assertEqual(res[0]["tags"], ["rust", "linux"])

    def test_fetch_bellwether_issues_sorts_by_plus1(self):
        fake_resp = {
            "items": [
                {
                    "repository_url": "https://api.github.com/repos/duckdb/duckdb",
                    "number": 42,
                    "title": "Support streaming Parquet writes to S3",
                    "html_url": "https://github.com/duckdb/duckdb/issues/42",
                    "reactions": {"+1": 85, "total_count": 90},
                    "comments": 31,
                    "created_at": "2026-10-01T00:00:00Z",
                    "updated_at": "2026-11-24T00:00:00Z",
                    "body": "We need **streaming** writes without buffering.",
                },
                {
                    "repository_url": "https://api.github.com/repos/duckdb/duckdb",
                    "number": 99,
                    "title": "Ignore PR",
                    "html_url": "https://github.com/duckdb/duckdb/pull/99",
                    "pull_request": {},
                    "reactions": {"+1": 200},
                    "comments": 50,
                },
            ]
        }
        with patch("urllib.request.urlopen", return_value=_mock_json_response(fake_resp)):
            issues = weekly_synthesis.fetch_bellwether_issues(
                ["duckdb/duckdb"], limit=5, today=date(2026, 11, 25)
            )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["repo"], "duckdb/duckdb")
        self.assertEqual(issues[0]["plus1"], 85)
        self.assertIn("streaming writes", issues[0]["snippet"])

    def test_detect_cross_platform_normalizes_urls(self):
        hn = [
            {
                "title": "SQLite WAL2",
                "url": "https://www.sqlite.org/wal2/",
                "hn_url": "https://news.ycombinator.com/item?id=1",
                "points": 500,
            }
        ]
        lob = [
            {
                "title": "SQLite WAL2 Mode",
                "url": "http://sqlite.org/wal2",
                "comments_url": "https://lobste.rs/s/xyz",
                "score": 60,
            }
        ]
        overlaps = weekly_synthesis.detect_cross_platform(hn, lob)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["hn_points"], 500)
        self.assertEqual(overlaps[0]["lobsters_score"], 60)


class TestWeeklySynthesisAndNotion(unittest.TestCase):
    def test_parse_render_and_notion_payload(self):
        raw = json.dumps(
            {
                "week_in_one_take": "Local-first tooling and agent sandboxing dominated serious engineering threads.",
                "persisted_vs_faded": [
                    {
                        "theme": "Deterministic agent CLIs",
                        "status": "persisted",
                        "detail": "Appeared across 4 HN threads and 2 bellwether repos.",
                    }
                ],
                "article_pitches": [
                    {
                        "working_title": "Why Coding Agents Are Converging on Plain CLIs Over MCP",
                        "core_thesis": "Engineers are ditching heavy protocol wrappers in favor of composable Unix tools.",
                        "evidence": [
                            {"label": "HN Thread", "url": "https://news.ycombinator.com/item?id=10"}
                        ],
                        "counterargument": "MCP fans argue typed schemas still beat brittle stdout parsing.",
                    }
                ],
                "validated_pain_points": [
                    {
                        "problem": "Local LLM tool calling breaks silently on quantized models.",
                        "proof": "ollama/ollama#1234 (+140 thumbs-up) & Ask HN thread",
                        "url": "https://github.com/ollama/ollama/issues/1234",
                        "build_angle": "Build a 1-command conformance test suite benchmarking tool-call accuracy per GGUF quant.",
                    }
                ],
                "cross_platform_highlights": [
                    {
                        "title": "SQLite WAL2",
                        "url": "https://sqlite.org/wal2",
                        "why_it_mattered": "Hit #1 on both HN and Lobste.rs.",
                    }
                ],
                "active_card_updates": [
                    {
                        "page_id": "page-active-1",
                        "relevance_summary": "A new HN benchmark directly supports your thesis on plain CLIs.",
                        "links": [
                            {"label": "HN Benchmark", "url": "https://news.ycombinator.com/item?id=99"}
                        ],
                    }
                ],
            }
        )
        summary = weekly_synthesis.parse_weekly_summary(raw)
        active_cards = [
            {
                "page_id": "page-active-1",
                "title": "CLI vs MCP Essay",
                "status": "Drafting",
                "card_type": "Article Pitch",
            }
        ]
        html = weekly_synthesis.render_weekly_html(
            summary,
            "2026-W48",
            "Sunday, 29 November 2026 (UTC)",
            {"hn_days": 7, "lobsters_count": 15, "issues_count": 12},
            active_cards=active_cards,
        )
        text = weekly_synthesis.render_weekly_text(
            summary, "2026-W48", "Sunday, 29 November 2026 (UTC)"
        )

        self.assertIn("Weekly Compounding Brief", html)
        self.assertIn("Follow-Up Evidence Added to Your Active Notion Drafts", html)
        self.assertIn("CLI vs MCP Essay", html)
        self.assertIn("PERSISTED", html)
        self.assertIn("Why Coding Agents Are Converging on Plain CLIs Over MCP", html)
        self.assertIn("Strongest counterargument:", html)
        self.assertIn("Validated Pain Points", html)
        self.assertIn("7 daily HN archives", html)

        self.assertIn("CONNECT-THE-DOTS ARTICLE PITCHES:", text)
        self.assertIn("VALIDATED PAIN POINTS & BUILD ANGLES:", text)

        # Test Kanban cards generation (1 brief + 1 pitch + 1 pain point = 3 cards)
        cards = weekly_synthesis.build_kanban_cards("2026-W48", summary)
        self.assertEqual(len(cards), 3)
        self.assertEqual([c["card_type"] for c in cards], ["Weekly Brief", "Article Pitch", "Pain Point"])

    def test_notion_two_way_discovery_query_sync_and_toggle_append(self):
        requests_made = []

        def fake_urlopen(req, timeout=30):
            body = json.loads(req.data.decode()) if req.data else None
            requests_made.append((req.get_method(), req.full_url, body))
            if req.full_url.endswith("/v1/databases/db-1"):
                return _mock_json_response(
                    {"id": "db-1", "data_sources": [{"id": "ds-1"}]}
                )
            if req.full_url.endswith("/v1/data_sources/ds-1"):
                return _mock_json_response(
                    {
                        "id": "ds-1",
                        "properties": {
                            "Title": {"type": "title"},
                            "Status": {"type": "status"},
                            "Type": {"type": "select"},
                            "Week": {"type": "rich_text"},
                            "URL": {"type": "url"},
                        },
                    }
                )
            if req.full_url.endswith("/v1/data_sources/ds-1/query"):
                return _mock_json_response(
                    {
                        "results": [
                            {
                                "id": "page-draft-1",
                                "properties": {
                                    "Title": {"type": "title", "title": [{"plain_text": "My Active Essay"}]},
                                    "Status": {"type": "status", "status": {"name": "Drafting"}},
                                    "Type": {"type": "select", "select": {"name": "Article Pitch"}},
                                    "Week": {"type": "rich_text", "rich_text": [{"plain_text": "2026-W47"}]},
                                },
                            },
                            {
                                "id": "page-dismissed-1",
                                "properties": {
                                    "Title": {"type": "title", "title": [{"plain_text": "Crypto Noise"}]},
                                    "Status": {"type": "status", "status": {"name": "Dismissed"}},
                                    "Type": {"type": "select", "select": {"name": "Article Pitch"}},
                                    "Week": {"type": "rich_text", "rich_text": [{"plain_text": "2026-W47"}]},
                                },
                            },
                            {
                                "id": "page-dup-1",
                                "properties": {
                                    "Title": {"type": "title", "title": [{"plain_text": "Weekly Brief — 2026-W48"}]},
                                    "Status": {"type": "status", "status": {"name": "Inbox"}},
                                    "Type": {"type": "select", "select": {"name": "Weekly Brief"}},
                                    "Week": {"type": "rich_text", "rich_text": [{"plain_text": "2026-W48"}]},
                                },
                            },
                        ]
                    }
                )
            return _mock_json_response({"id": "created-page"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client = weekly_synthesis.NotionKanbanClient("secret-key", "db-1")
            client.discover_schema()
            active, dismissed, existing = client.query_memory()

            self.assertEqual(client.title_prop, "Title")
            self.assertEqual(client.data_source_id, "ds-1")
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["page_id"], "page-draft-1")
            self.assertEqual(dismissed, ["Crypto Noise"])

            summary = {
                "week_in_one_take": "Great week.",
                "persisted_vs_faded": [],
                "article_pitches": [
                    {
                        "working_title": "Brand New Pitch",
                        "core_thesis": "Thesis text.",
                        "evidence": [{"label": "HN", "url": "https://news.ycombinator.com/item?id=1"}],
                        "counterargument": "Skeptic view.",
                    }
                ],
                "validated_pain_points": [],
                "cross_platform_highlights": [],
                "active_card_updates": [
                    {
                        "page_id": "page-draft-1",
                        "relevance_summary": "New supporting thread on HN.",
                        "links": [{"label": "Thread", "url": "https://news.ycombinator.com/item?id=2"}],
                    }
                ],
            }
            created, enriched = weekly_synthesis.sync_to_notion(
                client, "2026-W48", summary, active, existing
            )

        # Weekly Brief was already in existing -> skipped; Brand New Pitch -> created (1)
        self.assertEqual(created, 1)
        self.assertEqual(enriched, 1)
        # Verify PATCH /v1/blocks/page-draft-1/children appended a toggle block
        patch_calls = [r for r in requests_made if r[0] == "PATCH"]
        self.assertEqual(len(patch_calls), 1)
        self.assertTrue(patch_calls[0][1].endswith("/v1/blocks/page-draft-1/children"))
        self.assertEqual(patch_calls[0][2]["children"][0]["type"], "toggle")
        # Verify POST /v1/pages used discovered data_source_id and dynamic 'Title' column (not 'Name')
        page_posts = [r for r in requests_made if r[0] == "POST" and r[1].endswith("/v1/pages")]
        self.assertEqual(len(page_posts), 1)
        self.assertEqual(page_posts[0][2]["parent"]["data_source_id"], "ds-1")
        self.assertIn("Title", page_posts[0][2]["properties"])
        self.assertNotIn("Proof", page_posts[0][2]["properties"])  # omitted since not in schema!

    def test_extract_notion_id_and_create_kanban_database(self):
        self.assertEqual(
            weekly_synthesis.extract_notion_id(
                "https://www.notion.so/My-Page-1234567890abcdef1234567890abcdef?pvs=4"
            ),
            "1234567890abcdef1234567890abcdef",
        )
        captured = []

        def fake_urlopen(req, timeout=30):
            captured.append((req.full_url, json.loads(req.data.decode())))
            return _mock_json_response(
                {"id": "aaaa1111-2222-3333-4444-555566667777", "url": "https://notion.so/db"}
            )

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            db = weekly_synthesis.create_kanban_database(
                "secret", "https://notion.so/Home-1234567890abcdef1234567890abcdef"
            )
        self.assertEqual(db["id"], "aaaa1111-2222-3333-4444-555566667777")
        self.assertTrue(captured[0][0].endswith("/v1/databases"))
        props = captured[0][1]["initial_data_source"]["properties"]
        self.assertEqual(set(props.keys()), {"Name", "Status", "Type", "Week", "URL", "Proof"})


if __name__ == "__main__":
    unittest.main()
