"""Tests for the Rust bracket balance checker.

The checker is a safety net used when no Rust toolchain is available, so it
has to be trustworthy in both directions: it must not cry wolf on valid Rust
(which would train people to ignore it), and it must actually catch a real
imbalance (which is the only reason it exists).

The interesting cases are all about literals, because a naive brace counter is
wrong on ordinary Rust code -- braces inside strings, char literals and
comments must not count, and a lifetime tick must not be mistaken for the
start of a char literal.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from rust_bracket_check import check, strip_literals_and_comments  # noqa: E402


class StripTests(unittest.TestCase):
    def test_line_comment_contents_are_removed(self):
        cleaned = strip_literals_and_comments("// }}}}\nlet x = 1;\n")
        self.assertNotIn("}", cleaned)

    def test_block_comments_nest(self):
        # Rust allows nested block comments; C does not. If nesting were
        # mishandled the inner close would end the comment early and the
        # trailing brace would leak through.
        cleaned = strip_literals_and_comments("/* /* }} */ */")
        self.assertNotIn("}", cleaned)

    def test_raw_strings_with_hashes(self):
        cleaned = strip_literals_and_comments('let s = r#"{{{ unbalanced "#;')
        self.assertNotIn("{", cleaned)

    def test_escaped_quote_does_not_end_string(self):
        cleaned = strip_literals_and_comments(r'let s = "a\"} still in string";')
        self.assertNotIn("}", cleaned)

    def test_newlines_are_preserved_for_line_numbers(self):
        src = 'a\n/* x\ny */\nb\n'
        self.assertEqual(src.count("\n"), strip_literals_and_comments(src).count("\n"))


class CheckTests(unittest.TestCase):
    def _write(self, body: str) -> Path:
        import tempfile

        fd = tempfile.NamedTemporaryFile(
            "w", suffix=".rs", delete=False, encoding="utf-8"
        )
        fd.write(body)
        fd.close()
        return Path(fd.name)

    def test_balanced_file_reports_nothing(self):
        p = self._write("fn a() -> Vec<u8> { vec![1, 2, 3] }\n")
        self.assertEqual(check(p), [])

    def test_lifetime_is_not_a_char_literal(self):
        # `'a` and `'files:` are lifetimes/labels. If either were treated as
        # an unterminated char literal the rest of the file would be eaten
        # and the braces would appear unbalanced.
        p = self._write(
            "struct S<'a> { r: &'a str }\n"
            "fn f() { 'outer: for _ in 0..2 { break 'outer; } }\n"
        )
        self.assertEqual(check(p), [])

    def test_char_literal_brace_is_ignored(self):
        p = self._write("fn a() { let c = '}'; let d = '\\''; }\n")
        self.assertEqual(check(p), [])

    def test_unclosed_brace_is_caught(self):
        p = self._write("fn a() {\n    let x = 1;\n")
        problems = check(p)
        self.assertTrue(problems)
        self.assertIn("unclosed", problems[0])

    def test_mismatched_pair_is_caught(self):
        p = self._write("fn a() { let v = vec![1, 2); }\n")
        problems = check(p)
        self.assertTrue(problems)
        self.assertTrue(any("closes" in m for m in problems))

    def test_unexpected_closer_is_caught(self):
        p = self._write("fn a() { }\n}\n")
        problems = check(p)
        self.assertTrue(problems)
        self.assertIn("unexpected", problems[0])


class RepositoryTests(unittest.TestCase):
    def test_every_tracked_rust_file_balances(self):
        """The repo's own Rust must pass, or the tool is useless in practice."""
        from rust_bracket_check import tracked_rust_files

        files = tracked_rust_files()
        self.assertTrue(files, "expected to find tracked .rs files")
        for path in files:
            with self.subTest(path=path.name):
                self.assertEqual(check(path), [])


if __name__ == "__main__":
    unittest.main()
