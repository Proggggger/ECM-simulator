"""
UNIT tests for the Core-C lexer (``sim/compiler/lexer.py``).

WHY THIS FILE EXISTS
--------------------
The lexer had ZERO direct tests: it was only ever exercised transitively,
through ``compile_c`` on well-formed programs. That leaves its whole error
surface (unterminated block comment, malformed hex/binary literal, ``12abc``,
unexpected character) and its **line/column bookkeeping** unverified — and
column accuracy is load-bearing: every located ``CompilerError`` and the
``/lab`` page's source-map ribbon are built from ``Token.line``/``Token.col``.

Here ``tokenize()`` is driven directly on tiny inputs and the full
``(kind, value, line, col)`` tuple is asserted, so a shifted column or a
silently-split literal is a test failure rather than a slightly-wrong UI.

Token kinds (per the module docstring): KEYWORD / IDENT / NUMBER / OP / EOF.
There is exactly ONE EOF token and it is always last.

Run from repo root:  python3 -m pytest tests/test_compiler_lexer.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.compiler.errors import CompilerError
from sim.compiler.lexer import (
    KEYWORDS,
    Lexer,
    Token,
    _OPS_2,
    _OPS_3,
    _SINGLE_OPS,
    _UNSUPPORTED_KEYWORDS,
    tokenize,
)


def toks(src):
    """(kind, value) for every token EXCEPT the trailing EOF."""
    return [(t.kind, t.value) for t in tokenize(src)[:-1]]


def full(src):
    """(kind, value, line, col) for every token EXCEPT the trailing EOF."""
    return [(t.kind, t.value, t.line, t.col) for t in tokenize(src)[:-1]]


# ============================================================================
# 1. TOKEN KINDS
# ============================================================================
class TestTokenKinds:
    def test_eof_is_always_present_and_last(self):
        for src in ("", "   ", "int", "int main(){}"):
            stream = tokenize(src)
            assert stream[-1].kind == "EOF"
            assert stream[-1].value == ""
            assert [t.kind for t in stream].count("EOF") == 1

    def test_empty_source_yields_only_eof(self):
        stream = tokenize("")
        assert len(stream) == 1
        assert (stream[0].kind, stream[0].line, stream[0].col) == ("EOF", 1, 1)

    @pytest.mark.parametrize("word", sorted(KEYWORDS))
    def test_every_keyword_lexes_as_keyword(self, word):
        assert toks(word) == [("KEYWORD", word)]

    def test_keyword_set_is_exactly_the_documented_ten(self):
        assert KEYWORDS == {"int", "if", "else", "while", "for", "return",
                            "void", "break", "continue", "const"}

    @pytest.mark.parametrize("name", ["x", "foo", "_", "_foo1", "a1b2",
                                      "intx", "iff", "returned", "MAIN"])
    def test_identifiers_including_keyword_prefixes(self, name):
        """A keyword is only a keyword when it is the WHOLE word."""
        assert toks(name) == [("IDENT", name)]

    def test_token_to_dict_round_trips_all_four_fields(self):
        assert Token("OP", "==", 3, 7).to_dict() == {
            "kind": "OP", "value": "==", "line": 3, "col": 7}

    def test_tokenize_wrapper_matches_lexer_class(self):
        src = "int a = 1;"
        assert [t.to_dict() for t in tokenize(src)] == \
               [t.to_dict() for t in Lexer(src).tokenize()]


# ============================================================================
# 2. INTEGER LITERALS  (the forms the lexer ACTUALLY supports)
# ============================================================================
class TestNumberLiterals:
    """Core-C has decimal, 0x/0X hex and 0b/0B binary. There are NO char
    literals, no floats, no octal-by-leading-zero and no suffixes — the
    NUMBER token's value is a plain Python int."""

    @pytest.mark.parametrize("src,value", [
        ("0", 0), ("7", 7), ("42", 42), ("007", 7), ("2147483647", 2147483647),
        ("9999999999999", 9999999999999),        # no width clamp in the lexer
    ])
    def test_decimal(self, src, value):
        assert toks(src) == [("NUMBER", value)]

    @pytest.mark.parametrize("src,value", [
        ("0x0", 0), ("0xff", 255), ("0xFF", 255), ("0X1f", 31),
        ("0x1F", 31), ("0xdeadBEEF", 0xDEADBEEF),
    ])
    def test_hex_both_cases(self, src, value):
        assert toks(src) == [("NUMBER", value)]

    @pytest.mark.parametrize("src,value", [
        ("0b0", 0), ("0b1010", 10), ("0B11", 3), ("0b11111111", 255),
    ])
    def test_binary_both_cases(self, src, value):
        assert toks(src) == [("NUMBER", value)]

    def test_hex_stops_at_first_non_hex_digit(self):
        # 0x1Fg -> NUMBER(31) then IDENT('g'); the parser rejects the pair.
        assert full("0x1Fg") == [("NUMBER", 31, 1, 1), ("IDENT", "g", 1, 5)]

    def test_binary_stops_at_first_non_binary_digit(self):
        assert full("0b101x") == [("NUMBER", 5, 1, 1), ("IDENT", "x", 1, 6)]

    def test_no_char_literal_support(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("'a'")
        assert "unexpected character" in exc.value.message
        assert "'" in exc.value.message

    def test_no_string_literal_support(self):
        with pytest.raises(CompilerError) as exc:
            tokenize('"hi"')
        assert "unexpected character" in exc.value.message

    def test_no_float_literal_support(self):
        # '1.5' lexes '1' then chokes on '.' at column 2.
        with pytest.raises(CompilerError) as exc:
            tokenize("1.5")
        assert (exc.value.line, exc.value.col) == (1, 2)


# ============================================================================
# 3. OPERATORS — longest match first (3-char, 2-char, 1-char)
# ============================================================================
class TestOperators:
    @pytest.mark.parametrize("op", _OPS_3)
    def test_three_char_ops(self, op):
        assert toks(op) == [("OP", op)]

    def test_three_char_op_set_is_exactly_the_shift_assigns(self):
        assert set(_OPS_3) == {"<<=", ">>="}

    @pytest.mark.parametrize("op", _OPS_2)
    def test_two_char_ops(self, op):
        assert toks(op) == [("OP", op)]

    def test_two_char_op_set_is_complete(self):
        assert set(_OPS_2) == {
            "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "++", "--",
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        }

    @pytest.mark.parametrize("op", sorted(_SINGLE_OPS))
    def test_single_char_ops(self, op):
        assert toks(op) == [("OP", op)]

    def test_longest_match_wins_three_over_two_over_one(self):
        assert toks("<<=") == [("OP", "<<=")]        # not '<<' + '='
        assert toks("<<") == [("OP", "<<")]          # not '<' + '<'
        assert toks("<") == [("OP", "<")]
        assert toks(">>=") == [("OP", ">>=")]
        assert toks(">>") == [("OP", ">>")]

    def test_adjacent_operators_split_greedily(self):
        assert toks("a<<=1") == [("IDENT", "a"), ("OP", "<<="), ("NUMBER", 1)]
        assert toks("a<=b") == [("IDENT", "a"), ("OP", "<="), ("IDENT", "b")]
        assert toks("a<-b") == [("IDENT", "a"), ("OP", "<"), ("OP", "-"),
                                ("IDENT", "b")]
        assert toks("!x") == [("OP", "!"), ("IDENT", "x")]
        assert toks("!=") == [("OP", "!=")]

    def test_increment_vs_two_pluses(self):
        assert toks("a++") == [("IDENT", "a"), ("OP", "++")]
        assert toks("a+ +b") == [("IDENT", "a"), ("OP", "+"), ("OP", "+"),
                                 ("IDENT", "b")]

    def test_ternary_punctuation(self):
        assert toks("a?b:c") == [("IDENT", "a"), ("OP", "?"), ("IDENT", "b"),
                                 ("OP", ":"), ("IDENT", "c")]

    def test_brackets_braces_parens_comma_semicolon(self):
        assert toks("a[0]{},;()") == [
            ("IDENT", "a"), ("OP", "["), ("NUMBER", 0), ("OP", "]"),
            ("OP", "{"), ("OP", "}"), ("OP", ","), ("OP", ";"),
            ("OP", "("), ("OP", ")"),
        ]


# ============================================================================
# 4. COMMENTS AND WHITESPACE
# ============================================================================
class TestCommentsAndWhitespace:
    def test_line_comment_is_skipped_to_end_of_line(self):
        assert full("a // b c\nd") == [("IDENT", "a", 1, 1),
                                       ("IDENT", "d", 2, 1)]

    def test_line_comment_at_eof_without_newline(self):
        assert toks("a //trailing") == [("IDENT", "a")]

    def test_bare_line_comment_only(self):
        assert tokenize("// nothing here")[0].kind == "EOF"

    def test_block_comment_is_skipped(self):
        assert full("a /*x*/ b") == [("IDENT", "a", 1, 1),
                                     ("IDENT", "b", 1, 9)]

    def test_empty_block_comment(self):
        assert toks("/**/x") == [("IDENT", "x")]

    def test_multiple_block_comments_in_a_row(self):
        assert toks("a /* c1 */ /* c2 */ b") == [("IDENT", "a"), ("IDENT", "b")]

    def test_block_comment_spanning_lines_advances_the_line_counter(self):
        assert full("a\n/* one\n   two */ b") == [("IDENT", "a", 1, 1),
                                                  ("IDENT", "b", 3, 11)]

    def test_block_comments_do_not_nest(self):
        """'/* a /* b */ c' closes at the FIRST '*/', leaving 'c' visible."""
        assert full("/* a /* b */ c") == [("IDENT", "c", 1, 14)]

    def test_lone_slash_is_the_division_operator(self):
        assert toks("a/b") == [("IDENT", "a"), ("OP", "/"), ("IDENT", "b")]
        assert toks("/") == [("OP", "/")]

    @pytest.mark.parametrize("ws", [" ", "\t", "\r", "\n", "  \t \r\n "])
    def test_all_whitespace_forms_are_skipped(self, ws):
        assert toks(f"a{ws}b") == [("IDENT", "a"), ("IDENT", "b")]

    def test_whitespace_only_source_yields_only_eof(self):
        assert len(tokenize("  \n\t\r\n  ")) == 1


# ============================================================================
# 5. LINE / COLUMN TRACKING  (the /lab source map depends on this)
# ============================================================================
class TestLineAndColumnTracking:
    def test_columns_are_one_based_on_the_first_line(self):
        assert full("int a") == [("KEYWORD", "int", 1, 1), ("IDENT", "a", 1, 5)]

    def test_newline_resets_the_column_and_bumps_the_line(self):
        assert full("a\nbb\nccc") == [("IDENT", "a", 1, 1),
                                      ("IDENT", "bb", 2, 1),
                                      ("IDENT", "ccc", 3, 1)]

    def test_blank_lines_are_counted(self):
        assert full("a\n\n\nb") == [("IDENT", "a", 1, 1), ("IDENT", "b", 4, 1)]

    def test_tab_counts_as_one_column(self):
        """No tab expansion — a tab advances the column by exactly 1."""
        assert full("a\tb") == [("IDENT", "a", 1, 1), ("IDENT", "b", 1, 3)]

    def test_carriage_return_counts_as_one_column_before_the_newline(self):
        assert full("a\r\nb") == [("IDENT", "a", 1, 1), ("IDENT", "b", 2, 1)]

    def test_token_column_is_the_first_character_of_the_token(self):
        # '<<=' starts at col 2, the number after it at col 5.
        assert full("x<<=1") == [("IDENT", "x", 1, 1), ("OP", "<<=", 1, 2),
                                 ("NUMBER", 1, 1, 5)]

    def test_eof_position_after_a_trailing_newline(self):
        eof = tokenize("int\n")[-1]
        assert (eof.line, eof.col) == (2, 1)

    def test_eof_position_without_a_trailing_newline(self):
        eof = tokenize("int")[-1]
        assert (eof.line, eof.col) == (1, 4)

    def test_realistic_multiline_program_positions(self):
        src = "int main() {\n    int a = 1;\n    return a;\n}\n"
        by_line = {}
        for t in tokenize(src)[:-1]:
            by_line.setdefault(t.line, []).append((t.value, t.col))
        assert by_line[1] == [("int", 1), ("main", 5), ("(", 9), (")", 10),
                              ("{", 12)]
        assert by_line[2] == [("int", 5), ("a", 9), ("=", 11), (1, 13),
                              (";", 14)]
        assert by_line[3] == [("return", 5), ("a", 12), (";", 13)]
        assert by_line[4] == [("}", 1)]


# ============================================================================
# 6. LEXER ERRORS — every raise in lexer.py, with its LOCATION
# ============================================================================
class TestLexerErrors:
    def test_unexpected_character(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("a @ b")
        err = exc.value
        assert err.message == "unexpected character '@'"
        assert (err.line, err.col) == (1, 3)
        assert str(err) == "line 1 col 3: unexpected character '@'"

    @pytest.mark.parametrize("ch", ["@", "$", "#", "\\", "`", '"', "'", "."])
    def test_every_unsupported_punctuation_character_is_rejected(self, ch):
        with pytest.raises(CompilerError) as exc:
            tokenize(f"a {ch}")
        assert "unexpected character" in exc.value.message
        assert exc.value.col == 3

    def test_unexpected_character_on_a_later_line_reports_that_line(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("int main() {\n  int a = 1;\n  a = a @ 2;\n}\n")
        assert (exc.value.line, exc.value.col) == (3, 9)

    def test_unterminated_block_comment_points_at_the_OPENING_slash_star(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("int main() {\n/* open\n  return 0;\n}\n")
        err = exc.value
        assert err.message == "unterminated block comment"
        assert (err.line, err.col) == (2, 1)

    def test_unterminated_block_comment_with_a_dangling_star(self):
        # '/*/' — the '/' is consumed as part of the opener, never closes.
        with pytest.raises(CompilerError) as exc:
            tokenize("/*/")
        assert exc.value.message == "unterminated block comment"
        assert (exc.value.line, exc.value.col) == (1, 1)

    def test_malformed_hex_literal(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("return 0x;")
        assert exc.value.message == "malformed hex literal"
        assert (exc.value.line, exc.value.col) == (1, 8)

    @pytest.mark.parametrize("src", ["0x", "0X", "0xZ", "0x+1"])
    def test_hex_with_no_digits_is_malformed(self, src):
        with pytest.raises(CompilerError) as exc:
            tokenize(src)
        assert exc.value.message == "malformed hex literal"

    def test_malformed_binary_literal(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("return 0b;")
        assert exc.value.message == "malformed binary literal"
        assert (exc.value.line, exc.value.col) == (1, 8)

    @pytest.mark.parametrize("src", ["0b", "0B", "0b2", "0b,"])
    def test_binary_with_no_digits_is_malformed(self, src):
        with pytest.raises(CompilerError) as exc:
            tokenize(src)
        assert exc.value.message == "malformed binary literal"

    @pytest.mark.parametrize("src", ["12abc", "3x", "0_9", "1foo"])
    def test_digit_glued_to_an_identifier_is_rejected_not_split(self, src):
        with pytest.raises(CompilerError) as exc:
            tokenize(src)
        assert exc.value.message == "invalid number literal"
        assert (exc.value.line, exc.value.col) == (1, 1)

    def test_invalid_number_literal_reports_the_literal_start_column(self):
        with pytest.raises(CompilerError) as exc:
            tokenize("x = 12abc;")
        assert (exc.value.line, exc.value.col) == (1, 5)


# ============================================================================
# 7. RESERVED-BUT-UNSUPPORTED KEYWORDS  (flagged at LEX time, on purpose)
# ============================================================================
class TestUnsupportedKeywords:
    """Core-C v1 is int-only. Reserved C words that name an unsupported
    feature are rejected by the LEXER so the diagnostic carries a precise
    location instead of surfacing as a confusing downstream parse failure."""

    def test_unsupported_set_is_exactly_the_documented_nineteen(self):
        assert _UNSUPPORTED_KEYWORDS == {
            "char", "float", "double", "struct", "union", "enum", "long",
            "short", "unsigned", "signed", "typedef", "static", "switch",
            "case", "do", "goto", "sizeof", "malloc", "free",
        }

    @pytest.mark.parametrize("word", sorted(_UNSUPPORTED_KEYWORDS))
    def test_each_unsupported_keyword_raises_a_located_error(self, word):
        with pytest.raises(CompilerError) as exc:
            tokenize(f"int main() {{\n  {word} x;\n}}\n")
        err = exc.value
        assert f"unsupported type/keyword '{word}'" in err.message
        assert "only 'int'" in err.message
        assert (err.line, err.col) == (2, 3)

    def test_unsupported_keyword_set_is_disjoint_from_the_supported_set(self):
        assert not (_UNSUPPORTED_KEYWORDS & KEYWORDS)

    def test_an_identifier_merely_containing_a_reserved_word_is_fine(self):
        assert toks("charlie floater structure") == [
            ("IDENT", "charlie"), ("IDENT", "floater"), ("IDENT", "structure")]
