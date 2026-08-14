"""
Core-C front-end ERROR handling — parser + semantic diagnostics.

WHY THIS FILE EXISTS
--------------------
``tests/test_compiler.py`` / ``test_compiler_expansion.py`` cover the happy
path and a handful of rejections; what was missing is a systematic sweep of
the compiler's REJECTION surface. Every case below asserts three things:

  1. the exception is ``sim.compiler.errors.CompilerError`` (never a bare
     ``IndexError`` / ``KeyError`` / ``AttributeError`` leaking out of the
     recursive-descent walk — that distinction is what lets ``/compile``
     return a clean 400 instead of a 500),
  2. the MESSAGE identifies the actual problem, and
  3. the LOCATION (``err.line``, and ``err.col`` where the raise site has one)
     points at the offending construct — sources are deliberately written
     multi-line so a wrong line number cannot pass.

Where the compiler ACCEPTS something a reader might expect it to reject, that
is pinned too (``test_accepted_by_design``) so a future tightening is a
deliberate, noticed change.

WHERE ERRORS COME FROM
  * lexer.py    — unsupported keyword, bad literal, bad char, bad comment
                  (covered in depth by tests/test_compiler_lexer.py)
  * parser.py   — structure, lvalues, array sizes (``_const_eval``)
  * codegen/base.py — name resolution, arity, break/continue, globals layout
  * codegen/<isa>.py — per-backend capability limits

Run from repo root:  python3 -m pytest tests/test_compiler_errors.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.compiler import compile_c
from sim.compiler.codegen import get_codegen
from sim.compiler.errors import CompilerError
from sim.compiler.parser import parse


def bad(src, isa="riscv"):
    """Compile ``src``; assert it raises CompilerError and return the error."""
    with pytest.raises(CompilerError) as exc:
        compile_c(src, isa)
    return exc.value


def wrap(*body_lines):
    """Put ``body_lines`` inside main() so line N of the body is source line
    N+1 — every location assertion below is written against that offset."""
    return "int main() {\n" + "\n".join(body_lines) + "\n}\n"


# ============================================================================
# 1. THE ERROR TYPE ITSELF
# ============================================================================
class TestCompilerErrorType:
    def test_renders_line_and_column_when_both_known(self):
        err = CompilerError("boom", 4, 7)
        assert str(err) == "line 4 col 7: boom"
        assert (err.message, err.line, err.col) == ("boom", 4, 7)

    def test_renders_line_only_when_column_is_unknown(self):
        assert str(CompilerError("boom", 4)) == "line 4: boom"

    def test_renders_bare_message_when_location_is_unknown(self):
        err = CompilerError("boom")
        assert str(err) == "boom"
        assert (err.line, err.col) == (None, None)

    def test_is_an_exception_so_callers_can_catch_it_broadly(self):
        assert issubclass(CompilerError, Exception)


# ============================================================================
# 2. NOTHING TO COMPILE
# ============================================================================
class TestEmptyAndMissingMain:
    @pytest.mark.parametrize("src", ["", "   ", "\n\n", "// just a comment\n",
                                     "/* only a block comment */"])
    def test_source_with_no_functions(self, src):
        err = bad(src)
        assert err.message == "program has no functions"
        assert (err.line, err.col) == (1, 1)

    def test_globals_only_still_needs_a_function(self):
        err = bad("int g = 1;\nint h = 2;\n")
        assert err.message == "program has no functions"

    def test_program_without_main(self):
        err = bad("int helper() {\n  return 1;\n}\n")
        assert "must define an 'int main()' function" in err.message

    def test_source_must_be_a_string(self):
        with pytest.raises(CompilerError) as exc:
            compile_c(123, "riscv")
        assert exc.value.message == "source must be a string"

    @pytest.mark.parametrize("isa", ["", "zzz", "mips", "riscv64", None, 7])
    def test_unknown_isa(self, isa):
        with pytest.raises(CompilerError) as exc:
            compile_c("int main(){return 0;}", isa)
        assert "unknown ISA" in exc.value.message

    def test_codegen_dispatch_rejects_an_unknown_isa(self):
        with pytest.raises(CompilerError) as exc:
            get_codegen("mips")
        assert "no code generator for ISA 'mips'" in exc.value.message


# ============================================================================
# 3. STRUCTURAL / PUNCTUATION ERRORS
# ============================================================================
class TestStructuralErrors:
    def test_missing_semicolon_after_a_declaration(self):
        err = bad(wrap("  int a = 1", "  return a;"))
        assert "expected OP ';'" in err.message
        assert "'return'" in err.message          # the token actually found
        assert (err.line, err.col) == (3, 3)

    def test_missing_semicolon_after_an_expression_statement(self):
        err = bad(wrap("  int a = 1;", "  a = a + 1", "  return a;"))
        assert "expected OP ';'" in err.message
        assert err.line == 4

    def test_unclosed_brace_reports_end_of_input(self):
        err = bad("int main() {\n  int a = 1;\n")
        assert "expected OP '}'" in err.message
        assert "end of input" in err.message

    def test_extra_closing_brace_at_top_level(self):
        err = bad("int main() {\n  return 0;\n}\n}\n")
        assert "expected function definition beginning with 'int' or 'void'" \
            in err.message
        assert err.line == 4

    def test_unclosed_paren_in_an_expression(self):
        err = bad(wrap("  return (1 + 2;"))
        assert "expected OP ')'" in err.message
        assert err.line == 2

    def test_unclosed_paren_in_an_if_condition(self):
        err = bad(wrap("  int a = 1;", "  if (a > 0 {", "    a = 2;", "  }",
                       "  return a;"))
        assert "expected OP ')'" in err.message
        assert err.line == 3

    def test_unclosed_bracket_in_an_index(self):
        err = bad(wrap("  int a[3];", "  return a[0;"))
        assert "expected OP ']'" in err.message
        assert err.line == 3

    def test_else_without_a_matching_if(self):
        err = bad("int main() {\n  int a = 1;\n  else {\n    a = 2;\n  }\n"
                  "  return a;\n}\n")
        assert err.message == "'else' without matching 'if'"
        assert (err.line, err.col) == (3, 3)

    def test_second_else_on_one_if(self):
        err = bad("int main() {\n  int a = 1;\n  if (a) { a = 2; }\n"
                  "  else { a = 3; }\n  else { a = 4; }\n  return a;\n}\n")
        assert err.message == "'else' without matching 'if'"
        assert err.line == 5

    def test_function_name_must_be_an_identifier(self):
        err = bad("int 5;\nint main(){return 0;}\n")
        assert "expected IDENT" in err.message
        assert err.line == 1

    def test_garbage_at_top_level(self):
        err = bad("foo bar;\nint main(){return 0;}\n")
        assert "expected function definition" in err.message
        assert (err.line, err.col) == (1, 1)

    def test_expression_truncated_at_end_of_input(self):
        err = bad("int main() {\n  return 1 +")
        assert "unexpected token end of input in expression" in err.message

    def test_stray_operator_where_a_primary_was_expected(self):
        err = bad(wrap("  return * ;"))
        assert "unexpected token" in err.message
        assert err.line == 2

    def test_empty_statement_is_rejected(self):
        """A bare ';' is NOT a null statement in Core-C — every statement that
        is not a declaration/keyword form must be a real expression."""
        err = bad(wrap("  ;", "  return 0;"))
        assert err.message == "unexpected token ';' in expression"
        assert (err.line, err.col) == (2, 3)


# ============================================================================
# 4. NAME RESOLUTION  (raised in codegen/base.py)
# ============================================================================
class TestNameResolutionErrors:
    def test_use_of_undeclared_variable(self):
        err = bad(wrap("  int a = 1;", "  a = a + 1;", "  return zzz;"))
        assert err.message == "use of undeclared variable 'zzz'"
        assert err.line == 4

    def test_assignment_to_undeclared_variable(self):
        err = bad(wrap("  int a = 1;", "  zzz = 5;", "  return a;"))
        assert err.message == "assignment to undeclared variable 'zzz'"
        assert err.line == 3

    def test_address_of_an_undeclared_variable(self):
        err = bad(wrap("  int a = 1;", "  int p = &zzz;", "  return a;"))
        assert err.message == "cannot take address of undeclared 'zzz'"
        assert err.line == 3

    def test_assignment_to_a_whole_array(self):
        err = bad(wrap("  int a[3];", "  a = 1;", "  return a[0];"))
        assert err.message == "cannot assign to array 'a'"
        assert err.line == 3

    def test_assignment_to_a_whole_global_array(self):
        err = bad("int t[3];\nint main() {\n  t = 1;\n  return 0;\n}\n")
        assert err.message == "cannot assign to array 't'"
        assert err.line == 3

    def test_redeclaration_of_a_global(self):
        err = bad("int g = 1;\nint g = 2;\nint main(){return g;}\n")
        assert err.message == "redeclaration of global 'g'"
        assert err.line == 2

    @pytest.mark.parametrize("first,second", [
        ("int a[3];", "int a[4];"),      # array then array
        ("int a;", "int a[3];"),         # scalar then array
    ])
    def test_local_redeclaration_involving_an_array(self, first, second):
        err = bad(wrap(f"  {first}", f"  {second}", "  return 0;"))
        assert err.message == "redeclaration of 'a'"


# ============================================================================
# 5. FUNCTION CALLS
# ============================================================================
class TestCallErrors:
    def test_call_to_an_undefined_function(self):
        err = bad("int main() {\n  int a = 1;\n  return foo(a);\n}\n")
        assert err.message == "call to undefined function 'foo'"
        assert err.line == 3

    def test_too_many_arguments(self):
        err = bad("int f(int a) {\n  return a;\n}\n"
                  "int main() {\n  return f(1, 2);\n}\n")
        assert err.message == "function 'f' expects 1 argument(s), got 2"
        assert err.line == 5

    def test_too_few_arguments(self):
        err = bad("int f(int a, int b) {\n  return a + b;\n}\n"
                  "int main() {\n  return f(1);\n}\n")
        assert err.message == "function 'f' expects 2 argument(s), got 1"
        assert err.line == 5

    def test_no_arguments_when_some_are_required(self):
        err = bad("int f(int a) {\n  return a;\n}\n"
                  "int main() {\n  return f();\n}\n")
        assert "expects 1 argument(s), got 0" in err.message

    def test_arguments_passed_to_a_zero_parameter_function(self):
        err = bad("int f() {\n  return 1;\n}\n"
                  "int main() {\n  return f(9);\n}\n")
        assert "expects 0 argument(s), got 1" in err.message

    @pytest.mark.parametrize("n_args", [0, 2, 3])
    def test_print_builtin_takes_exactly_one_argument(self, n_args):
        args = ", ".join(str(i) for i in range(n_args))
        err = bad(wrap("  int a = 1;", f"  print({args});", "  return a;"))
        assert err.message == "print() takes exactly one argument"
        assert err.line == 3

    @pytest.mark.parametrize("isa", ["riscv", "arm"])
    def test_more_than_eight_arguments(self, isa):
        params = ", ".join(f"int p{i}" for i in range(9))
        args = ", ".join(str(i) for i in range(9))
        err = bad(f"int f({params}) {{\n  return p0;\n}}\n"
                  f"int main() {{\n  return f({args});\n}}\n", isa)
        assert "passes 9 args" in err.message and "max 8" in err.message


# ============================================================================
# 6. ARRAY DECLARATIONS  (parser._const_eval)
# ============================================================================
class TestArrayDeclarationErrors:
    def test_non_constant_array_size_from_a_plain_local(self):
        err = bad(wrap("  int n;", "  n = 5;", "  int a[n];", "  return 0;"))
        assert "'n' is not a compile-time constant" in err.message
        assert "array sizes must be constant" in err.message
        assert err.line == 4

    def test_non_constant_array_size_from_a_parameter(self):
        err = bad("int f(int n) {\n  int a[n];\n  return a[0];\n}\n"
                  "int main(){return f(3);}\n")
        assert "is not a compile-time constant" in err.message
        assert err.line == 2

    def test_array_size_from_a_function_call_is_not_constant(self):
        err = bad("int three() {\n  return 3;\n}\n"
                  "int main() {\n  int a[three()];\n  return 0;\n}\n")
        assert err.message == "not a compile-time constant expression"
        assert err.line == 5

    @pytest.mark.parametrize("size_expr", ["0", "-3", "2-2", "1-9", "0*5"])
    def test_zero_or_negative_array_size(self, size_expr):
        err = bad(wrap(f"  int a[{size_expr}];", "  return 0;"))
        assert err.message == "array size must be positive"
        assert (err.line, err.col) == (2, 7)

    def test_array_with_neither_size_nor_initializer(self):
        err = bad(wrap("  int a[];", "  return 0;"))
        assert err.message == "array 'a' needs a size or an initializer"
        assert (err.line, err.col) == (2, 7)

    def test_empty_initializer_without_a_size(self):
        err = bad(wrap("  int a[] = {};", "  return 0;"))
        assert err.message == "empty initializer needs an array size"
        assert err.line == 2

    def test_too_many_initializers(self):
        err = bad(wrap("  int q = 1;", "  int a[2] = {1, 2, 3};", "  return q;"))
        assert err.message == "too many initializers (3) for 'a[2]'"
        assert (err.line, err.col) == (3, 7)

    def test_division_by_zero_in_a_constant_size_expression(self):
        err = bad(wrap("  int a[8 / 0];", "  return 0;"))
        assert err.message == "division by zero in constant expression"
        assert err.line == 2

    def test_modulo_by_zero_in_a_constant_size_expression(self):
        err = bad(wrap("  int a[8 % 0];", "  return 0;"))
        assert err.message == "division by zero in constant expression"

    @pytest.mark.parametrize("expr", ["!1", "-!0", "&x"])
    def test_operator_not_allowed_in_a_constant_expression(self, expr):
        err = bad(wrap("  int x = 1;", f"  int a[{expr}];", "  return 0;"))
        assert err.line == 3
        assert ("operator not allowed in a constant expression" in err.message
                or "not a compile-time constant" in err.message)

    def test_a_const_from_one_function_does_not_leak_into_another(self):
        """``const_env`` is snapshotted per function, so ``N`` declared in main
        is NOT visible as an array size in a sibling function."""
        err = bad("int main() {\n  const int N = 4;\n  int a[N];\n"
                  "  return a[0];\n}\n"
                  "int f() {\n  int b[N];\n  return b[0];\n}\n")
        assert "'N' is not a compile-time constant" in err.message
        assert err.line == 7


# ============================================================================
# 7. LVALUE / OPERAND ERRORS
# ============================================================================
class TestLvalueErrors:
    @pytest.mark.parametrize("stmt", [
        "1 = 2;", "(a + 1) = 2;", "f() = 2;", "-a = 2;",
    ])
    def test_invalid_plain_assignment_target(self, stmt):
        err = bad("int f() {\n  return 1;\n}\n"
                  "int main() {\n  int a = 1;\n  " + stmt + "\n  return a;\n}\n")
        assert err.message == "invalid assignment target"
        assert err.line == 6

    @pytest.mark.parametrize("op", ["+=", "-=", "*=", "/=", "%=",
                                    "&=", "|=", "^=", "<<=", ">>="])
    def test_invalid_compound_assignment_target(self, op):
        err = bad(wrap("  int a = 1;", f"  (a + 1) {op} 2;", "  return a;"))
        assert err.message == "invalid assignment target"
        assert err.line == 3

    @pytest.mark.parametrize("expr", ["++1", "--(a+1)", "1++", "(a+1)--"])
    def test_incdec_requires_an_lvalue(self, expr):
        err = bad(wrap("  int a = 1;", f"  {expr};", "  return a;"))
        assert err.message == "invalid assignment target"
        assert err.line == 3

    @pytest.mark.parametrize("expr", ["&(a + 1)", "&1", "&f()", "&&a"])
    def test_address_of_requires_an_lvalue(self, expr):
        err = bad("int f() {\n  return 1;\n}\n"
                  "int main() {\n  int a = 1;\n  int p = " + expr + ";\n"
                  "  return a;\n}\n")
        assert err.line == 6
        assert ("'&' requires an lvalue" in err.message
                or "invalid assignment target" in err.message
                or "unexpected token" in err.message)


# ============================================================================
# 8. OUT-OF-SCOPE C CONSTRUCTS
# ============================================================================
class TestUnsupportedConstructs:
    @pytest.mark.parametrize("kw", ["float", "double", "char", "long", "short",
                                    "unsigned", "signed", "static"])
    def test_unsupported_types_are_rejected_with_a_location(self, kw):
        err = bad(wrap("  int a = 1;", f"  {kw} b;", "  return a;"))
        assert f"unsupported type/keyword '{kw}'" in err.message
        assert (err.line, err.col) == (3, 3)

    @pytest.mark.parametrize("kw,src", [
        ("struct", "struct P { int x; };\nint main(){return 0;}\n"),
        ("union", "union U { int x; };\nint main(){return 0;}\n"),
        ("enum", "enum E { A };\nint main(){return 0;}\n"),
        ("typedef", "typedef int myint;\nint main(){return 0;}\n"),
    ])
    def test_aggregate_and_alias_declarations(self, kw, src):
        err = bad(src)
        assert f"unsupported type/keyword '{kw}'" in err.message
        assert err.line == 1

    @pytest.mark.parametrize("kw,stmt", [
        ("switch", "switch (a) { }"),
        ("do", "do { a = 1; } while (0);"),
        ("goto", "goto end;"),
        ("sizeof", "a = sizeof(a);"),
        ("malloc", "a = malloc(4);"),
        ("free", "free(a);"),
    ])
    def test_unsupported_statement_keywords(self, kw, stmt):
        err = bad(wrap("  int a = 1;", f"  {stmt}", "  return a;"))
        assert f"unsupported type/keyword '{kw}'" in err.message
        assert err.line == 3

    def test_pointer_variable_declaration(self):
        err = bad(wrap("  int a = 1;", "  int *p;", "  return a;"))
        assert "pointer variable declarations are not supported" in err.message
        assert (err.line, err.col) == (3, 7)

    def test_double_pointer_declaration(self):
        err = bad(wrap("  int **pp;", "  return 0;"))
        assert "pointer variable declarations are not supported" in err.message

    def test_string_literal(self):
        err = bad(wrap('  return "hi";'))
        assert "unexpected character" in err.message
        assert err.line == 2

    def test_character_literal(self):
        err = bad(wrap("  int c = 'a';", "  return c;"))
        assert "unexpected character" in err.message
        assert err.line == 2

    def test_float_literal(self):
        err = bad(wrap("  int a = 1;", "  a = a + 1.5;", "  return a;"))
        assert "unexpected character '.'" in err.message
        assert err.line == 3


# ============================================================================
# 9. BREAK / CONTINUE OUTSIDE A LOOP  (raised in codegen/base.py)
# ============================================================================
class TestLoopControlErrors:
    @pytest.mark.parametrize("kw", ["break", "continue"])
    def test_at_function_top_level(self, kw):
        err = bad(wrap("  int a = 1;", f"  {kw};", "  return a;"))
        assert err.message == f"'{kw}' outside a loop"
        assert err.line == 3

    @pytest.mark.parametrize("kw", ["break", "continue"])
    def test_inside_an_if_that_is_not_inside_a_loop(self, kw):
        err = bad(wrap("  int a = 1;", "  if (a) {", f"    {kw};", "  }",
                       "  return a;"))
        assert err.message == f"'{kw}' outside a loop"
        assert err.line == 4

    @pytest.mark.parametrize("kw", ["break", "continue"])
    def test_after_a_loop_has_closed(self, kw):
        err = bad(wrap("  int i;", "  int s = 0;",
                       "  for (i = 0; i < 3; i++) { s = s + i; }",
                       f"  {kw};", "  return s;"))
        assert err.message == f"'{kw}' outside a loop"
        assert err.line == 5

    @pytest.mark.parametrize("kw", ["break", "continue"])
    def test_in_a_callee_called_from_inside_a_loop(self, kw):
        """The loop stack is per-function, not per call chain."""
        err = bad(f"int helper() {{\n  {kw};\n  return 1;\n}}\n"
                  "int main() {\n  int i;\n  int s = 0;\n"
                  "  for (i = 0; i < 3; i++) { s = s + helper(); }\n"
                  "  return s;\n}\n")
        assert err.message == f"'{kw}' outside a loop"
        assert err.line == 2


# ============================================================================
# 10. RESOURCE LIMITS
# ============================================================================
class TestResourceLimitErrors:
    @pytest.mark.parametrize("isa,limit", [("riscv", 2032), ("arm", 2032),
                                           ("x86", 256)])
    def test_global_region_overflow(self, isa, limit):
        err = bad(f"int big[{limit}];\nint main(){{return 0;}}\n", isa)
        assert "exceeds the" in err.message and "global region" in err.message
        assert str(limit) in err.message

    @pytest.mark.parametrize("isa", ["riscv", "arm"])
    def test_single_frame_larger_than_the_software_stack_region(self, isa):
        err = bad("int main() {\n  int a[16000];\n  return a[0];\n}\n", isa)
        assert "exceeds the software stack region" in err.message
        assert "frame" in err.message

    def test_x86_frame_slot_offset_exceeds_disp8(self):
        err = bad("int main() {\n  int a[40];\n  a[0] = 1;\n"
                  "  return a[0];\n}\n", "x86")
        assert "exceeds the disp8 range" in err.message
        assert "124" in err.message


# ============================================================================
# 11. PER-BACKEND CAPABILITY LIMITS
# ============================================================================
class TestBackendCapabilityErrors:
    @pytest.mark.parametrize("isa,label", [("arm", "ARM"), ("x86", "x86")])
    @pytest.mark.parametrize("expr", ["a << 2", "a >> 2", "a <<= 1", "a >>= 1"])
    def test_shifts_are_rejected_on_arm_and_x86(self, isa, label, expr):
        err = bad(wrap("  int a = 8;", f"  a = {expr};", "  return a;"), isa)
        assert "shift operators (<< >>) are not supported" in err.message
        assert f"on the {label} backend" in err.message

    @pytest.mark.parametrize("expr", ["a << 2", "a >> 2"])
    def test_shifts_ARE_supported_on_riscv(self, expr):
        assert compile_c(wrap("  int a = 8;", f"  a = {expr};", "  return a;"),
                         "riscv").asm.strip()

    def test_x86_rejects_multiple_functions(self):
        err = bad("int f() {\n  return 1;\n}\n"
                  "int main() {\n  return 2;\n}\n", "x86")
        assert "supports only a single function" in err.message
        assert "found 2 functions" in err.message

    def test_x86_rejects_self_recursion(self):
        err = bad("int main() {\n  return main();\n}\n", "x86")
        assert "recursion is not supported" in err.message
        assert "'main' calls itself" in err.message
        assert err.line == 1

    def test_x86_rejects_function_parameters(self):
        err = bad("int main(int x) {\n  return x;\n}\n", "x86")
        assert "x86 backend functions take no parameters" in err.message

    def test_x86_call_error_is_reachable_only_via_direct_codegen(self):
        """``emit_call`` guards the datapath even if ``_detect_recursion``
        were relaxed; drive it directly since the earlier check normally
        fires first."""
        from sim.compiler import ast
        cg = get_codegen("x86")
        with pytest.raises(CompilerError) as exc:
            cg.emit_call(ast.Call("f", [], line=9),
                         ast.Function("f", [], ast.Block([], line=1), line=1))
        assert "function calls are not supported on the x86 backend" \
            in exc.value.message
        assert exc.value.line == 9


# ============================================================================
# 12. CODEGEN-LEVEL GUARDS reached by hand-built ASTs
# ============================================================================
class TestCodegenGuards:
    """A few ``CompilerError``s in codegen/base.py cannot be produced by any
    text the parser accepts (the parser rejects it first). They are the
    contract for anyone constructing an AST programmatically, so drive them
    directly rather than leaving the branches dead."""

    @staticmethod
    def _gen(stmt):
        from sim.compiler import ast
        cg = get_codegen("riscv")
        return cg.generate(ast.Program(
            [ast.Function("main", [], ast.Block([stmt], line=1), line=1)]))

    class _Alien:
        """A node type the walker has never heard of."""
        line = 3

        def to_dict(self):
            return {}

    def test_unsupported_statement_node(self):
        with pytest.raises(CompilerError) as exc:
            self._gen(self._Alien())
        assert "unsupported statement _Alien" in exc.value.message
        assert exc.value.line == 3

    def test_unsupported_expression_node(self):
        from sim.compiler import ast
        with pytest.raises(CompilerError) as exc:
            self._gen(ast.ExprStmt(self._Alien(), line=4))
        assert "unsupported expression _Alien" in exc.value.message

    def test_unsupported_unary_operator(self):
        from sim.compiler import ast
        with pytest.raises(CompilerError) as exc:
            self._gen(ast.Return(ast.Unary("@", ast.Number(1, line=5), line=5),
                                 line=5))
        assert exc.value.message == "unsupported unary operator '@'"
        assert exc.value.line == 5

    def test_unsupported_binary_operator(self):
        from sim.compiler import ast
        with pytest.raises(CompilerError) as exc:
            self._gen(ast.Return(ast.Binary("@@", ast.Number(1, line=6),
                                            ast.Number(2, line=6), line=6),
                                 line=6))
        assert exc.value.message == "unsupported binary operator '@@'"

    def test_address_of_a_non_lvalue_node(self):
        from sim.compiler import ast
        with pytest.raises(CompilerError) as exc:
            self._gen(ast.Return(ast.Unary("&", ast.Number(1, line=7), line=7),
                                 line=7))
        assert exc.value.message == "'&' requires a variable or array element"

    def test_invalid_assignment_target_node(self):
        from sim.compiler import ast
        with pytest.raises(CompilerError) as exc:
            self._gen(ast.ExprStmt(ast.Assign(ast.Number(1, line=8),
                                              ast.Number(2, line=8), line=8),
                                   line=8))
        assert exc.value.message == "invalid assignment target"
        assert exc.value.line == 8

    def test_every_isa_primitive_on_the_abstract_base_raises(self):
        """CodeGenBase declares 32 ``emit_*`` primitives that subclasses MUST
        implement; the base versions must fail loudly, not silently no-op."""
        import inspect

        from sim.compiler.codegen.base import CodeGenBase
        base = CodeGenBase()
        names = [n for n in dir(CodeGenBase) if n.startswith("emit_")]
        assert len(names) == 32
        for name in names:
            fn = getattr(base, name)
            args = [0] * len(inspect.signature(fn).parameters)
            with pytest.raises(NotImplementedError):
                fn(*args)


# ============================================================================
# 13. ACCEPTED BY DESIGN — pins on what is deliberately NOT an error
# ============================================================================
class TestAcceptedByDesign:
    """Constructs a C programmer might expect to be diagnosed, which Core-C
    accepts on purpose. Pinned so a future tightening is deliberate."""

    @pytest.mark.parametrize("src,note", [
        ("int main() {\n  return;\n}\n",
         "bare 'return' in an int function returns 0 by convention"),
        ("void main() {\n  int a = 1;\n}\n",
         "'void' return type is accepted (and falls through to return 0)"),
        ("const int main() {\n  return 3;\n}\n",
         "a const-qualified return type is tolerated"),
        ("int f(void) {\n  return 3;\n}\nint main(){return f();}\n",
         "'void' as the sole parameter marker"),
        ("int f(int a[3]) {\n  return a[0];\n}\n"
         "int main(){int x[3]={9,1,2}; return f(x);}\n",
         "a SIZED array parameter decays like int a[]"),
        ("int main() {\n  int a = 1;\n  int a = 2;\n  return a;\n}\n",
         "redeclaring a SCALAR local reuses the same frame slot"),
        ("int main() {\n  int a[3];\n  int a;\n  return a;\n}\n",
         "array-then-scalar redeclaration is silently ignored"),
        ("int main() {\n  return +5;\n}\n", "unary plus is the identity"),
        ("int main() {\n  return ;\n}\n", "'return ;' with a space"),
        ("int main() {\n  int i;\n  int s = 0;\n"
         "  for (int i2 = 0; i2 < 3; i2++) { s = s + i2; }\n  return s;\n}\n",
         "a declaration in the for-init clause"),
        ("int g = 1;\nint main() {\n  int g = 9;\n  return g;\n}\n",
         "a local shadows a same-named global"),
    ])
    def test_compiles_without_error(self, src, note):
        assert compile_c(src, "riscv").asm.strip(), note

    def test_parse_entry_point_returns_a_program_node(self):
        prog = parse("int g = 1;\nint main() {\n  return g;\n}\n")
        assert [f.name for f in prog.functions] == ["main"]
        assert [g.name for g in prog.globals] == ["g"]
