from ai_pm_contracts import QueryRewrite
from retrieval_build.query import compile_fts_match


def test_query_compiler_quotes_values_and_never_accepts_raw_fts() -> None:
    rewrite = QueryRewrite(
        original_query="ignored",
        semantic_query_en="Steve Jobs product focus",
        lexical_terms_en=("focus", 'x" OR body:evil'),
        lexical_phrases_en=("Steve Jobs",),
    )
    assert compile_fts_match(rewrite) == '"Steve Jobs" OR "focus" OR "x"" OR body:evil"'


def test_query_compiler_has_safe_english_fallback() -> None:
    rewrite = QueryRewrite(
        original_query="What did Airbnb say about activation?",
        semantic_query_en="Airbnb activation",
    )
    assert compile_fts_match(rewrite) == '"Airbnb" OR "activation"'


def test_query_compiler_falls_back_to_rewritten_english_for_chinese_questions() -> None:
    rewrite = QueryRewrite(original_query="如何增长？", semantic_query_en="how to grow")
    assert compile_fts_match(rewrite) == '"grow"'
