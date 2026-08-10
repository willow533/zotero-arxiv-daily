"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest
from types import SimpleNamespace

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


def test_tldr_prompt_preserves_english_academic_terms():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这篇论文提出了一种 Transformer 方法。"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper(title="Attention Is All You Need")

    result = paper.generate_tldr(client, llm_params)

    messages = captured["messages"]
    assert result == "这篇论文提出了一种 Transformer 方法。"
    assert "Preserve the paper title" in messages[1]["content"]
    assert "Do not translate the paper title" in messages[1]["content"]
    assert "paper titles" in messages[0]["content"]


def test_chinese_tldr_retries_english_response():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="This paper proposes a Transformer method for robot manipulation."
                        ),
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这篇论文提出了用于 robot manipulation 的 Transformer 方法。"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == "这篇论文提出了用于 robot manipulation 的 Transformer 方法。"
    assert len(calls) == 2
    assert "Rewrite it now as exactly one concise Simplified Chinese sentence" in calls[1]["messages"][1]["content"]


def test_chinese_tldr_does_not_fall_back_to_english_abstract_on_error():
    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    llm_params = {
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper()

    result = paper.generate_tldr(broken_client, llm_params)

    assert result == "TL;DR 生成失败，请打开 PDF 或查看论文原文摘要。"
    assert result != paper.abstract


def test_chinese_tldr_returns_failure_message_after_bad_retry():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="This is still an English summary."),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == "TL;DR 生成失败，请打开 PDF 或查看论文原文摘要。"
    assert len(calls) == 2


def test_tldr_retries_siliconflow_with_qwen_when_gpt_model_fails():
    models = []

    def create(**kwargs):
        model = kwargs.get("model")
        models.append(model)
        if model == "gpt-4o-mini":
            raise RuntimeError("model not found")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这篇论文提出了一个 concise Chinese TLDR。"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "api": {"base_url": "https://api.siliconflow.cn/v1"},
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == "这篇论文提出了一个 concise Chinese TLDR。"
    assert models == ["gpt-4o-mini", "Qwen/Qwen3-8B"]


def test_tldr_caps_oversized_max_tokens():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="这篇论文给出了一句中文总结。"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "language": "Chinese with key academic terms and paper titles kept in English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }
    paper = make_sample_paper()

    paper.generate_tldr(client, llm_params)

    assert captured["max_tokens"] == 512


def test_tldr_moves_enable_thinking_into_extra_body():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A concise English summary."),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "language": "English",
        "generation_kwargs": {
            "model": "Qwen/Qwen3-32B",
            "max_tokens": 16384,
            "enable_thinking": False,
            "temperature": 0.2,
        },
    }
    paper = make_sample_paper()

    result = paper.generate_tldr(client, llm_params)

    assert result == "A concise English summary."
    assert "enable_thinking" not in captured
    assert captured["extra_body"]["enable_thinking"] is False
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 512


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


def test_affiliations_moves_enable_thinking_into_extra_body():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='["TsingHua University"]'),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    llm_params = {
        "generation_kwargs": {
            "model": "Qwen/Qwen3-32B",
            "max_tokens": 16384,
            "enable_thinking": False,
        },
    }
    paper = make_sample_paper()

    result = paper.generate_affiliations(client, llm_params)

    assert result == ["TsingHua University"]
    assert "enable_thinking" not in captured
    assert captured["extra_body"]["enable_thinking"] is False
    assert captured["max_tokens"] == 1024
