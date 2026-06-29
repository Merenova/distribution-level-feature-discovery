import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_sample_branches_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "2_branch_sampling"
        / "sample_branches.py"
    )
    spec = importlib.util.spec_from_file_location("sample_branches", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeGemmaTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.calls.append(
            {
                "text": text,
                "return_tensors": return_tensors,
                "add_special_tokens": add_special_tokens,
            }
        )
        if add_special_tokens:
            token_ids = [2, 2, 105, 2364, 107]
        else:
            token_ids = [2, 105, 2364, 107]
        return SimpleNamespace(input_ids=torch.tensor([token_ids]))


class FakeNoBosTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return SimpleNamespace(input_ids=torch.tensor([[105, 2364, 107]]))


class FakeLLM:
    def __init__(self):
        self.prompts = None

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts = prompts
        output = SimpleNamespace(
            text="continuation",
            token_ids=[999],
            cumulative_logprob=-0.5,
            logprobs=None,
        )
        return [SimpleNamespace(outputs=[output])]


class FakeScoringLLM:
    def __init__(self):
        self.prompts = None

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.prompts = prompts
        prompt_logprobs = [
            None,
            {105: SimpleNamespace(logprob=-0.1)},
            {300: SimpleNamespace(logprob=-1.0)},
            {400: SimpleNamespace(logprob=-2.0)},
        ]
        return [SimpleNamespace(prompt_logprobs=prompt_logprobs)]


class FakeSamplingConfig:
    temperature = 1.0
    nucleus_p = 0.95
    max_tokens = 8
    batch_size = 1
    stop_tokens = None


class FakeLogger:
    def info(self, *args, **kwargs):
        pass


def test_gemma_chat_template_prefix_does_not_get_double_bos():
    module = load_sample_branches_module()
    tokenizer = FakeGemmaTokenizer()

    token_ids = module.build_prefix_tokens_with_bos(
        "<bos><start_of_turn>user\nQuestion", tokenizer
    )

    assert token_ids == [2, 105, 2364, 107]
    assert tokenizer.calls[0]["add_special_tokens"] is False


def test_prefix_without_bos_gets_single_bos_prepended():
    module = load_sample_branches_module()

    token_ids = module.build_prefix_tokens_with_bos(
        "<start_of_turn>user\nQuestion", FakeNoBosTokenizer()
    )

    assert token_ids == [2, 105, 2364, 107]


def test_sampling_uses_explicit_token_prompt_when_prefix_token_ids_are_available():
    module = load_sample_branches_module()
    llm = FakeLLM()
    prefix_token_ids = [2, 105, 2364, 107]

    continuations, total_samples = module.sample_continuations_natural(
        llm=llm,
        prefix="<bos><start_of_turn>user\nQuestion",
        prefix_token_ids=prefix_token_ids,
        sampling_config=FakeSamplingConfig(),
        max_total_continuations=1,
        max_batches=1,
        logger=FakeLogger(),
    )

    assert llm.prompts == [
        {
            "prompt": "<bos><start_of_turn>user\nQuestion",
            "prompt_token_ids": prefix_token_ids,
        }
    ]
    assert total_samples == 1
    assert continuations[0].text == "continuation"


def test_temp1_rescore_uses_prompt_logprobs_at_matching_token_positions():
    module = load_sample_branches_module()
    cont = module.Continuation(
        text=" continuation",
        token_ids=[300, 400],
        logprob=0.0,
        probability=1.0,
    )

    module.compute_temp1_logprobs(
        llm=FakeScoringLLM(),
        tokenizer=object(),
        prefix="<bos><start_of_turn>user\nQuestion",
        continuations=[cont],
        logger=FakeLogger(),
        batch_size=1,
        prefix_token_ids=[2, 105],
    )

    assert cont.logprob == pytest.approx(-3.0)
    assert cont.probability == pytest.approx(0.2231301601)
