"""
App tests — Streamlit UI behaviour (Model 4)
Runs app.py headlessly with the LLM call replaced, so no API key or network is needed.

Run: pytest test_app.py -v
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from main import MarketingStrategyGenerator

APP_PATH = str(Path(__file__).resolve().parent / "app.py")
RUN_TIMEOUT_SEC = 60

GROQ_LABEL = "Groq (Fast & Free)"
OLLAMA_LABEL = "Ollama (Local)"
FAKE_STRATEGY = "FAKE STRATEGY TEXT"
PREDICTION_HEADER = "Model 1: Prediction Details"


@pytest.fixture(autouse=True)
def no_api_keys_in_env():
    """Keep keys from the developer's shell out of the tests."""
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in ("OPENAI_API_KEY", "GROQ_API_KEY")}
    with patch.dict(os.environ, clean_env, clear=True):
        yield


@pytest.fixture
def llm_calls():
    """Replace the cloud LLM call and record which backend/key it was made with."""
    calls = []

    def fake_call(self, user_prompt):
        calls.append({"backend": self.backend, "api_key": self.api_key})
        return {
            "success": True,
            "strategy": FAKE_STRATEGY,
            "metadata": {"model": self.model, "tokens_used": 1000, "cost_estimate": 0.002},
        }

    with patch.object(MarketingStrategyGenerator, "_call_openai_compatible", fake_call):
        yield calls


def launch_app(secrets: dict) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=RUN_TIMEOUT_SEC)
    for name, value in secrets.items():
        at.secrets[name] = value
    return at.run()


def click_run(at: AppTest) -> AppTest:
    return at.sidebar.button[0].click().run()


def page_text(at: AppTest) -> str:
    elements = [*at.markdown, *at.caption, *at.info, *at.warning, *at.error]
    return "\n".join(str(el.value) for el in elements)


def test_results_stay_visible_after_a_sidebar_input_changes(llm_calls):
    at = click_run(launch_app({"OPENAI_API_KEY": "openai-key"}))
    assert PREDICTION_HEADER in page_text(at)

    at.sidebar.slider(key="sb_energy").set_value(0.2).run()

    assert PREDICTION_HEADER in page_text(at)
    assert FAKE_STRATEGY in page_text(at)


def test_results_show_the_inputs_they_were_generated_from(llm_calls):
    at = click_run(launch_app({"OPENAI_API_KEY": "openai-key"}))

    at.sidebar.slider(key="sb_tempo").set_value(80).run()

    text = page_text(at)
    assert "BPM · 🟢 Dance" in text  # the 125 BPM the run used, not the new 80
    assert "inputs have changed" in text.lower()


def test_no_stale_notice_right_after_a_run(llm_calls):
    at = click_run(launch_app({"OPENAI_API_KEY": "openai-key"}))

    assert "inputs have changed" not in page_text(at).lower()


def test_groq_backend_is_called_with_the_groq_key(llm_calls):
    at = launch_app({"OPENAI_API_KEY": "openai-key", "GROQ_API_KEY": "groq-key"})
    at.sidebar.selectbox[0].select(GROQ_LABEL).run()

    click_run(at)

    assert llm_calls == [{"backend": "groq", "api_key": "groq-key"}]


def test_a_missing_key_is_never_replaced_by_another_providers_key(llm_calls):
    with patch.dict(os.environ, {"GROQ_API_KEY": "groq-env-key"}):
        at = launch_app({"UNRELATED": "x"})  # OpenAI selected by default, no OpenAI key anywhere

        click_run(at)

    assert llm_calls == [{"backend": "openai", "api_key": None}]


def test_generator_reads_its_own_backends_key_from_env():
    with patch.dict(os.environ, {"GROQ_API_KEY": "groq-env-key", "OPENAI_API_KEY": "openai-env-key"}):
        assert MarketingStrategyGenerator(backend="groq").api_key == "groq-env-key"
        assert MarketingStrategyGenerator(backend="openai").api_key == "openai-env-key"
        assert MarketingStrategyGenerator(backend="ollama").api_key is None


def test_missing_groq_key_is_reported_by_name(llm_calls):
    at = launch_app({"OPENAI_API_KEY": "openai-key"})

    at.sidebar.selectbox[0].select(GROQ_LABEL).run()

    assert "GROQ_API_KEY" in page_text(at)


def test_openai_run_is_not_labelled_local_and_free(llm_calls):
    at = click_run(launch_app({"OPENAI_API_KEY": "openai-key"}))

    cost_line = next(c.value for c in at.caption if "Generated using" in c.value)
    assert "$0.0020" in cost_line
    assert "local" not in cost_line.lower()


def test_cloud_failure_does_not_tell_the_user_to_start_ollama():
    failure = {"success": False, "error": "API error: invalid key", "strategy": None}
    with patch.object(MarketingStrategyGenerator, "_call_openai_compatible",
                      lambda self, user_prompt: failure):
        at = click_run(launch_app({"OPENAI_API_KEY": "openai-key"}))

    text = page_text(at)
    assert "API error: invalid key" in text
    assert "Ollama is running" not in text
    assert "OPENAI_API_KEY" in text


def test_ollama_failure_still_tells_the_user_to_start_ollama():
    failure = {"success": False, "error": "Cannot connect to Ollama.", "strategy": None}
    with patch.object(MarketingStrategyGenerator, "_call_ollama",
                      lambda self, user_prompt: failure), \
         patch.object(MarketingStrategyGenerator, "check_ollama_status", lambda self: False):
        at = launch_app({"OPENAI_API_KEY": "openai-key"})
        at.sidebar.selectbox[0].select(OLLAMA_LABEL).run()
        click_run(at)

    assert "Ollama is running" in page_text(at)
