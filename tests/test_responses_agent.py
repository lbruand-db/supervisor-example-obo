"""Smoke tests for the Model Serving wrapper.

We don't exercise predict() here — that requires a live serving runtime
where ModelServingUserCredentials can resolve. We only verify the module
imports cleanly and the subclass is wired correctly.
"""


def test_responses_agent_imports():
    from agent_server import responses_agent  # noqa: F401

    assert hasattr(responses_agent, "SupervisorAgent")


def test_supervisor_agent_subclass_shape():
    from mlflow.pyfunc import ResponsesAgent

    from agent_server.responses_agent import SupervisorAgent

    assert issubclass(SupervisorAgent, ResponsesAgent)
    # Both sync and streaming hooks must exist.
    assert callable(getattr(SupervisorAgent, "predict", None))
    assert callable(getattr(SupervisorAgent, "predict_stream", None))
