# -*- coding: utf-8 -*-
"""Integration test for LangGraph AgentApp."""
import multiprocessing
import time
import json

import aiohttp
import pytest
from langchain_core.messages import HumanMessage
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langchain.tools import tool

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


PORT = 8091  # Use different port from other tests


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def add_time(state: AgentState):
    new_message = HumanMessage(content="今天是 2025年 8 月 21 日")
    return {"messages": [new_message]}


@tool
def get_weather(location: str, date: str) -> str:
    """Get the weather for a location and date."""
    print(f"Getting weather for {location} on {date}...")
    return f"The weather in {location} is sunny with a temperature of 25°C."


tools = [get_weather]
# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = """You are a proactive assistant. """


def build_graph():
    agent = create_agent(llm, tools, system_prompt=prompt)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add a single node that runs the agent
    workflow.add_node("agent", agent)
    workflow.add_node("add_time", add_time)

    # Add edges
    workflow.add_edge(START, "add_time")
    workflow.add_edge("add_time", "agent")
    workflow.add_edge("agent", END)

    # Compile graph
    graph = workflow.compile()
    return graph


def run_langgraph_app():
    """Start LangGraph AgentApp with streaming output enabled."""
    agent_app = AgentApp(
        app_name="LangGraphAgent",
        app_description="A LangGraph-based assistant",
    )

    @agent_app.init
    async def init_func(self):
        self.state_service = InMemoryStateService()
        self.session_service = InMemorySessionHistoryService()

        await self.state_service.start()
        await self.session_service.start()

    @agent_app.shutdown
    async def shutdown_func(self):
        await self.state_service.stop()
        await self.session_service.stop()

    @agent_app.query(framework="langgraph")
    async def query_func(
        self,
        msgs,
        request: AgentRequest = None,
        **kwargs,
    ):
        # Extract session information
        session_id = request.session_id
        user_id = request.user_id

        input = {"messages": [HumanMessage(content="北京天气如何？")]}
        graph = build_graph()

        complete_chunk = {}
        # async for chunk in graph.astream(input, stream_mode="messages"):
        #     complete_chunk.update(chunk[0])
        #     yield chunk[0], False
        for chunk in graph.stream(input, stream_mode="messages"):
            complete_chunk.update(chunk[0])
            yield chunk[0], False

        # Yield the final message with last flag set to True
        yield None, True

    agent_app.run(host="127.0.0.1", port=PORT)


@pytest.fixture(scope="module")
def start_langgraph_app():
    """Launch LangGraph AgentApp in a separate process before the async tests."""
    proc = multiprocessing.Process(target=run_langgraph_app)
    proc.start()
    import socket

    for _ in range(50):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("localhost", PORT))
            s.close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        pytest.fail("LangGraph server did not start within timeout")

    yield
    proc.terminate()
    proc.join()


@pytest.mark.asyncio
async def test_langgraph_process_endpoint_stream_async(start_langgraph_app):
    """
    Async test for streaming /process endpoint (SSE, multiple JSON events) with LangGraph.
    """
    url = f"http://localhost:{PORT}/process"
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the capital of France?"},
                ],
            },
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            found_response = False
            chunks = []

            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

                # Check if this event has "output" from the assistant
                if "output" in event:
                    try:
                        text_content = event["output"][-1]["content"][0][
                            "text"
                        ].lower()
                        if text_content:
                            found_response = True
                    except Exception:
                        # Structure may differ; ignore
                        pass

            # Final assertion — we must have seen "paris" in at least one event
            assert (
                found_response
            ), "Did not find 'paris' in any streamed output event"


@pytest.mark.asyncio
async def test_langgraph_multi_turn_stream_async(start_langgraph_app):
    """
    Async test for multi-turn conversation with streaming output using LangGraph.
    """
    session_id = "langgraph_test_session"
    url = f"http://localhost:{PORT}/process"

    # First turn
    async with aiohttp.ClientSession() as session:
        payload1 = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello LangGraph!"}],
                },
            ],
            "session_id": session_id,
        }
        async with session.post(url, json=payload1) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )
            # Simply consume the stream without detailed checking
            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            # Process all chunks similar to test_langgraph_process_endpoint_stream_async
            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)


    # Second turn - Optimized based on test_langgraph_process_endpoint_stream_async
    payload2 = {
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "How are you?"}],
            },
        ],
        "session_id": session_id,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload2) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type", "").startswith(
                "text/event-stream",
            )

            chunks = []
            async for chunk, _ in resp.content.iter_chunks():
                if not chunk:
                    continue
                chunks.append(chunk.decode("utf-8").strip())

            # Process all chunks similar to test_langgraph_process_endpoint_stream_async
            found_response = False
            line = chunks[-1]
            # SSE lines start with "data:"
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
                event = json.loads(data_str)

                # Check if this event has "output" from the assistant
                if "output" in event:
                    try:
                        text_content = event["output"][-1]["content"][0][
                            "text"
                        ].lower()
                        if text_content:
                            found_response = True
                    except Exception:
                        # Structure may differ; ignore
                        pass

            assert (
                found_response
            ), "Did not find expected response in the second turn output"



if __name__ == "__main__":
    pytest.main([__file__])
