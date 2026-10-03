# Research Agent

A lightweight research agent built from scratch to understand the infrastructure that turns a language model into an agent.

Rather than relying on an agent framework, this project implements the core agent harness directly: model invocation, state and context management, tool execution, validation, and the iterative agent loop.

Given a research objective such as:

> "Research speculative decoding"

the agent autonomously decides which tools to use, observes their results, updates its state, and continues until it produces a final answer.

## Architecture

![Research Agent Architecture](<img width="366" height="558" alt="Screenshot 2026-10-03 at 6 53 10 PM" src="https://github.com/user-attachments/assets/1632c8d4-9a2b-4325-b60d-50f0d5db772d" />
)

The system separates the **language model** from the **harness that controls its execution**.

The harness maintains three core components:

- **Objective / Config** — defines the research query, model, step limit, and timeout.
- **Agent State** — tracks information accumulated during execution, including visited URLs, notes, tool history, and current status.
- **Model Context** — contains the information exposed to the model on each invocation, including instructions, available tools, relevant history, and previous tool results.

The orchestrator runs the core agent loop:

`build context → call model → inspect response → route action → execute tool → update state → repeat`

The model can either request a **tool call** or indicate that it has finished. Tool calls are validated by the harness before execution, ensuring the requested tool exists, its arguments are valid, the action is permitted, and unnecessary repeated calls are prevented.

## Tools

The initial agent has four tools:

- `search_web()` — discover relevant sources
- `read_page()` — retrieve and inspect a source
- `save_note()` — persist useful research findings
- `read_notes()` — retrieve accumulated findings

The model chooses **what it wants to do**, while the harness determines **what is actually executed**.

## Why Build It From Scratch?

Agent frameworks abstract away much of the machinery that makes an LLM agentic. This project intentionally implements that machinery directly to explore how models interact with tools, how state differs from model context, how actions are validated and executed, and how an agent's lifecycle is controlled.

## Status

🚧 Work in progress.
