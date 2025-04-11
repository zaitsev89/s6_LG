# LangGraph Examples

This repository contains various examples of using LangGraph for building AI agent workflows.

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs using a directed graph. These examples demonstrate different capabilities of LangGraph.

## Examples

1. **Basic Graph (1_graph_basic.py)**: A simple graph with a single LLM node.
2. **Graph with Tools (2_graph_tools.py)**: Adding tool capabilities to the graph.
3. **Graph with Memory (3_graph_memory.py)**: Implementing persistent memory in the graph.
4. **Human-in-the-Loop (4_graph_human_in_the_loop.py)**: Adding human interaction to the workflow.
5. **State Management (5_graph_state.py)**: Managing state in a LangGraph workflow.

## Features

- OpenAI integration
- Perplexity for internet search
- Human-in-the-loop capabilities
- Memory persistence
- State management
- Tool usage

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   ```

## Usage

Run the examples using Python:

```bash
python main.py
```

Or run a specific example directly:

```bash
python 1_graph_basic.py
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt 