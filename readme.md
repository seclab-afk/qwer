# AFK

Autonomous Fuzzing Kit

# how to build

## docker build

```
sudo docker build -t afk .
```

- you need to bind directory to docker container

```
sudo docker run --privileged -it -v /path/to/afk/:/afk/ -v /var/run/docker.sock:/var/run/docker.sock --name afk afk
```

# need to run

## Ollama

- You must upload OSS-20B model to Ollama server.

## config.py

Edit `agent/config.py` or set environment variables:

| Variable               | Description                | Default                  |
| ---------------------- | -------------------------- | ------------------------ |
| `LLM_SERVER_URL`       | Ollama server URL          | -                        |
| `LLM_MODEL_NAME`       | LLM model name             | `gpt-oss:20b`            |
| `EMBEDDING_MODEL_NAME` | Embedding model for RAG    | `BAAI/bge-small-en-v1.5` |
| `HOST_BASE_PATH`       | Host path for Docker mount | -                        |

# AFK usage

```
python3 agent.py -l libpng -c 50
```

## Running Options

| Option                 | Short | Description                                         | Default |
| ---------------------- | ----- | --------------------------------------------------- | ------- |
| `--lib`                | `-l`  | Library name (required, based on build_config.yaml) | -       |
| `--coverage-threshold` | `-c`  | Target coverage threshold (%)                       | 50.0    |
| `--context-window`     | `-w`  | RAG context window size                             | 60000   |
| `--max-time`           | `-m`  | Maximum execution time in seconds                   | None    |

# File Structure

```
afk/
├── Dockerfile              # Docker image build configuration
├── requirements.txt        # Python dependencies
├── readme.md               # This file
└── agent/
    ├── agent.py            # Main entry point, LangGraph workflow orchestrator
    ├── harness_agent.py    # Parallel harness generation workflow per category
    ├── module.py           # Core functions (build, fuzzing, coverage, RAG)
    ├── dashboard_manager.py # Real-time progress dashboard
    ├── logger_config.py    # Unified logging configuration
    ├── config.py           # LLM server and model configuration
    ├── build_config.yaml   # Library build configurations
    ├── coverage_config.yaml # Coverage measurement settings
    ├── prompts/            # LLM prompt templates
    │   ├── categorize_api_template.txt    # API categorization prompt
    │   ├── make_api_flow_template.txt     # API flow generation prompt
    │   ├── make_harness_template.txt      # Harness generation prompt
    │   ├── fix_harness_template.txt       # Harness fix prompt
    │   └── judgement_template.txt         # LLM harness quality judgement
    ├── corpus/             # Initial seed corpus for fuzzing
    ├── library/            # Built libraries and generated harnesses
    ├── rag_store/          # RAG index and BM25 embeddings
    └── log/                # Execution logs
```

## Key Files Description

| File                | Description                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `agent.py`          | Main orchestrator using LangGraph. Handles library build, API extraction, categorization, and coverage iteration loop.                   |
| `harness_agent.py`  | Manages parallel harness generation per API category. Includes flow generation, harness creation, AFL build, fuzzing, and LLM judgement. |
| `module.py`         | Core utility functions: library cloning/building, RAG index management, BM25 search, Docker fuzzing, coverage measurement.               |
| `build_config.yaml` | YAML config defining how to clone and build each library (api, fuzz, cov variants).                                                      |
