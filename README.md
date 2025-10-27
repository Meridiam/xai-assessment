# xAI Assessment

This repository builds upon [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench).
Please refer to the $\tau^2$-bench Github docs for any information that may be missing here.

## Installation

1. Create a new environment

$\tau^2$-bench requires Python 3.10 or higher. You may create and activate a new environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install tau2

```bash
pip install -e .
```

This will enable you to run the `tau2` command.

**Note:** If you use `pip install .` (without `-e`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:

```bash
export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
```

**Check your data directory setup:**

After installation, you can verify that your data directory is correctly configured by running:

```bash
tau2 check-data
```

This command will check if the data directory exists and print instructions if it is missing.

To remove all the generated files and the virtual environment, run:
```bash
make clean
```

## Quick Start

### Setup xAI API keys

To provide your API keys, create a `.env` file and add your xAI API key
as follows:
`XAI_API_KEY=<YOUR_API_KEY_HERE>`.

### Run xAI benchmark

To run the prototype benchmark described in the project writeup:

```tau2 run --domain xai_domain --agent-llm xai/grok-4-fast-non-reasoning --user-llm xai/grok-4-fast-non-reasoning --num-trials 4 --save-to my_model_xai```

Results will be saved in `data/tau2/simulations/` under the name `my_model_xai`.

## Command Line Interface

The `tau2` command provides a unified interface for all functionality:

### Running Benchmark 
```bash
tau2 run \
  --domain <domain> \
  --agent-llm <llm_name> \
  --user-llm <llm_name> \
  --num-trials <trial_count> \
  --task-ids <task_ids> \
  --max-concurrency <concurrent_sims> \
  ...
```

### Viewing Results
```bash
tau2 view
```
This tool allows you to:
- Browse simulation files (in `data/tau2/simulations/`)
- View agent performance metrics
- View a particular simulation
- View task details