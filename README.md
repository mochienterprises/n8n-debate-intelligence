# Debate Intelligence System

Automated debate analysis pipeline with n8n workflow orchestration, Flask API server, and a 20,000+ debate database sourced from Reddit.

## Features

- **n8n Workflow Automation** - Orchestrates debate data processing
- **Flask REST API** - Provides structured access to debate data
- **MySQL Database** - 20k+ debates with arguments, positions, and metadata
- **Automated Processing** - Reduces manual data handling

## Architecture

```
n8n Workflow → Flask API → MySQL Database (20k debates)
```

## Tech Stack

- n8n (workflow automation)
- Flask (Python API)
- MySQL (database)
- Docker (containerization)

## Setup

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. Import `workflows/Debate_Judge_-_Two-Stage_Evaluation_Pipeline.json` into n8n

3. Install dependencies: `pip install -r api/requirements.txt`

4. Run API server: `python api/api_server.py`

5. Load database from `database/` folder

See [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for detailed instructions.

## Use Cases

- Debate research and analysis
- Argument pattern recognition
- Training data for AI/ML models
- Automated debate processing pipelines

## Results

Automated processing of 20,000+ Reddit debates with RESTful API access and scalable workflow orchestration.
