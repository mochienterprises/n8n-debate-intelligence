# Debate Intelligence System - Setup Guide

Complete setup instructions for the debate analysis pipeline.

## Prerequisites

- **Docker Desktop** installed and running
- **n8n** (via Docker)
- **Python 3.8+**
- **MySQL** (or compatible database)
- **Git**

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/n8n-debate-intelligence.git
cd n8n-debate-intelligence
```

---

## Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:
   ```bash
   # Required: Add your API keys
   OPENAI_API_KEY=your_actual_openai_key
   ANTHROPIC_API_KEY=your_actual_anthropic_key
   
   # Optional: Adjust these if needed
   AI_PROVIDER=openai
   AI_MODEL=gpt-4
   ```

---

## Step 3: Set Up the Database

### Option A: Using the Provided Database

The `database/` folder contains the debate data files.

1. Import into MySQL:
   ```bash
   mysql -u your_username -p your_database_name < database/debates.sql
   ```
   
   Or if using a different format, load the data according to your database setup.

### Option B: Create Fresh Database

1. Create a new MySQL database:
   ```sql
   CREATE DATABASE debates_db;
   ```

2. Set up tables (schema will be created by the API on first run)

---

## Step 4: Start n8n with Docker

1. Run n8n in Docker:
   ```bash
   docker run -it --rm \
     --name n8n \
     -p 5678:5678 \
     -v ~/.n8n:/home/node/.n8n \
     n8nio/n8n
   ```

2. Access n8n at `http://localhost:5678`

3. Import the workflow:
   - In n8n, go to **Workflows** → **Import from File**
   - Select `workflows/Debate_Judge_-_Two-Stage_Evaluation_Pipeline.json`
   - Click **Import**

---

## Step 5: Start the Flask API Server

1. Navigate to the API directory:
   ```bash
   cd api
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   python api_server.py
   ```

4. The API should now be running at `http://localhost:5000`

---

## Step 6: Configure n8n Workflow

Once the workflow is imported:

1. **Update API credentials** in n8n nodes:
   - Open the workflow in n8n
   - Click on HTTP Request nodes
   - Update the URL to point to `http://localhost:5000`

2. **Configure AI provider nodes**:
   - Add your OpenAI or Anthropic credentials in n8n settings
   - Update the model selection if needed

3. **Test the workflow**:
   - Click **Execute Workflow** to test
   - Check for any connection errors
   - Verify API responses

---

## Step 7: Verify Everything Works

### Test the API

```bash
# Get all debates
curl http://localhost:5000/debates

# Get specific debate
curl http://localhost:5000/debates/1

# Search debates
curl http://localhost:5000/debates/search?q=climate
```

### Test the Workflow

1. In n8n, click **Execute Workflow**
2. Check that data flows through all nodes
3. Verify output format

---

## Common Issues

### n8n can't connect to API
- Ensure Flask server is running (`python api/api_server.py`)
- Check that port 5000 is not in use
- Verify firewall settings

### Database connection errors
- Confirm MySQL is running
- Check database credentials in `.env`
- Ensure database exists and is accessible

### API key errors
- Verify API keys are correctly set in `.env`
- Check that keys are active and have credits
- Ensure no extra spaces in `.env` file

### Docker issues
- Ensure Docker Desktop is running
- Try restarting Docker
- Check Docker logs: `docker logs n8n`

---

## Directory Structure

```
n8n-debate-intelligence/
├── README.md
├── .env.example          # Template for environment variables
├── .env                  # Your actual credentials (not committed)
├── .gitignore
├── workflows/
│   ├── Debate_Judge_-_Two-Stage_Evaluation_Pipeline.json
│   └── debates_hub.json
├── api/
│   ├── api_server.py     # Flask API server
│   └── requirements.txt  # Python dependencies
├── database/             # Debate database files (20k+ debates)
└── docs/
    └── SETUP_GUIDE.md    # This file
```

---

## Next Steps

- Explore the n8n workflow and customize nodes
- Add more debates to the database
- Integrate with additional AI providers
- Build frontend interface for debate analysis

---

## Support

For issues or questions, open an issue on GitHub.
