"""
Debate Judge API Server v2
Two-stage argument evaluation with configurable AI providers.

Endpoints:
- /api/health - Health check & stats
- /api/debates - List debates from database
- /api/debates/<id> - Get debate with comments
- /api/analyze - Stage 1: Analyze argument potential
- /api/evaluate - Stage 2: Evaluate responses
- /api/full-evaluation - Complete two-stage evaluation
- /api/hub - Debate hub for storing/retrieving evaluated debates
- /api/hub/<id> - Get specific hub entry
- /api/export - Export results
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = Path(__file__).parent / "database" / "debates.db"
HUB_PATH = Path(__file__).parent / "hub_data" / "debates_hub.json"

# AI Provider Configuration
AI_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4",
        "env_key": "OPENAI_API_KEY"
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-opus-20240229",
        "env_key": "ANTHROPIC_API_KEY"
    }
}

# Default provider (can be overridden per-request)
DEFAULT_PROVIDER = os.getenv("AI_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("AI_MODEL", "gpt-4")

# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

STAGE1_ARGUMENT_POTENTIAL_PROMPT = """You are an expert debate analyst. Your task is to analyze an argument and predict what makes a strong or weak response.

ARGUMENT TO ANALYZE:
Title: {title}
Content: {content}

Perform a comprehensive analysis:

1. **CORE CLAIMS**: Identify the 2-4 main claims being made
2. **STRONGEST POINTS**: What aspects of this argument are most defensible?
3. **VULNERABILITIES**: What weaknesses could be exploited by counter-arguments?
4. **EVIDENCE GAPS**: What evidence is missing or assumed?
5. **POTENTIAL FALLACIES**: What logical fallacies might the argument contain?
6. **STRONG COUNTER-ARGUMENT CRITERIA**: What would a high-quality response need to include?
7. **WEAK COUNTER-ARGUMENT PATTERNS**: What response patterns would be low quality?
8. **STANCE PREDICTION**: What percentage of quality responses would likely be FOR vs AGAINST?

Respond in this exact JSON format:
{{
    "core_claims": [
        {{"claim": "<claim text>", "strength": <1-10>, "importance": <1-10>}}
    ],
    "strongest_points": ["<point 1>", "<point 2>"],
    "vulnerabilities": [
        {{"weakness": "<description>", "severity": <1-10>, "exploit_difficulty": <1-10>}}
    ],
    "evidence_gaps": ["<gap 1>", "<gap 2>"],
    "potential_fallacies": [
        {{"type": "<fallacy type>", "location": "<where in argument>", "confidence": <0-100>}}
    ],
    "strong_response_criteria": [
        {{"criterion": "<what good responses should do>", "weight": <1-10>}}
    ],
    "weak_response_patterns": ["<pattern 1>", "<pattern 2>"],
    "predicted_stance_split": {{"for": <0-100>, "against": <0-100>, "neutral": <0-100>}},
    "overall_argument_quality": <0-100>,
    "debate_potential": "<high|medium|low>",
    "analysis_reasoning": "<2-3 sentence summary of your analysis>"
}}"""

STAGE2_RESPONSE_EVALUATION_PROMPT = """You are an expert in argument analysis and critical thinking. Evaluate this response to a debate.

ORIGINAL ARGUMENT:
Title: {original_title}
Content: {original_content}

STAGE 1 ANALYSIS (What makes a good response):
{stage1_analysis}

RESPONSE TO EVALUATE:
Author: {response_author}
Content: {response_content}

Evaluate this response across these dimensions:

1. **ARGUMENT QUALITY** (0-100 each):
   - Overall Score
   - Logical Soundness
   - Evidence Strength  
   - Persuasiveness

2. **LOGICAL FALLACIES**: Identify any fallacies with severity (1-10)

3. **RATIONALITY CONCEPTS**: How well does it demonstrate rational thinking?
   - Bayesian Reasoning, Steelmanning, Epistemic Humility, Occam's Razor, etc.

4. **SENTIMENT & TONE**:
   - Sentiment (-1 to +1)
   - Civility (0-100)
   - Emotional Intensity (0-100)
   - Tone categories

5. **RESPONSE-SPECIFIC ANALYSIS**:
   - Stance (for/against/neutral relative to original)
   - Does it address the core claims?
   - Does it exploit the vulnerabilities identified?
   - Does it meet the strong response criteria?

Respond in this exact JSON format:
{{
    "argument_quality": {{
        "overall_score": <0-100>,
        "logical_soundness": <0-100>,
        "evidence_strength": <0-100>,
        "persuasiveness": <0-100>,
        "reasoning": "<detailed explanation>"
    }},
    "logical_fallacies": {{
        "fallacies_found": [
            {{
                "type": "<fallacy name>",
                "severity": <1-10>,
                "quote": "<relevant quote>",
                "explanation": "<why this is a fallacy>"
            }}
        ],
        "fallacy_count": <number>,
        "overall_reasoning": "<summary>"
    }},
    "rationality_concepts": {{
        "concepts": [
            {{
                "concept_name": "<concept>",
                "relevance_score": <0-10>,
                "demonstration": "<positive|negative|neutral>",
                "explanation": "<how it relates>"
            }}
        ],
        "overall_rationality_score": <0-100>,
        "reasoning": "<summary>"
    }},
    "sentiment_tone": {{
        "sentiment_score": <-1.0 to 1.0>,
        "civility_score": <0-100>,
        "emotional_intensity": <0-100>,
        "tone_categories": ["<category1>", "<category2>"],
        "reasoning": "<explanation>"
    }},
    "response_analysis": {{
        "stance": "<for|against|neutral>",
        "addresses_core_claims": <true|false>,
        "claims_addressed": ["<claim 1>", "<claim 2>"],
        "exploits_vulnerabilities": <true|false>,
        "vulnerabilities_targeted": ["<vulnerability 1>"],
        "meets_strong_criteria": <0-100>,
        "criteria_met": ["<criterion 1>"],
        "exhibits_weak_patterns": <true|false>,
        "weak_patterns_found": ["<pattern>"]
    }},
    "final_verdict": {{
        "quality_tier": "<excellent|good|fair|poor>",
        "recommendation": "<summary recommendation for this response>",
        "key_strengths": ["<strength 1>", "<strength 2>"],
        "key_weaknesses": ["<weakness 1>", "<weakness 2>"]
    }}
}}"""

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection():
    """Get database connection if database exists."""
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def dict_from_row(row):
    """Convert sqlite row to dictionary."""
    return {key: row[key] for key in row.keys()}


def ensure_hub_exists():
    """Ensure hub data directory and file exist."""
    HUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HUB_PATH.exists():
        with open(HUB_PATH, 'w') as f:
            json.dump({"debates": [], "metadata": {"created": datetime.now().isoformat()}}, f)


def load_hub_data():
    """Load hub data from JSON file."""
    ensure_hub_exists()
    with open(HUB_PATH, 'r') as f:
        return json.load(f)


def save_hub_data(data):
    """Save hub data to JSON file."""
    ensure_hub_exists()
    with open(HUB_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# AI PROVIDER INTEGRATION
# ============================================================================

def call_ai(prompt: str, provider: str = None, model: str = None) -> str:
    """
    Call AI provider with the given prompt.
    Supports OpenAI and Anthropic.
    """
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    
    if provider == "openai":
        return call_openai(prompt, model)
    elif provider == "anthropic":
        return call_anthropic(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic API."""
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_ai_response(response_text: str) -> dict:
    """Parse JSON from AI response, handling markdown code blocks."""
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()
    
    return json.loads(json_str)


# ============================================================================
# API ENDPOINTS - Database Access
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check and system status."""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_provider": DEFAULT_PROVIDER,
        "ai_model": DEFAULT_MODEL,
        "database_available": DB_PATH.exists(),
        "hub_entries": 0
    }
    
    # Check database
    if DB_PATH.exists():
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reddit_debates")
        status["debates_count"] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM reddit_comments")
        status["comments_count"] = cursor.fetchone()[0]
        conn.close()
    
    # Check hub
    hub_data = load_hub_data()
    status["hub_entries"] = len(hub_data.get("debates", []))
    
    return jsonify(status)


@app.route('/api/debates', methods=['GET'])
def get_debates():
    """Get debates from database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available", "debates": []}), 200
    
    limit = min(int(request.args.get('limit', 10)), 50)
    min_comments = int(request.args.get('min_comments', 5))
    random_selection = request.args.get('random', 'false').lower() == 'true'
    
    order_clause = "ORDER BY RANDOM()" if random_selection else "ORDER BY score DESC"
    
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, title, selftext, author, score, num_comments, 
               upvote_ratio, created_datetime, flair, permalink
        FROM reddit_debates 
        WHERE num_comments >= ?
        {order_clause}
        LIMIT ?
    """, (min_comments, limit))
    
    debates = [dict_from_row(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({"count": len(debates), "debates": debates})


@app.route('/api/debates/<debate_id>', methods=['GET'])
def get_debate_with_comments(debate_id):
    """Get a specific debate with balanced for/against comments."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available"}), 404
    
    num_comments = min(int(request.args.get('num_comments', 5)), 20)
    
    cursor = conn.cursor()
    
    # Get debate
    cursor.execute("""
        SELECT id, title, selftext, author, score, num_comments,
               upvote_ratio, created_datetime, flair, permalink
        FROM reddit_debates WHERE id = ?
    """, (debate_id,))
    
    debate_row = cursor.fetchone()
    if not debate_row:
        conn.close()
        return jsonify({'error': 'Debate not found'}), 404
    
    debate = dict_from_row(debate_row)
    
    # Get top comments by score (we'll analyze stance later)
    cursor.execute("""
        SELECT id, body, score, author, depth, parent_id, is_op
        FROM reddit_comments 
        WHERE debate_id = ? AND body IS NOT NULL AND LENGTH(body) > 50
        ORDER BY score DESC
        LIMIT ?
    """, (debate_id, num_comments * 2))  # Get extra to allow filtering
    
    comments = [dict_from_row(row) for row in cursor.fetchall()][:num_comments]
    conn.close()
    
    return jsonify({
        'debate': debate,
        'comments': comments,
        'comment_count': len(comments)
    })


# ============================================================================
# API ENDPOINTS - Stage 1: Argument Analysis
# ============================================================================

@app.route('/api/analyze', methods=['POST'])
def analyze_argument():
    """
    Stage 1: Analyze an argument's potential.
    
    Request body:
    {
        "title": "Argument title",
        "content": "Full argument text",
        "provider": "openai",  // optional
        "model": "gpt-4"       // optional
    }
    """
    data = request.get_json()
    if not data or not data.get('content'):
        return jsonify({"error": "Missing 'content' field"}), 400
    
    title = data.get('title', 'Untitled Argument')
    content = data.get('content')
    provider = data.get('provider', DEFAULT_PROVIDER)
    model = data.get('model', DEFAULT_MODEL)
    
    try:
        prompt = STAGE1_ARGUMENT_POTENTIAL_PROMPT.format(
            title=title,
            content=content
        )
        
        response_text = call_ai(prompt, provider, model)
        analysis = parse_ai_response(response_text)
        
        result = {
            "stage": 1,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "input": {
                "title": title,
                "content": content[:500] + "..." if len(content) > 500 else content
            },
            "provider": provider,
            "model": model,
            "analysis": analysis
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# API ENDPOINTS - Stage 2: Response Evaluation
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_response():
    """
    Stage 2: Evaluate a response to an argument.
    
    Request body:
    {
        "original_title": "Original argument title",
        "original_content": "Original argument text",
        "stage1_analysis": {...},  // From /api/analyze
        "response": {
            "author": "responder",
            "content": "Response text"
        },
        "provider": "openai",
        "model": "gpt-4"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    required = ['original_title', 'original_content', 'stage1_analysis', 'response']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing '{field}' field"}), 400
    
    provider = data.get('provider', DEFAULT_PROVIDER)
    model = data.get('model', DEFAULT_MODEL)
    
    try:
        # Format stage1 analysis for prompt
        stage1_summary = json.dumps(data['stage1_analysis'], indent=2)[:2000]
        
        prompt = STAGE2_RESPONSE_EVALUATION_PROMPT.format(
            original_title=data['original_title'],
            original_content=data['original_content'][:1500],
            stage1_analysis=stage1_summary,
            response_author=data['response'].get('author', 'Anonymous'),
            response_content=data['response'].get('content', '')
        )
        
        response_text = call_ai(prompt, provider, model)
        evaluation = parse_ai_response(response_text)
        
        result = {
            "stage": 2,
            "evaluation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "response_author": data['response'].get('author'),
            "evaluation": evaluation
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# API ENDPOINTS - Full Evaluation Pipeline
# ============================================================================

@app.route('/api/full-evaluation', methods=['POST'])
def full_evaluation():
    """
    Complete two-stage evaluation pipeline.
    
    Request body:
    {
        "debate": {
            "id": "optional_id",
            "title": "Argument title",
            "content": "Argument text"
        },
        "responses": [
            {"author": "user1", "content": "Response 1"},
            {"author": "user2", "content": "Response 2"}
        ],
        "provider": "openai",
        "model": "gpt-4",
        "save_to_hub": true
    }
    """
    data = request.get_json()
    if not data or not data.get('debate'):
        return jsonify({"error": "Missing 'debate' field"}), 400
    
    debate = data['debate']
    responses = data.get('responses', [])
    provider = data.get('provider', DEFAULT_PROVIDER)
    model = data.get('model', DEFAULT_MODEL)
    save_to_hub = data.get('save_to_hub', False)
    
    evaluation_id = str(uuid.uuid4())
    
    try:
        # Stage 1: Analyze the argument
        stage1_prompt = STAGE1_ARGUMENT_POTENTIAL_PROMPT.format(
            title=debate.get('title', 'Untitled'),
            content=debate.get('content', '')
        )
        stage1_response = call_ai(stage1_prompt, provider, model)
        stage1_analysis = parse_ai_response(stage1_response)
        
        # Stage 2: Evaluate each response
        evaluated_responses = []
        stage1_summary = json.dumps(stage1_analysis, indent=2)[:2000]
        
        for response in responses:
            stage2_prompt = STAGE2_RESPONSE_EVALUATION_PROMPT.format(
                original_title=debate.get('title', 'Untitled'),
                original_content=debate.get('content', '')[:1500],
                stage1_analysis=stage1_summary,
                response_author=response.get('author', 'Anonymous'),
                response_content=response.get('content', '')
            )
            
            stage2_response = call_ai(stage2_prompt, provider, model)
            evaluation = parse_ai_response(stage2_response)
            
            evaluated_responses.append({
                "response_id": str(uuid.uuid4()),
                "author": response.get('author', 'Anonymous'),
                "content": response.get('content', ''),
                "reddit_score": response.get('score'),
                "evaluation": evaluation
            })
        
        # Compile full result
        result = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "model": model,
            "debate": {
                "id": debate.get('id', evaluation_id),
                "title": debate.get('title', 'Untitled'),
                "content": debate.get('content', ''),
                "author": debate.get('author'),
                "source": debate.get('source', 'manual_input')
            },
            "stage1_analysis": stage1_analysis,
            "evaluated_responses": evaluated_responses,
            "summary": generate_evaluation_summary(stage1_analysis, evaluated_responses)
        }
        
        # Save to hub if requested
        if save_to_hub:
            save_to_debate_hub(result)
            result["saved_to_hub"] = True
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def generate_evaluation_summary(stage1: dict, responses: list) -> dict:
    """Generate summary statistics from evaluation results."""
    if not responses:
        return {"total_responses": 0}
    
    scores = [r['evaluation']['argument_quality']['overall_score'] 
              for r in responses if 'evaluation' in r]
    
    stances = {}
    for r in responses:
        if 'evaluation' in r:
            stance = r['evaluation'].get('response_analysis', {}).get('stance', 'unknown')
            stances[stance] = stances.get(stance, 0) + 1
    
    quality_tiers = {}
    for r in responses:
        if 'evaluation' in r:
            tier = r['evaluation'].get('final_verdict', {}).get('quality_tier', 'unknown')
            quality_tiers[tier] = quality_tiers.get(tier, 0) + 1
    
    return {
        "total_responses": len(responses),
        "average_score": sum(scores) / len(scores) if scores else 0,
        "highest_score": max(scores) if scores else 0,
        "lowest_score": min(scores) if scores else 0,
        "stance_distribution": stances,
        "quality_distribution": quality_tiers,
        "debate_potential": stage1.get('debate_potential', 'unknown'),
        "original_quality": stage1.get('overall_argument_quality', 0)
    }


# ============================================================================
# API ENDPOINTS - Debate Hub
# ============================================================================

def save_to_debate_hub(evaluation_result: dict):
    """Save evaluation result to the debate hub."""
    hub_data = load_hub_data()
    
    hub_entry = {
        "hub_id": evaluation_result['evaluation_id'],
        "added_at": datetime.now().isoformat(),
        "debate_title": evaluation_result['debate']['title'],
        "debate_content": evaluation_result['debate']['content'],
        "debate_author": evaluation_result['debate'].get('author'),
        "source": evaluation_result['debate'].get('source', 'manual'),
        "stage1_analysis": evaluation_result['stage1_analysis'],
        "responses": evaluation_result['evaluated_responses'],
        "summary": evaluation_result['summary'],
        "provider": evaluation_result['provider'],
        "model": evaluation_result['model']
    }
    
    hub_data['debates'].append(hub_entry)
    hub_data['metadata']['last_updated'] = datetime.now().isoformat()
    hub_data['metadata']['total_debates'] = len(hub_data['debates'])
    
    save_hub_data(hub_data)


@app.route('/api/hub', methods=['GET'])
def get_hub():
    """Get all debates in the hub."""
    hub_data = load_hub_data()
    
    # Pagination
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 10)), 50)
    
    debates = hub_data.get('debates', [])
    total = len(debates)
    start = (page - 1) * per_page
    end = start + per_page
    
    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "debates": debates[start:end],
        "metadata": hub_data.get('metadata', {})
    })


@app.route('/api/hub/<hub_id>', methods=['GET'])
def get_hub_entry(hub_id):
    """Get a specific debate from the hub."""
    hub_data = load_hub_data()
    
    for debate in hub_data.get('debates', []):
        if debate.get('hub_id') == hub_id:
            return jsonify(debate)
    
    return jsonify({"error": "Debate not found in hub"}), 404


@app.route('/api/hub', methods=['POST'])
def add_to_hub():
    """Manually add a debate to the hub (for importing)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    hub_data = load_hub_data()
    
    entry = {
        "hub_id": data.get('hub_id', str(uuid.uuid4())),
        "added_at": datetime.now().isoformat(),
        **data
    }
    
    hub_data['debates'].append(entry)
    hub_data['metadata']['last_updated'] = datetime.now().isoformat()
    hub_data['metadata']['total_debates'] = len(hub_data['debates'])
    
    save_hub_data(hub_data)
    
    return jsonify({"success": True, "hub_id": entry['hub_id']})


# ============================================================================
# API ENDPOINTS - Export
# ============================================================================

@app.route('/api/export', methods=['GET'])
def export_results():
    """Export hub data in various formats."""
    format_type = request.args.get('format', 'json')
    
    hub_data = load_hub_data()
    
    if format_type == 'json':
        return jsonify(hub_data)
    elif format_type == 'summary':
        # Return just summaries for quick overview
        summaries = []
        for debate in hub_data.get('debates', []):
            summaries.append({
                "hub_id": debate.get('hub_id'),
                "title": debate.get('debate_title'),
                "summary": debate.get('summary'),
                "added_at": debate.get('added_at')
            })
        return jsonify({"count": len(summaries), "summaries": summaries})
    else:
        return jsonify({"error": f"Unknown format: {format_type}"}), 400


# ============================================================================
# API ENDPOINTS - Batch Processing (for n8n)
# ============================================================================

@app.route('/api/batch/from-database', methods=['POST'])
def batch_from_database():
    """
    Fetch debates from database and prepare for batch evaluation.
    Used by n8n scheduled trigger.
    
    Request body:
    {
        "count": 5,
        "min_comments": 10,
        "comments_per_debate": 5,
        "random": true
    }
    """
    data = request.get_json() or {}
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database not available", "debates": []}), 200
    
    count = min(data.get('count', 5), 20)
    min_comments = data.get('min_comments', 10)
    comments_per = min(data.get('comments_per_debate', 5), 10)
    random_selection = data.get('random', True)
    
    order_clause = "ORDER BY RANDOM()" if random_selection else "ORDER BY score DESC"
    
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, title, selftext, author, score, num_comments
        FROM reddit_debates 
        WHERE num_comments >= ?
        {order_clause}
        LIMIT ?
    """, (min_comments, count))
    
    results = []
    for row in cursor.fetchall():
        debate = dict_from_row(row)
        
        # Get balanced comments (top by score)
        cursor.execute("""
            SELECT id, body, score, author, depth
            FROM reddit_comments 
            WHERE debate_id = ? AND body IS NOT NULL AND LENGTH(body) > 50
            ORDER BY score DESC
            LIMIT ?
        """, (debate['id'], comments_per))
        
        comments = [dict_from_row(c) for c in cursor.fetchall()]
        
        results.append({
            "debate": {
                "id": debate['id'],
                "title": debate['title'],
                "content": debate['selftext'] or '',
                "author": debate['author'],
                "source": "reddit_database",
                "reddit_score": debate['score']
            },
            "responses": [
                {
                    "id": c['id'],
                    "author": c['author'],
                    "content": c['body'],
                    "score": c['score']
                }
                for c in comments
            ]
        })
    
    conn.close()
    
    return jsonify({
        "count": len(results),
        "debates_with_responses": results
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DEBATE JUDGE API SERVER v2")
    print("=" * 60)
    print(f"Database: {DB_PATH} ({'exists' if DB_PATH.exists() else 'NOT FOUND'})")
    print(f"Hub: {HUB_PATH}")
    print(f"AI Provider: {DEFAULT_PROVIDER}")
    print(f"AI Model: {DEFAULT_MODEL}")
    print()
    print("Endpoints:")
    print("  GET  /api/health              - Health check")
    print("  GET  /api/debates             - List debates from DB")
    print("  GET  /api/debates/<id>        - Get debate + comments")
    print("  POST /api/analyze             - Stage 1: Analyze argument")
    print("  POST /api/evaluate            - Stage 2: Evaluate response")
    print("  POST /api/full-evaluation     - Complete pipeline")
    print("  GET  /api/hub                 - Get hub debates")
    print("  GET  /api/hub/<id>            - Get specific hub entry")
    print("  POST /api/hub                 - Add to hub")
    print("  GET  /api/export              - Export results")
    print("  POST /api/batch/from-database - Batch fetch for n8n")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
