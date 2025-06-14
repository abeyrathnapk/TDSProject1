# TDS Virtual TA

A virtual Teaching Assistant API that automatically answers student questions for the Tools in Data Science course.

## Features

- Automatically answers questions about model choices, token counting, and course requirements
- Uses course content and Discourse posts as knowledge base
- Supports both text and image-based questions
- Returns relevant links to course materials and discussions

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn requests beautifulsoup4 selenium
```

2. Run the scrapers to build the knowledge base:
```bash
python scrape_tds_course.py
python scrape_tds_discourse.py
```

3. Start the API server:
```bash
uvicorn main:app --reload
```

## API Usage

Send a POST request to `/api/` with a JSON body:

```bash
curl "http://localhost:8000/api/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"}'
```

Response format:
```json
{
  "answer": "You must use `gpt-3.5-turbo-0125`, even if the AI Proxy supports other models like `gpt-4o-mini`. Use the OpenAI API directly for this question.",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/...",
      "text": "GA5 Question 8 Clarification"
    }
  ]
}
```

## Testing

Run the test suite:
```bash
npx -y promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
```

## License

MIT License - see LICENSE file for details.
