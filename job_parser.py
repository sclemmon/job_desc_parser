"""
job_parser.py
-------------
Reads job URLs from jobs.txt, fetches the page, extracts the job description text,
sends it to an LLM for structured skill extraction, and saves the result as a JSON file.

Usage:
    python job_parser.py
"""

import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to the file containing your job URLs (one per line)
JOBS_FILE = "jobs.txt"

# Directory where parsed job JSON files are saved
OUTPUT_DIR = "parsed_jobs"

# OpenAI model to use for parsing
MODEL = "gpt-4o-mini"  # cheap and fast; swap to "gpt-4o" if you want higher quality

# Maximum number of NEW jobs to process in a single run.
# This is a safety cap — even if jobs.txt has hundreds of URLs,
# only this many will be fetched and parsed per run.
MAX_JOBS_PER_RUN = 50

# ---------------------------------------------------------------------------
# STEP 1: Read URLs from jobs.txt
# ---------------------------------------------------------------------------

def load_urls(filepath):
    """
    Reads jobs.txt and returns a list of URLs.
    Skips blank lines and lines starting with #.
    """
    urls = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


# ---------------------------------------------------------------------------
# STEP 2: Check if a URL has already been processed
# ---------------------------------------------------------------------------

def get_output_path(url):
    """
    Generates a unique filename for a given URL using a short hash.
    Example: parsed_jobs/job_a1b2c3d4.json
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return os.path.join(OUTPUT_DIR, f"job_{url_hash}.json")


def already_processed(url):
    """
    Returns True if we've already parsed this URL (output file exists).
    This prevents re-fetching and re-parsing jobs on every run.
    """
    return os.path.exists(get_output_path(url))


# ---------------------------------------------------------------------------
# STEP 3: Fetch and extract text from a job posting page
# ---------------------------------------------------------------------------

def fetch_job_text(url):
    """
    Fetches the job posting page and extracts the visible text.
    Uses a realistic User-Agent header to reduce the chance of being blocked.
    Returns the extracted text as a string, or None if fetching fails.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FETCH FAILED] Could not retrieve {url}")
        print(f"  Reason: {e}")
        print(f"  -> You can manually paste the job description text into the")
        print(f"     'manual_text' field in a JSON file in parsed_jobs/ instead.")
        return None

    # Parse the HTML and extract all visible text
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements (they contain code, not visible text)
    for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
        script_or_style.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text


# ---------------------------------------------------------------------------
# STEP 4: Send text to OpenAI for structured skill extraction
# ---------------------------------------------------------------------------

def parse_with_llm(job_text, url):
    """
    Sends the raw job description text to OpenAI and asks it to extract
    and categorize skills into a structured JSON format.
    """
    client = OpenAI()  # automatically reads OPENAI_API_KEY from environment

    prompt = f"""You are a job description parser. Extract and categorize skills from the following job posting.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "job_title": "the title of the role",
  "company": "the company name",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "technical_tools": ["tool1", "tool2"],
  "languages_and_frameworks": ["Python", "JavaScript"],
  "industry_experience": ["Fintech", "Healthcare"],
  "certifications": ["cert1"],
  "soft_skills": ["skill1", "skill2"],
  "education_requirements": "brief description or null",
  "experience_years": "e.g. '3-5 years' or null"
}}

Rules:
- Use the exact phrasing from the job posting where possible (this matters for keyword matching later).
- If a skill appears under "required" or "must have", put it in required_skills.
- If it appears under "nice to have" or "preferred", put it in preferred_skills.
- If the posting does not clearly distinguish required vs preferred, use your best judgment
  and put the most heavily emphasized skills in required_skills.
- technical_tools is for specific software/platforms (e.g. Jira, AWS, Salesforce).
- languages_and_frameworks is for programming languages and frameworks (e.g. Python, React).
- industry_experience is for specific industries or domains the posting expects familiarity with
  (e.g. Fintech, Healthcare, E-commerce, SaaS, Government). Use the exact terms from the posting.
  This includes both explicit mentions ("experience in fintech") and implicit ones (e.g. a posting
  for a payments company that references "compliance" and "regulatory" work).
- Do not duplicate skills across categories.
- If a category has no matches, use an empty list.

Job posting text:
---
{job_text}
---"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic output for consistency
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case the model wraps it in markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ---------------------------------------------------------------------------
# STEP 5: Save the parsed result
# ---------------------------------------------------------------------------

def save_result(url, parsed_data):
    """
    Saves the parsed job data as a JSON file.
    Also stores the source URL and the timestamp of when it was parsed.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "source_url": url,
        "parsed_at": datetime.now().isoformat(),
        **parsed_data
    }

    output_path = get_output_path(url)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  [SAVED] {output_path}")
    return output


# ---------------------------------------------------------------------------
# MAIN: Orchestrate the pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Job Description Parser")
    print("=" * 60)

    # Load URLs
    urls = load_urls(JOBS_FILE)
    if not urls:
        print(f"\nNo URLs found in {JOBS_FILE}. Add some and try again.")
        return

    print(f"\nFound {len(urls)} job URL(s) in {JOBS_FILE}\n")

    processed_count = 0  # tracks how many NEW jobs we've processed this run

    for i, url in enumerate(urls, 1):
        # Safety cap: stop if we've hit the per-run limit
        if processed_count >= MAX_JOBS_PER_RUN:
            print(f"\n[CAP REACHED] Processed {MAX_JOBS_PER_RUN} new jobs this run. Stopping.")
            print(f"  Remaining URLs will be picked up on the next run.\n")
            break

        print(f"[{i}/{len(urls)}] {url}")

        # Skip if already processed
        if already_processed(url):
            print(f"  [SKIP] Already parsed. Delete {get_output_path(url)} to re-process.\n")
            continue

        # Fetch the page
        print(f"  [FETCH] Retrieving job page...")
        job_text = fetch_job_text(url)
        if not job_text:
            print(f"  [SKIP] Moving to next job.\n")
            continue

        # Parse with LLM
        print(f"  [PARSE] Sending to LLM for skill extraction...")
        try:
            parsed = parse_with_llm(job_text, url)
        except Exception as e:
            print(f"  [PARSE FAILED] {e}")
            print(f"  -> Check your OPENAI_API_KEY and try again.\n")
            continue

        # Save
        save_result(url, parsed)
        processed_count += 1
        print()

    print("=" * 60)
    print(" Done! Check the parsed_jobs/ folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
