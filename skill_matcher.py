"""
skill_matcher.py
----------------
Reads your master skills inventory and all parsed job JSON files,
uses an LLM to semantically match your skills against each job's requirements,
and generates a tailored skills section for each posting.

Usage:
    python skill_matcher.py
"""

import os
import json
import glob
from openai import OpenAI
from datetime import datetime

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Path to your master skills inventory
SKILLS_FILE = "skills_inventory.txt"

# Directory where parsed job JSONs live (output of job_parser.py)
PARSED_JOBS_DIR = "parsed_jobs"

# Directory where matched/tailored output will be saved
OUTPUT_DIR = "tailored_skills"

# OpenAI model
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# STEP 1: Parse the skills inventory file
# ---------------------------------------------------------------------------

def load_skills_inventory(filepath):
    """
    Reads skills_inventory.txt and returns a dict keyed by category.
    Example: { "TECHNICAL_TOOLS": ["SQL", "Python", ...], ... }
    """
    inventory = {}
    current_category = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip blanks and comments
            if not line or line.startswith("#"):
                continue

            # Category header
            if line.startswith("[") and line.endswith("]"):
                current_category = line[1:-1]
                inventory[current_category] = []
                continue

            # Skill entry
            if current_category:
                inventory[current_category].append(line)

    return inventory


# ---------------------------------------------------------------------------
# STEP 2: Load all parsed job JSONs
# ---------------------------------------------------------------------------

def load_parsed_jobs(directory):
    """
    Reads all JSON files in parsed_jobs/ and returns a list of dicts.
    """
    jobs = []
    for filepath in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(filepath, "r") as f:
            job = json.load(f)
            job["_source_file"] = filepath  # track where it came from
            jobs.append(job)
    return jobs


# ---------------------------------------------------------------------------
# STEP 3: Check if a job has already been matched
# ---------------------------------------------------------------------------

def get_output_path(job):
    """
    Generates a human-readable output path for a matched job.
    Uses company and job title from the parsed data.
    E.g. tailored_skills/Acme_Corp_Senior_Product_Manager.json
    """
    job_title = job.get("job_title", "Unknown_Role")
    company = job.get("company", "Unknown_Company")
    
    # Sanitize for filenames: keep alphanumeric, spaces, and hyphens
    safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in job_title)
    safe_company = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in company)
    
    # Replace spaces with underscores
    safe_title = safe_title.replace(' ', '_').strip('_')
    safe_company = safe_company.replace(' ', '_').strip('_')
    
    # Truncate if too long
    safe_title = safe_title[:50]
    safe_company = safe_company[:30]
    
    filename = f"{safe_company}_{safe_title}.json"
    return os.path.join(OUTPUT_DIR, filename)


def already_matched(job):
    """Returns True if we've already generated a tailored skills section for this job."""
    return os.path.exists(get_output_path(job))


# ---------------------------------------------------------------------------
# STEP 4: Send to LLM for semantic matching and tailored output
# ---------------------------------------------------------------------------

def match_and_tailor(job, inventory):
    """
    Sends the parsed job data and your skills inventory to the LLM.
    The LLM does semantic matching (e.g. it knows "stakeholder management"
    matches "cross-functional leadership") and generates a tailored skills
    section using the job posting's own phrasing where possible.
    """
    client = OpenAI()

    # Format the inventory as a readable block for the prompt
    inventory_text = ""
    for category, skills in inventory.items():
        if skills:
            inventory_text += f"\n{category}:\n"
            for skill in skills:
                inventory_text += f"  - {skill}\n"

    # Format the job's extracted data
    job_text = json.dumps(
        {k: v for k, v in job.items() if not k.startswith("_")},
        indent=2
    )

    prompt = f"""You are a resume tailoring assistant. A candidate is applying for a job and needs
a tailored skills section that maximizes their match against the posting.

You have two inputs:
1. The candidate's master skills inventory (everything they know/have done)
2. A parsed job description (the skills the posting is looking for)

Your job:
- Match the candidate's skills against what the job is asking for, using SEMANTIC matching.
  For example, if the job asks for "cross-functional leadership" and the candidate has
  "cross-functional collaboration" and "stakeholder management", that's a match.
- For each match, use the JOB POSTING's phrasing (not the candidate's), because ATS systems
  screen for the posting's exact keywords.
- Only include skills the candidate genuinely has. Do not invent or exaggerate.
- Organize the output into a clean, resume-ready skills section.
- If there are strong matches, note them. If there are gaps (skills the job wants that the
  candidate doesn't have), flag them separately so the candidate is aware.

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{
  "job_title": "from the parsed job",
  "company": "from the parsed job",
  "match_summary": "1-2 sentence high-level assessment of how well the candidate fits",
  "tailored_skills_section": {{
    "Technical Skills": ["skill1", "skill2"],
    "Tools & Platforms": ["tool1", "tool2"],
    "Industry Experience": ["industry1"],
    "Soft Skills & Leadership": ["skill1", "skill2"]
  }},
  "gaps": ["skill the job wants that the candidate has little to no relevant overlap with", ...],
  "partial_matches": ["skill the job wants where the candidate covers some but not all aspects — e.g. the posting asks for 'data science, underwriting, or risk' and the candidate has data science but not underwriting", ...],
  "notes": "Any additional observations about the match or how to position the application"
}}

Rules:
- Use the job posting's phrasing for skills in tailored_skills_section.
- Only put a skill in a category if there is a genuine match to the candidate's inventory.
- CRITICAL: Only include skills in tailored_skills_section if they are DIRECTLY mentioned in the
  job posting OR clearly implied by specific requirements. Do NOT include generic skills that are
  common to all product management roles but not specifically called out in this posting.
  Examples:
  - If the posting mentions "working with engineering teams" but does NOT mention JIRA, Confluence,
    or any specific project management tools, do NOT include those tools.
  - If the posting mentions "data analysis" or "working with data", you CAN include SQL, Python, or
    data tools the candidate has.
  - If the posting mentions "payments", "fintech", or "financial systems", you CAN include related
    industry experience.
  The test: Would an ATS keyword scan flag this skill? If the word/phrase doesn't appear in the
  posting (or a close synonym), don't include it.
- If a category has no matches, use an empty list — do not omit the category.
- gaps should ONLY contain skills where the candidate has little to no relevant experience.
  These are things that would be genuinely difficult to argue for in an interview.
- partial_matches should contain skills where the candidate has meaningful but incomplete overlap.
  For example: the posting asks for "experimentation in regulated industries" and the candidate has
  both experimentation experience and regulated industry experience — that's a partial match, not a gap.
  Or: the posting asks for "data science, underwriting, or risk partners" and the candidate has
  data science experience — that's a partial match, not a gap.
  When in doubt, err toward partial_matches rather than gaps. The goal is to flag only the things
  that are truly out of reach.
- Keep tailored_skills_section concise — aim for the strongest 3-5 skills per category.

---

CANDIDATE'S MASTER SKILLS INVENTORY:
{inventory_text}

---

PARSED JOB DESCRIPTION:
{job_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case the model wraps it in markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ---------------------------------------------------------------------------
# STEP 5: Save the tailored output
# ---------------------------------------------------------------------------

def save_result(job, matched_data):
    """Saves the tailored skills output as a JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "source_url": job.get("source_url", "unknown"),
        "matched_at": datetime.now().isoformat(),
        **matched_data
    }

    output_path = get_output_path(job)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  [SAVED] {output_path}")
    return output


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Skill Matcher & Resume Tailor")
    print("=" * 60)

    # Load skills inventory
    if not os.path.exists(SKILLS_FILE):
        print(f"\n[ERROR] Skills inventory not found: {SKILLS_FILE}")
        print("Make sure skills_inventory.txt is in the repo root.")
        return

    inventory = load_skills_inventory(SKILLS_FILE)
    total_skills = sum(len(v) for v in inventory.values())
    print(f"\nLoaded {total_skills} skills across {len(inventory)} categories from {SKILLS_FILE}")

    # Load parsed jobs
    jobs = load_parsed_jobs(PARSED_JOBS_DIR)
    if not jobs:
        print(f"\nNo parsed jobs found in {PARSED_JOBS_DIR}/.")
        print("Run job_parser.py first to generate parsed job files.")
        return

    print(f"Found {len(jobs)} parsed job(s)\n")

    for i, job in enumerate(jobs, 1):
        title = job.get("job_title", "Unknown")
        company = job.get("company", "Unknown")
        print(f"[{i}/{len(jobs)}] {title} at {company}")

        if already_matched(job):
            print(f"  [SKIP] Already matched. Delete {get_output_path(job)} to re-run.\n")
            continue

        print(f"  [MATCH] Sending to LLM for matching and tailoring...")
        try:
            matched = match_and_tailor(job, inventory)
        except Exception as e:
            print(f"  [MATCH FAILED] {e}\n")
            continue

        save_result(job, matched)

        # Print a quick preview
        print(f"  [SUMMARY] {matched.get('match_summary', 'N/A')}")
        gaps = matched.get("gaps", [])
        if gaps:
            print(f"  [GAPS] {', '.join(gaps)}")
        print()

    print("=" * 60)
    print(" Done! Check the tailored_skills/ folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
