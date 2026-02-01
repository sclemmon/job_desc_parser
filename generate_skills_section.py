"""
generate_skills_section.py
---------------------------
Reads tailored_skills JSON files and generates formatted "Areas of Expertise"
sections that fit your resume's 9-slot, 35-character-per-slot layout.

Usage:
    # Single file mode (prints to console)
    python generate_skills_section.py <path_to_tailored_skills_json>

    # Batch mode (processes all files in tailored_skills/, saves to skills_sections/)
    python generate_skills_section.py --batch

Example:
    python generate_skills_section.py tailored_skills/job_a1b2c3d4.json
    python generate_skills_section.py --batch
"""

import sys
import os
import json
import glob
from textwrap import wrap

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MAX_SLOTS = 9
MAX_CHARS_PER_SLOT = 35  # "Gen AI + Machine Learning and thi" = 35 chars

TAILORED_SKILLS_DIR = "tailored_skills"
OUTPUT_DIR = "skills_sections"
BASELINE_SKILLS_FILE = "baseline_skills.txt"

# ---------------------------------------------------------------------------
# STEP 1: Load the tailored skills JSON
# ---------------------------------------------------------------------------

def load_tailored_skills(filepath):
    """Reads a tailored_skills JSON file and returns the parsed data."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_baseline_skills(filepath):
    """
    Reads the baseline skills file and returns a list of default skills.
    These are used as fallbacks when job-specific matches don't reach 9.
    """
    skills = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip blanks and comments
            if line and not line.startswith("#"):
                skills.append(line)
    return skills


# ---------------------------------------------------------------------------
# STEP 2: Extract and flatten all skills from the tailored section
# ---------------------------------------------------------------------------

def get_output_filename(data):
    """
    Generates a human-readable filename for the output based on company and job title.
    E.g. "Acme_Corp_Senior_Product_Manager_skills.txt"
    """
    job_title = data.get("job_title", "Unknown_Role")
    company = data.get("company", "Unknown_Company")
    
    # Sanitize for filenames: keep alphanumeric, spaces, and hyphens
    safe_title = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in job_title)
    safe_company = "".join(c if c.isalnum() or c in (' ', '-') else '' for c in company)
    
    # Replace spaces with underscores
    safe_title = safe_title.replace(' ', '_').strip('_')
    safe_company = safe_company.replace(' ', '_').strip('_')
    
    # Truncate if too long
    safe_title = safe_title[:50]
    safe_company = safe_company[:30]
    
    return f"{safe_company}_{safe_title}_skills.txt"


def extract_skills(data):
    """
    Pulls all skills from the tailored_skills_section.
    Returns a flat list of skill strings.
    """
    skills = []
    tailored = data.get("tailored_skills_section", {})
    
    for category, skill_list in tailored.items():
        skills.extend(skill_list)
    
    return skills


# ---------------------------------------------------------------------------
# STEP 3: Use LLM to intelligently condense skills to fit character limit
# ---------------------------------------------------------------------------

def condense_skills_with_llm(skills, baseline_skills, max_chars=MAX_CHARS_PER_SLOT, max_slots=MAX_SLOTS, retry_count=0):
    """
    Sends the full list of skills to an LLM and asks it to:
    1. Identify strong job-specific matches
    2. If fewer than 9 strong matches, pad with baseline skills to reach 9 total
    3. Condense each one to fit within max_chars while preserving meaning
    
    Returns a list of exactly 9 condensed skill strings.
    """
    from openai import OpenAI
    
    client = OpenAI()
    
    skills_text = "\n".join([f"  - {s}" for s in skills])
    baseline_text = "\n".join([f"  - {s}" for s in baseline_skills])
    
    prompt = f"""You are a resume formatting assistant. A candidate needs to select and condense skills for their resume's "Areas of Expertise" section.

You have two inputs:
1. JOB-SPECIFIC SKILLS: Skills extracted from matching the candidate's background to this specific job posting
2. BASELINE SKILLS: The candidate's default skills that are broadly applicable across PM roles

Your task:
1. Identify the STRONG job-specific matches — skills that are directly relevant to this posting and would improve ATS screening odds
2. If you have fewer than {max_slots} strong job-specific matches, pad the remaining slots with baseline skills that maximize diversity
3. CRITICAL: Filter out overly generic skills that are obvious given the role:
   - NEVER include "Product Management" or "Product Management Experience" — this is redundant for a PM role
   - NEVER include "Product Manager" or variations like "Experienced Product Manager"
   - Avoid other self-evident skills like "Problem Solving" or "Working with Teams"
   - Only include skills that are specific and meaningful
4. For each selected skill, if it's longer than {max_chars} characters, condense it intelligently:
   - Preserve the core meaning
   - CRITICAL: NEVER EVER truncate with "..." — always rephrase to fit naturally within the limit
   - If a skill is too long, you MUST creatively shorten it using these techniques:
     * Use "&" instead of "and"
     * Drop articles (the, a, an)
     * Use common abbreviations ONLY if the full version won't fit (ML for Machine Learning, AI for Artificial Intelligence, AP for Accounts Payable, AR for Accounts Receivable)
     * Rephrase to be more concise while keeping the meaning
   - Example: "Accounts Payable & Receivable Workflows" (40 chars) → "AP & AR Workflows" (17 chars)
   - Example: "SaaS & Cloud-Based Financial Solutions" (38 chars) → "SaaS Financial Solutions" (24 chars)
   - The goal is professional, scannable text — not cryptic abbreviations or truncation
5. FORMAT: Use Title Case for every skill (capitalize the first letter of each major word)
   - Example: "product development" → "Product Development"
   - Example: "0-to-1 product dev" → "0-1 Product Development"
   - Keep common acronyms in all caps (SQL, API, ML, AI, AWS, AP, AR, etc.)

Prioritization logic:
- Strong job-specific matches come first
- If you need to use baseline skills to reach {max_slots} total, select the ones that maximize coverage
  and diversity. For example:
  - If job-specific matches already include SQL/Python, don't add more technical tools from baseline
  - If job-specific matches are all technical, add soft skills or strategic skills from baseline
  - Pick baseline skills that round out the skillset and show breadth
- Do NOT force weak job-specific matches just to avoid using baseline skills
- If a job-specific skill is generic (e.g. "JIRA" when the job never mentions it), don't count it as a strong match

Return ONLY a valid JSON array of exactly {max_slots} strings (no markdown, no explanation):
["skill1", "skill2", "skill3", "skill4", "skill5", "skill6", "skill7", "skill8", "skill9"]

Each string MUST be {max_chars} characters or fewer and in Title Case.
ABSOLUTE REQUIREMENTS:
- NEVER use "..." anywhere in any skill
- NEVER include "Product Management" or "Product Management Experience"
- If a skill doesn't fit in {max_chars} chars, you MUST creatively condense it (use &, drop articles, abbreviate key terms only when necessary)
- Every skill must be a complete, professional phrase — no truncation marks

JOB-SPECIFIC SKILLS (from matching this posting):
{skills_text}

BASELINE SKILLS (fallbacks to use if needed):
{baseline_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case the model wraps it in markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    condensed = json.loads(raw)
    
    # Post-processing validation and cleanup
    filtered = []
    has_ellipses = False
    
    for skill in condensed:
        # Filter out overly generic PM skills
        if any(generic in skill.lower() for generic in [
            "product management experience",
            "product management",
            "experienced product manager",
            "product manager"
        ]):
            continue  # Skip this skill
        
        # Check for ellipses
        if "..." in skill:
            has_ellipses = True
            print(f"  [WARNING] Skill contains ellipses: '{skill}'")
            continue  # Skip this skill entirely
        
        # Strip redundant " Skills" suffix/infix
        # E.g. "Leadership Skills" → "Leadership"
        # E.g. "Product Design & Development Skills" → "Product Design & Development"
        if skill.endswith(" Skills"):
            skill = skill[:-7]  # Remove " Skills" (7 characters)
        elif skill.endswith(" Skill"):
            skill = skill[:-6]  # Remove " Skill" (6 characters)
        elif skill.endswith(" Expertise"):
            skill = skill[:-10] # Remove " Expertise" (10 characters
        
        # Validate character limit (after cleanup)
        if len(skill) > max_chars:
            print(f"  [WARNING] Skill exceeds {max_chars} chars: '{skill}' ({len(skill)} chars)")
            continue  # Skip this skill
        
        filtered.append(skill)
    
    # If we found ellipses and this is our first attempt, retry
    if has_ellipses and retry_count == 0:
        print(f"  [RETRY] Ellipses detected. Regenerating skills...")
        return condense_skills_with_llm(skills, baseline_skills, max_chars, max_slots, retry_count=1)
    
    # If we filtered out skills and now have fewer than max_slots, that's a problem
    if len(filtered) < max_slots:
        print(f"  [WARNING] Only {len(filtered)} valid skills after filtering (need {max_slots}). This may leave empty slots.")
        # Pad with remaining baseline skills to avoid empty slots
        for baseline in baseline_skills:
            if len(filtered) >= max_slots:
                break
            if baseline not in filtered and len(baseline) <= max_chars:
                filtered.append(baseline)
    
    return filtered[:max_slots]  # Ensure we return exactly max_slots


# ---------------------------------------------------------------------------
# STEP 4: Format as a 3-column layout
# ---------------------------------------------------------------------------

def format_as_columns(skills):
    """
    Formats the skills into a 3-column layout (3 rows x 3 columns).
    Each column is center-aligned.
    
    Returns a formatted string ready for copy/paste.
    """
    # Pad the list to 9 items if we have fewer
    while len(skills) < 9:
        skills.append("")
    
    # Split into 3 columns
    col1 = skills[0:3]
    col2 = skills[3:6]
    col3 = skills[6:9]
    
    # Determine column widths (fixed at MAX_CHARS_PER_SLOT for alignment)
    col_width = MAX_CHARS_PER_SLOT
    
    # Build the formatted output
    output = "Areas of Expertise\n"
    output += "=" * (col_width * 3 + 10) + "\n\n"
    
    for i in range(3):
        line = f"{col1[i]:^{col_width}}  {col2[i]:^{col_width}}  {col3[i]:^{col_width}}"
        output += line.rstrip() + "\n"
    
    return output


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def process_single_file(filepath, baseline_skills, print_output=True, save_to_file=False, skip_if_exists=False):
    """
    Process a single tailored_skills JSON file.
    
    Args:
        filepath: Path to the JSON file
        baseline_skills: List of default skills to use as fallbacks
        print_output: If True, print to console
        save_to_file: If True, save to skills_sections/ directory
        skip_if_exists: If True, skip if output file already exists
    
    Returns the formatted skills section as a string.
    """
    # Load data first to get job title and company for filename
    try:
        data = load_tailored_skills(filepath)
    except FileNotFoundError:
        if print_output:
            print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        if print_output:
            print(f"Error: Invalid JSON in {filepath}")
        return None
    
    # Check if output already exists (using human-readable filename)
    if skip_if_exists and save_to_file:
        output_filename = get_output_filename(data)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            if print_output:
                print(f"[SKIP] Skills section already exists: {output_path}")
            return None
    
    # Extract skills
    skills = extract_skills(data)
    
    if not skills:
        if print_output:
            print(f"No skills found in {filepath}")
        return None
    
    # Use LLM to select strong matches and pad with baseline skills
    if print_output:
        print(f"  Sending {len(skills)} job-specific skills to LLM...")
        print(f"  Baseline skills available for padding if needed")
    
    try:
        top_skills = condense_skills_with_llm(skills, baseline_skills)
    except Exception as e:
        if print_output:
            print(f"  [ERROR] LLM condensing failed: {e}")
        return None
    
    # Format
    formatted = format_as_columns(top_skills)
    
    # Add job metadata header
    job_title = data.get("job_title", "Unknown")
    company = data.get("company", "Unknown")
    
    output = f"TAILORED SKILLS SECTION\n"
    output += f"Job: {job_title}\n"
    output += f"Company: {company}\n"
    output += "=" * 80 + "\n\n"
    output += formatted
    output += "\n" + "=" * 80 + "\n"
    
    if print_output:
        print(f"\nProcessed: {job_title} at {company}")
        print(f"Selected {len(top_skills)} skills:\n")
        for i, skill in enumerate(top_skills, 1):
            print(f"  {i}. {skill} ({len(skill)} chars)")
        print("\n" + output)
    
    # Save to file if requested
    if save_to_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_filename = get_output_filename(data)
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, "w") as f:
            f.write(output)
        
        if print_output:
            print(f"Saved to: {output_path}\n")
    
    return output


def process_batch():
    """
    Process all JSON files in the tailored_skills/ directory
    and save formatted outputs to skills_sections/.
    Skips files that already have an output.
    """
    # Load baseline skills
    if not os.path.exists(BASELINE_SKILLS_FILE):
        print(f"Error: Baseline skills file not found: {BASELINE_SKILLS_FILE}")
        print("Create this file with your default skills before running.")
        return
    
    baseline_skills = load_baseline_skills(BASELINE_SKILLS_FILE)
    print(f"Loaded {len(baseline_skills)} baseline skills from {BASELINE_SKILLS_FILE}")
    
    files = sorted(glob.glob(os.path.join(TAILORED_SKILLS_DIR, "*.json")))
    
    if not files:
        print(f"No JSON files found in {TAILORED_SKILLS_DIR}/")
        return
    
    print(f"Found {len(files)} tailored skills file(s)")
    print("=" * 80 + "\n")
    
    processed = 0
    skipped = 0
    
    for filepath in files:
        result = process_single_file(filepath, baseline_skills, print_output=True, save_to_file=True, skip_if_exists=True)
        if result is None:
            skipped += 1
        else:
            processed += 1
    
    print("=" * 80)
    print(f"Done! Processed {processed} file(s), skipped {skipped} file(s).")
    print(f"Check {OUTPUT_DIR}/ for formatted skills sections.")


def main():
    # Batch mode
    if len(sys.argv) == 2 and sys.argv[1] == "--batch":
        process_batch()
        return
    
    # Single file mode
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python generate_skills_section.py <path_to_tailored_skills_json>")
        print("  python generate_skills_section.py --batch")
        print("\nExamples:")
        print("  python generate_skills_section.py tailored_skills/job_a1b2c3d4.json")
        print("  python generate_skills_section.py --batch")
        sys.exit(1)
    
    # Load baseline skills
    if not os.path.exists(BASELINE_SKILLS_FILE):
        print(f"Error: Baseline skills file not found: {BASELINE_SKILLS_FILE}")
        print("Create this file with your default skills before running.")
        sys.exit(1)
    
    baseline_skills = load_baseline_skills(BASELINE_SKILLS_FILE)
    
    filepath = sys.argv[1]
    output = process_single_file(filepath, baseline_skills, print_output=True, save_to_file=False)
    
    if output:
        print("\nCopy the section above and paste it into your resume.")
        print("Note: This is plain text. You'll need to apply center-alignment")
        print("and any font styling manually in your document editor.\n")


if __name__ == "__main__":
    main()
