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

# ---------------------------------------------------------------------------
# STEP 1: Load the tailored skills JSON
# ---------------------------------------------------------------------------

def load_tailored_skills(filepath):
    """Reads a tailored_skills JSON file and returns the parsed data."""
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# STEP 2: Extract and flatten all skills from the tailored section
# ---------------------------------------------------------------------------

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

def condense_skills_with_llm(skills, max_chars=MAX_CHARS_PER_SLOT, max_slots=MAX_SLOTS):
    """
    Sends the full list of skills to an LLM and asks it to:
    1. Select the top 9 most important skills
    2. Condense each one to fit within max_chars while preserving meaning
    
    Returns a list of up to 9 condensed skill strings.
    """
    from openai import OpenAI
    
    client = OpenAI()
    
    skills_text = "\n".join([f"  - {s}" for s in skills])
    
    prompt = f"""You are a resume formatting assistant. A candidate needs to select and condense skills for their resume's "Areas of Expertise" section.

Constraints:
- The section has exactly {max_slots} slots (3 rows x 3 columns)
- Each slot can hold a MAXIMUM of {max_chars} characters (including spaces)
- Skills that exceed {max_chars} characters MUST be condensed intelligently

Your task:
1. Select the {max_slots} most important/relevant skills from the list below
2. For each selected skill, if it's longer than {max_chars} characters, condense it intelligently:
   - Preserve the core meaning
   - AVOID abbreviations unless absolutely necessary to fit the limit. Always try the full word first.
   - Only use abbreviations if the full version exceeds {max_chars} characters
   - When abbreviations are necessary, use common professional ones (ML for Machine Learning, AI for Artificial Intelligence, etc.)
   - For lists, keep the most important items
   - Use "&" instead of "and" when natural
   - The goal is professional, scannable text — not cryptic abbreviations
3. FORMAT: Use Title Case for every skill (capitalize the first letter of each major word)
   - Example: "product development" → "Product Development"
   - Example: "0-to-1 product dev" → "0-1 Product Development"
   - Keep common acronyms in all caps (SQL, API, ML, AI, AWS, etc.)

Return ONLY a valid JSON array of exactly {max_slots} strings (no markdown, no explanation):
["skill1", "skill2", "skill3", "skill4", "skill5", "skill6", "skill7", "skill8", "skill9"]

Each string MUST be {max_chars} characters or fewer and in Title Case.

Skills to select and condense from:
{skills_text}
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
    
    # Validate that each skill is within the character limit
    for i, skill in enumerate(condensed):
        if len(skill) > max_chars:
            # Force truncate if LLM didn't respect the limit
            condensed[i] = skill[:max_chars - 3] + "..."
    
    return condensed


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

def process_single_file(filepath, print_output=True, save_to_file=False, skip_if_exists=False):
    """
    Process a single tailored_skills JSON file.
    
    Args:
        filepath: Path to the JSON file
        print_output: If True, print to console
        save_to_file: If True, save to skills_sections/ directory
        skip_if_exists: If True, skip if output file already exists
    
    Returns the formatted skills section as a string.
    """
    # Check if output already exists
    if skip_if_exists and save_to_file:
        source_filename = os.path.basename(filepath)
        output_filename = source_filename.replace(".json", "_skills.txt")
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        if os.path.exists(output_path):
            if print_output:
                print(f"[SKIP] Skills section already exists: {output_path}")
            return None
    
    # Load data
    try:
        data = load_tailored_skills(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        return None
    
    # Extract skills
    skills = extract_skills(data)
    
    if not skills:
        if print_output:
            print(f"No skills found in {filepath}")
        return None
    
    # Use LLM to select top 9 and condense to fit character limit
    if print_output:
        print(f"  Sending {len(skills)} skills to LLM for selection and condensing...")
    
    try:
        top_skills = condense_skills_with_llm(skills)
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
        source_filename = os.path.basename(filepath)
        output_filename = source_filename.replace(".json", "_skills.txt")
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
    files = sorted(glob.glob(os.path.join(TAILORED_SKILLS_DIR, "*.json")))
    
    if not files:
        print(f"No JSON files found in {TAILORED_SKILLS_DIR}/")
        return
    
    print(f"Found {len(files)} tailored skills file(s)")
    print("=" * 80 + "\n")
    
    processed = 0
    skipped = 0
    
    for filepath in files:
        result = process_single_file(filepath, print_output=True, save_to_file=True, skip_if_exists=True)
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
    
    filepath = sys.argv[1]
    output = process_single_file(filepath, print_output=True, save_to_file=False)
    
    if output:
        print("\nCopy the section above and paste it into your resume.")
        print("Note: This is plain text. You'll need to apply center-alignment")
        print("and any font styling manually in your document editor.\n")


if __name__ == "__main__":
    main()
