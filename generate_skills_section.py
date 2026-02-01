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
# STEP 3: Truncate or abbreviate skills to fit the character limit
# ---------------------------------------------------------------------------

def fit_to_limit(skill, max_chars=MAX_CHARS_PER_SLOT):
    """
    If a skill is longer than max_chars, try to shorten it intelligently.
    
    Strategies:
    1. If it has common abbreviations (e.g. "Machine Learning" -> "ML"), use them.
    2. If it's a list (e.g. "Tool1, Tool2, Tool3"), keep as many as fit.
    3. As a last resort, truncate with "..." 
    
    Returns the shortened skill string.
    """
    if len(skill) <= max_chars:
        return skill
    
    # Strategy 1: Common abbreviations
    abbreviations = {
        "Machine Learning": "ML",
        "Artificial Intelligence": "AI",
        "Application Programming Interface": "API",
        "User Experience": "UX",
        "User Interface": "UI",
        "Software as a Service": "SaaS",
        "Customer Relationship Management": "CRM",
    }
    
    for full, abbr in abbreviations.items():
        if full in skill:
            skill = skill.replace(full, abbr)
            if len(skill) <= max_chars:
                return skill
    
    # Strategy 2: If it's a comma-separated list, keep as many items as fit
    if "," in skill:
        parts = [p.strip() for p in skill.split(",")]
        result = parts[0]
        for part in parts[1:]:
            if len(result + ", " + part) <= max_chars:
                result += ", " + part
            else:
                break
        if len(result) <= max_chars:
            return result
    
    # Strategy 3: Truncate with ellipsis
    return skill[:max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# STEP 4: Prioritize and select the top 9 skills
# ---------------------------------------------------------------------------

def select_top_skills(skills, max_slots=MAX_SLOTS):
    """
    Takes a list of skills and selects the top max_slots (9) for the resume.
    
    Prioritization logic:
    - Prefer shorter skills (they fit cleanly without truncation)
    - Prefer skills earlier in the list (they're from higher-priority categories)
    
    Returns a list of up to 9 skills.
    """
    # Fit all skills to the character limit
    fitted_skills = [fit_to_limit(s) for s in skills]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in fitted_skills:
        if skill not in seen:
            seen.add(skill)
            unique_skills.append(skill)
    
    # If we have more than 9, prioritize
    if len(unique_skills) > max_slots:
        # Sort by: (1) original order (implicit via enumerate), (2) length (shorter is better)
        # We keep the first 9 after sorting by length within similar priority
        # For now, just take the first 9 — they're already in priority order from the JSON
        unique_skills = unique_skills[:max_slots]
    
    return unique_skills


# ---------------------------------------------------------------------------
# STEP 5: Format as a 3-column layout
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

def process_single_file(filepath, print_output=True, save_to_file=False):
    """
    Process a single tailored_skills JSON file.
    
    Args:
        filepath: Path to the JSON file
        print_output: If True, print to console
        save_to_file: If True, save to skills_sections/ directory
    
    Returns the formatted skills section as a string.
    """
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
    
    # Select top 9
    top_skills = select_top_skills(skills)
    
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
    """
    files = sorted(glob.glob(os.path.join(TAILORED_SKILLS_DIR, "*.json")))
    
    if not files:
        print(f"No JSON files found in {TAILORED_SKILLS_DIR}/")
        return
    
    print(f"Found {len(files)} tailored skills file(s)")
    print("=" * 80 + "\n")
    
    for filepath in files:
        process_single_file(filepath, print_output=True, save_to_file=True)
    
    print("=" * 80)
    print(f"Done! Check {OUTPUT_DIR}/ for formatted skills sections.")


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
