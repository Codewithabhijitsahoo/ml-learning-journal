#!/usr/bin/env python3
"""
AI-Powered ML Learning Journal - Daily Note Generator

This script automates the generation of daily machine learning notes. It reads topics
from a text file, identifies the next uncompleted topic, calls the Gemini API to
generate a detailed markdown note, and saves the file with a date prefix.

Requirements:
- Python 3.11+
- google-genai
"""

import argparse
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Set, Tuple, Optional

# Setup logging with a professional format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try importing the Google GenAI SDK
try:
    from google import genai
    from google.genai import errors
    from google.genai import types
except ImportError:
    logger.error(
        "Failed to import 'google-genai'. "
        "Please install it using 'pip install google-genai'."
    )
    sys.exit(1)


def slugify(text: str) -> str:
    """
    Convert a topic name to a URL-friendly, filesystem-safe lowercase slug.
    
    Args:
        text: The raw topic string.
        
    Returns:
        A cleaned slug string.
    """
    # Convert to lowercase and replace non-alphanumeric with spaces
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    # Replace spaces and multiple hyphens with a single hyphen
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")


def get_covered_topics(notes_dir: Path) -> Set[str]:
    """
    Scan the notes directory to extract slugs of topics already generated.
    
    Filenames are expected to follow the pattern: YYYY-MM-DD-topic-slug.md
    
    Args:
        notes_dir: Path to the directory containing notes.
        
    Returns:
        A set of already covered topic slugs.
    """
    covered = set()
    if not notes_dir.exists():
        return covered

    # Pattern to match YYYY-MM-DD-slug.md
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-(.+)\.md$")
    for file_path in notes_dir.glob("*.md"):
        match = pattern.match(file_path.name)
        if match:
            covered.add(match.group(1))
            
    return covered


def get_next_topic(topics_file: Path, covered_slugs: Set[str]) -> Optional[Tuple[str, str]]:
    """
    Read topics.txt and return the first topic that is not yet covered.
    
    Args:
        topics_file: Path to topics.txt.
        covered_slugs: A set of already generated topic slugs.
        
    Returns:
        A tuple of (topic_name, topic_slug) if found, otherwise None.
    """
    if not topics_file.exists():
        logger.error(f"Topics file not found at: {topics_file}")
        raise FileNotFoundError(f"Topics file not found at: {topics_file}")

    with open(topics_file, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    for topic in topics:
        slug = slugify(topic)
        if slug not in covered_slugs:
            return topic, slug

    return None


def generate_note_content(topic: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Call the Gemini API using the google-genai library to generate a note for the given topic.
    
    Args:
        topic: The topic name.
        model_name: The Gemini model identifier.
        
    Returns:
        The generated markdown content as a string.
        
    Raises:
        ValueError: If the API returns an empty response.
        Exception: For API errors or connection failures.
    """
    api_key_env = os.environ.get("GEMINI_API_KEY")
    if not api_key_env:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    # Support multiple comma-separated keys for rate-limiting rotation
    api_keys = [k.strip() for k in api_key_env.split(",") if k.strip()]

    system_instruction = (
        "You are an expert AI and Machine Learning educator. Your goal is to write a highly detailed, "
        "comprehensive, and beginner-friendly study note in Markdown format about the requested ML topic. "
        "Do not wrap your final output in markdown code blocks like ```markdown ... ```. Output raw markdown. "
        "Strictly follow the requested headings and ensure the explanation is rich, engaging, and contains practical code and questions."
    )

    prompt = f"""
Write a comprehensive, beginner-friendly, and detailed study note on the topic: "{topic}".
The output must strictly be in markdown format and contain the following exact headers and structure:

# {topic}

## Overview
Provide a clear, high-level, and beginner-friendly introduction to {topic}.

## What Problem It Solves
Explain the core problems or challenges that {topic} addresses. Why is it needed in machine learning?

## How It Works
Detail the step-by-step mechanism of {topic}. Break down the algorithm, training process, or pipeline in simple terms.

## Mathematical Intuition
Explain the mathematical concepts, equations, and logic behind {topic}. Use LaTeX formatting for mathematical expressions:
- Inline math should be wrapped in single dollar signs, like $y = mx + c$.
- Block equations should be wrapped in double dollar signs, like $$f(x) = \\frac{{1}}{{1 + e^{{-x}}}}$$.
Break down the equations so a beginner can understand them.

## Advantages
List the pros and strengths of using {topic} (e.g., in bullet points).

## Disadvantages
List the cons, limitations, and potential pitfalls of {topic} (e.g., in bullet points).

## Real World Applications
Provide 3-5 concrete real-world use cases or industries where {topic} is actively applied.

## Python Example
Provide a complete, standalone, and working Python code snippet demonstrating {topic}.
- Use popular libraries such as scikit-learn, numpy, pandas, or matplotlib.
- The example should generate or load a dummy dataset, fit the model/operation, make predictions/results, and evaluate or print the output.
- Write clean Python code with helpful comments.

## Interview Questions
Provide a list of at least 10 relevant technical interview questions about {topic}, complete with comprehensive, detailed answers.

## Quiz
Provide a multiple-choice quiz with at least 5 conceptual questions.
- Each question must have 4 options: A), B), C), and D).
- At the very end of the Quiz section, provide an Answer Key with brief explanations for each correct option.

## Further Reading
List at least 3 high-quality learning resources, research papers, or documentation links (e.g., official docs, textbook chapters).

Ensure that all sections are highly detailed, informative, and free of placeholder text. Do not wrap the entire output in ```markdown ... ```. Output raw markdown.
"""

    fallback_prompt = f"""
Write a concise study note on the topic: "{topic}".
The output must strictly be in markdown format and contain the following exact headers and structure:

# {topic}

## Overview
Provide a clear, high-level introduction to {topic}.

## What Problem It Solves
Explain the core problems that {topic} addresses.

## How It Works
Detail the mechanism of {topic} in simple terms.

## Mathematical Intuition
Briefly explain the key mathematical concepts and logic behind {topic} using LaTeX.

## Advantages
List the main advantages of using {topic}.

## Disadvantages
List the key limitations of {topic}.

## Real World Applications
Provide 2-3 real-world use cases where {topic} is applied.

## Python Example
Provide a short, standalone Python code snippet demonstrating {topic}.

## Interview Questions
Provide a list of 3 key technical interview questions about {topic}, with answers.

## Quiz
Provide a multiple-choice quiz with 2 conceptual questions, with answers.

## Further Reading
List 2-3 learning resources or documentation links.

Ensure that all sections are concise and informative. Do not wrap the entire output in ```markdown ... ```. Output raw markdown.
"""

    max_retries = 6
    base_delay = 5.0  # seconds
    response = None

    for attempt in range(1, max_retries + 1):
        # Rotate API key based on the attempt number
        current_key = api_keys[(attempt - 1) % len(api_keys)]
        
        # Switch to fallback prompt (retrieving less content) on attempt 3 or later to avoid token/rate limits
        current_prompt = prompt if attempt < 3 else fallback_prompt
        if attempt >= 3:
            logger.warning(
                f"Attempt {attempt}: Switching to simplified/concise fallback prompt (retrieving less content) "
                f"to resolve or avoid API quota/rate limit issues."
            )
        try:
            logger.info(
                f"Sending request to Gemini API for topic: '{topic}' "
                f"using model: '{model_name}' (Attempt {attempt}/{max_retries})..."
            )
            # Initialize Client dynamically with the rotated key
            client = genai.Client(api_key=current_key)
            response = client.models.generate_content(
                model=model_name,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,  # Lower temperature for educational accuracy
                )
            )
            break  # Successfully generated content, exit the retry loop
        except errors.APIError as e:
            # Check if error is due to rate limits (HTTP 429) or transient server errors (HTTP 500/503/504)
            status_code = getattr(e, "code", None)
            is_rate_limit = status_code in [429, 500, 503, 504] or "quota" in str(e).lower() or "exhausted" in str(e).lower()

            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Gemini API rate limit or transient error hit (status code: {status_code}). "
                    f"Retrying in {delay:.1f} seconds... Error: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(f"Gemini API error occurred on attempt {attempt}: {e}")
                raise e
        except Exception as e:
            # Catch transient network/client issues and retry
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Transient network or client error: {e}. Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Unexpected error occurred on attempt {attempt}: {e}")
                raise e

    if not response or not response.text:
        logger.error("Received empty response from Gemini API.")
        raise ValueError("Received empty response from Gemini API.")

    # Clean the response in case the model ignored instructions and wrapped in code blocks
    content = response.text.strip()
    if content.startswith("```markdown"):
        content = content[11:].strip()
    if content.startswith("```html"):
        content = content[7:].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content


def save_note(notes_dir: Path, topic_slug: str, content: str, date_str: Optional[str] = None) -> Path:
    """
    Save the generated content into a markdown file in the notes directory.
    
    Args:
        notes_dir: Path to save the note.
        topic_slug: Slug of the topic.
        content: Markdown content.
        date_str: Optional date string in YYYY-MM-DD format (defaults to today).
        
    Returns:
        The Path to the saved file.
    """
    notes_dir.mkdir(parents=True, exist_ok=True)
    
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        
    filename = f"{date_str}-{topic_slug}.md"
    file_path = notes_dir / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    logger.info(f"Successfully generated and saved note: {file_path}")
    return file_path


# =====================================================================
# Placeholders for Future Enhancements (as requested in specifications)
# =====================================================================

def generate_daily_quiz_standalone(topic: str) -> None:
    """
    Placeholder: Generates a dedicated daily quiz file or interactive quiz for the topic.
    To be implemented in future phase.
    """
    logger.info(f"[Future Enhancement] Generating standalone quiz for: {topic} (Not implemented)")
    pass


def generate_daily_interview_questions_standalone(topic: str) -> None:
    """
    Placeholder: Generates a standalone PDF/Markdown flashcard set of interview questions.
    To be implemented in future phase.
    """
    logger.info(f"[Future Enhancement] Generating interview flashcards for: {topic} (Not implemented)")
    pass


def generate_daily_coding_exercise(topic: str) -> None:
    """
    Placeholder: Generates a Jupyter notebook (.ipynb) containing a guided coding exercise for the topic.
    To be implemented in future phase.
    """
    logger.info(f"[Future Enhancement] Generating Jupyter notebook coding exercise for: {topic} (Not implemented)")
    pass


def generate_daily_ml_project(topic: str) -> None:
    """
    Placeholder: Generates a small end-to-end project directory for the topic.
    To be implemented in future phase.
    """
    logger.info(f"[Future Enhancement] Generating end-to-end ML project folder for: {topic} (Not implemented)")
    pass


def generate_weekly_summary(week_number: int) -> None:
    """
    Placeholder: Aggregates the last 7 notes and generates a weekly summary, review guide, and aggregate quiz.
    To be implemented in future phase.
    """
    logger.info(f"[Future Enhancement] Generating weekly summary for week {week_number} (Not implemented)")
    pass


# =====================================================================
# Main Flow
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a daily ML learning note using Gemini API.")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--notes-dir",
        type=str,
        default="notes",
        help="Directory to save generated notes (default: notes)"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default="topics.txt",
        help="File containing list of topics (default: topics.txt)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Generate a new note even if one was already generated today."
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=5,
        help="Maximum number of notes to generate per day (default: 5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find the next topic and print it, but do not call the API or write any files."
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.resolve()
    notes_path = base_dir / args.notes_dir
    topics_path = base_dir / args.topics_file

    logger.info("Starting Daily Note Generator...")
    
    # 1. Check if the limit of daily notes has already been reached (to avoid accidental duplicates/excess runs)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    if not args.force and not args.dry_run:
        today_notes = list(notes_path.glob(f"{today_str}-*.md"))
        if len(today_notes) >= args.max_per_day:
            logger.info(f"Already generated {len(today_notes)} note(s) today (limit: {args.max_per_day}).")
            logger.info("Use --force to override this and generate another note.")
            sys.exit(0)
            
    # 2. Get list of covered topics
    try:
        covered_slugs = get_covered_topics(notes_path)
        logger.info(f"Found {len(covered_slugs)} already covered topics in notes directory.")
    except Exception as e:
        logger.error(f"Error scanning notes directory: {e}")
        sys.exit(1)
        
    # 3. Determine the next topic
    try:
        next_topic_info = get_next_topic(topics_path, covered_slugs)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading topics file: {e}")
        sys.exit(1)
        
    if not next_topic_info:
        logger.warning("All topics in topics.txt have already been covered!")
        # Instead of failing the GitHub Action, exit successfully with warning
        sys.exit(0)
        
    topic_name, topic_slug = next_topic_info
    logger.info(f"Next topic identified: '{topic_name}' (Slug: '{topic_slug}')")
    
    if args.dry_run:
        logger.info("[Dry Run] Skipping API call and file generation.")
        sys.exit(0)
        
    # 4. Generate content from Gemini API
    try:
        content = generate_note_content(topic_name, model_name=args.model)
    except Exception as e:
        logger.error(f"Failed to generate content: {e}")
        sys.exit(1)
        
    # 5. Save the note
    try:
        saved_file = save_note(notes_path, topic_slug, content, today_str)
        
        # Write to GITHUB_OUTPUT environment file if available
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            try:
                with open(github_output, "a", encoding="utf-8") as f:
                    f.write(f"note_file={saved_file.relative_to(base_dir)}\n")
                    f.write(f"topic_name={topic_name}\n")
                logger.info("Wrote outputs to GITHUB_OUTPUT.")
            except Exception as e:
                logger.error(f"Failed to write to GITHUB_OUTPUT: {e}")
                
    except Exception as e:
        logger.error(f"Failed to save note: {e}")
        sys.exit(1)
        
    # 6. Trigger placeholder logging for context
    # These functions demonstrate the interface placeholders required
    generate_daily_quiz_standalone(topic_name)
    generate_daily_interview_questions_standalone(topic_name)
    generate_daily_coding_exercise(topic_name)
    generate_daily_ml_project(topic_name)
    
    # Example logic for weekly summaries (e.g. if covered count is a multiple of 7)
    if len(covered_slugs) + 1 >= 7 and (len(covered_slugs) + 1) % 7 == 0:
        week_num = (len(covered_slugs) + 1) // 7
        generate_weekly_summary(week_num)

    logger.info("Process finished successfully.")


if __name__ == "__main__":
    main()
