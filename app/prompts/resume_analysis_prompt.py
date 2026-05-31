"""Prompt construction for resume analysis."""

from __future__ import annotations

from textwrap import dedent


def build_resume_analysis_query(job_description: str) -> str:
    """Build the exact analysis prompt used by the original app."""

    return dedent(
        f"""
        You are an expert ATS Resume Analyzer.

        Analyze the uploaded resume against the given Job Description.

        JOB DESCRIPTION:
        {job_description}

        Provide the following:

        1. ATS Match Percentage
        2. Missing Skills
        3. Technical Strengths
        4. Weak Areas
        5. Resume Improvement Suggestions
        6. Recommended Projects
        7. Possible Interview Questions
        8. Final Hiring Recommendation

        Give the response in professional formatting.
        """
    ).strip()

