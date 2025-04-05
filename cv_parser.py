#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CV Parser: Extract structured information from CVs using LlamaIndex and OpenAI.

This script processes PDF or DOCX CV files, extracts text, uses an LLM to parse
structured information, and saves the results to a CSV file.
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional

# Text extraction libraries
from pypdf import PdfReader
import docx

# LLM and data structuring
import openai
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field

# Data handling and storage
import pandas as pd
from dotenv import load_dotenv

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Define the Pydantic model for structured output
class CVSummary(BaseModel):
    """Structured representation of CV information."""
    name_lastname: str = Field(..., alias="Name-Lastname")
    specialization: str = Field(..., alias="Specialization")
    spoken_languages: List[str] = Field(..., alias="Spoken-Languages")
    programming_languages: List[str] = Field(..., alias="Programming-Languages")
    summary: str = Field(..., alias="Summary")

    class Config:
        populate_by_name = True


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    logging.info(f"Extracting text from PDF: {file_path}")
    
    text = ""
    reader = PdfReader(file_path)
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text.strip()


def extract_text_from_docx(file_path: Path) -> str:
    """Extract text from a DOCX file."""
    logging.info(f"Extracting text from DOCX: {file_path}")
    
    text = ""
    doc = docx.Document(file_path)
    
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    return text.strip()


def extract_text_from_cv(file_path: Path) -> str:
    """Extract text from a CV file (PDF or DOCX)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path.suffix.lower()
    
    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please provide a PDF or DOCX file.")


def parse_cv_with_llm(cv_text: str, api_key: str) -> dict:
    """
    Use LlamaIndex with OpenAI to parse structured information from CV text.
    """
    logging.info("Parsing CV text with LLM...")
    
    # Initialize the OpenAI LLM
    llm = OpenAI(api_key=api_key, model="gpt-3.5-turbo")
    
    # Define the prompt for CV parsing
    prompt = f"""
    You are an intelligent CV parser. Given the full text content of a CV, extract structured information and return it in the following JSON format:

    {{
      "Name-Lastname": String,
      "Specialization": String,
      "Spoken-Languages": List[String],
      "Programming-Languages": List[String],
      "Summary": String
    }}

    Extraction Rules:
    1. "Name-Lastname": Extract the full name from the top of the document or contact section.
    2. "Specialization": Identify the main professional or academic area (e.g., AI, Backend Development, etc.).
    3. "Spoken-Languages": List all spoken languages mentioned.
    4. "Programming-Languages": Extract programming languages such as Python, Java, C++, etc.
    5. "Summary": Write a concise summary (2–4 sentences) of the candidate's background, skills, and experience.

    Only return the JSON object without any additional commentary.

    CV Text:
    {cv_text}
    """
    
    # Call the LLM with the prompt
    response = llm.complete(prompt)
    
    # Extract and parse the JSON response
    import json
    try:
        # Clean the response - sometimes the LLM might include markdown code fences
        clean_response = response.text.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:] # Remove ```json
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3] # Remove ```
        
        parsed_data = json.loads(clean_response)
        return parsed_data
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse LLM response as JSON: {e}")
        logging.debug(f"Raw response: {response.text}")
        raise


def main():
    """Main function to process CV files."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse CV files (PDF/DOCX) and extract structured information")
    parser.add_argument("cv_file", help="Path to the CV file (PDF or DOCX)")
    parser.add_argument("-o", "--output", default="cv_summary.csv", help="Path to save the output CSV file")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        return
    
    try:
        # Convert input and output paths to Path objects
        cv_file_path = Path(args.cv_file)
        output_path = Path(args.output)
        
        # Create output directory if it doesn't exist
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        
        # Extract text from the CV file
        cv_text = extract_text_from_cv(cv_file_path)
        logging.info(f"Extracted {len(cv_text)} characters from {cv_file_path.name}")
        
        # Parse the CV text with the LLM
        parsed_data = parse_cv_with_llm(cv_text, api_key)
        
        # Validate the parsed data with Pydantic
        cv_summary = CVSummary.parse_obj(parsed_data)
        logging.info("Successfully parsed and validated CV data")
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame([cv_summary.dict(by_alias=True)])
        df.to_csv(output_path, index=False)
        
        logging.info(f"CV summary saved to {output_path}")
        print(f"Successfully extracted CV information and saved to {output_path}")
        
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        print(f"Error: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main() 