# CV Summarizer

A tool for extracting structured information from CV/resume files using LlamaIndex and OpenAI.

## Overview

This project implements an automated CV parsing workflow that:
1. Extracts text from PDF or DOCX CV files
2. Uses LLM (OpenAI via LlamaIndex) to analyze the CV content
3. Extracts structured information (name, specialization, languages, skills, etc.)
4. Validates the data with Pydantic
5. Saves results as CSV files

## Project Structure

```

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd cv_summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

## Usage

### Using the standalone script

For quick processing of a single CV:

```bash
python cv_parser.py path/to/your/cv.pdf -o output/summary.csv
```

## Extracted Information

The tool extracts and structures the following information:

- **Name-Lastname**: The candidate's full name
- **Specialization**: Primary professional/academic area
- **Spoken-Languages**: List of languages the candidate speaks
- **Programming-Languages**: List of programming languages the candidate knows
- **Summary**: Brief overview of the candidate's background and skills

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt):
  - llama-index
  - openai
  - pydantic
  - pandas
  - python-dotenv
  - pypdf
  - python-docx

## License

NONE

## Contributing

NONE