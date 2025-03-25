# Data Analysis Agent Interface with Streamlit

This Streamlit application provides an interface for interacting with a data analysis agent powered by OpenAI's language models. It allows users to ask questions about data in a CSV file and receive answers in the form of Pandas code, data tables, and visualizations.  The application also supports generating a PDF report of the analysis.

## Features

*   **Natural Language Queries:** Ask questions in plain English (or Romanian) about the data.
*   **Automatic Code Generation:** The agent generates Pandas code to answer the query.
*   **Data Display:** Results are displayed as interactive DataFrames.
*   **Visualization:**  Generates various plots (bar charts, pie charts, histograms, heatmaps, scatter plots, line plots, box plots, violin plots, area charts, and radar charts) based on the query and data.
*   **PDF Report Generation:**  Download a PDF report containing the query, generated code, data table, and plots.
*   **Syntax-Highlighted Code:**  The generated Python code is displayed in a scrollable, syntax-highlighted code block for easy readability.
*   **Collapsible Code Display:** The generated code is hidden by default, with an expander to reveal it on demand.
*   **Sample Questions:**  Provides a set of sample questions to get started.
*   **Powered by ZEGA.ai:**  Includes ZEGA.ai branding.

## Getting Started

### Prerequisites

*   Python 3.7+
*   An OpenAI API key
*   pdfkit: you need to have wkhtmltopdf installed on your system.
    *   **Windows**: Download and install from [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html). Add the `wkhtmltopdf/bin` directory to your system's PATH.
    *   **macOS**: `brew install wkhtmltopdf`
    *   **Linux (Debian/Ubuntu)**: `sudo apt-get install wkhtmltopdf`
    *   **Linux (CentOS/RHEL)**: `sudo yum install wkhtmltopdf`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    Create the `requirements.txt` and place this in:
    ```
    streamlit
    pandas
    matplotlib
    plotly
    python-dotenv
    langchain
    langchain-experimental
    langchain-openai
    seaborn
    pdfkit
    openai
    ```

3.  **Create a `.env` file:**

    Create a file named `.env` in the root directory of your project.  Add your OpenAI API key to this file:

    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    Replace `your_openai_api_key_here` with your actual API key.

4.  **Place the CSV data file:**

    Place the `asig_sales_31012025.csv` file in the same directory as your script.  If you use a different CSV file, update the `csv_path` variable in the script.

5. **Place Zega logo**
    Place the `zega_logo.png` into the folder.

### Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run your_script_name.py
    ```
    Replace `your_script_name.py` with the name of your Python script.

2.  **Interact with the app:**

    *   Select a sample question from the sidebar or enter your own question in the text area.  Ensure you ask only one question at a time.
    *   Click the "Submit" button.
    *   The results (data table and plots) will be displayed.
    *   Click the "Show the code" expander to view the generated Pandas code.
    *   Click the "Download PDF" button to generate a PDF report.

## File Structure

*   **`your_script_name.py`:**  The main Streamlit application script.
*   **`.env`:**  Contains your OpenAI API key (should *not* be committed to Git).
*   **`requirements.txt`:**  Lists the required Python packages.
*   **`asig_sales_31012025.csv`:**  The CSV data file (or your custom data file).
*  **`zega_logo.png`:**  Zega logo.
*   **`exported_pdfs/`:**  A directory (created automatically) where generated PDF reports are saved.
*   **`README.md`:** This file.

## Important Notes

*   **Date Format:** The script is specifically configured to handle dates in the European DD/MM/YYYY format.  Ensure your CSV data uses this format.  The `parse_dates` argument in `pd.read_csv` is crucial for correct date handling.
*   **OpenAI API Key:** Keep your OpenAI API key secure.  Do *not* commit the `.env` file to your Git repository. Add `.env` to your `.gitignore` file.
*   **Error Handling:** The script includes basic error handling (checking for the CSV file), but you might want to add more robust error handling for production use.
*   **wkhtmltopdf:** Ensure `wkhtmltopdf` is correctly installed and accessible in your system's PATH for PDF generation to work.
*   **Prompt Engineering:**  The quality of the generated code depends heavily on the prompt used in the `generate_code` function.  The provided prompt is highly detailed and includes specific instructions for the agent.  You may need to adjust the prompt if you encounter issues or use a different CSV file with different column names or data structures.
* **One Question:** The app is designed to process one question at a time. Asking multiple questions in a single input may lead to unexpected behavior.


