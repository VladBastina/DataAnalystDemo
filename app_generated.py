import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import seaborn as sns
import plotly.graph_objects as go
import json
import pdfkit
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import html
import re
from openai import OpenAI
from io import StringIO
from pathlib import Path
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
csv_path = "SalesData.csv"


if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        exit(1)

def get_csv_sample(csv_path, sample_size=5):
    """Reads a CSV file and returns column info, a sample, and the DataFrame."""
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    return df.dtypes.to_string(), sample_df.to_string(index=False), df

column_info, sample_str, _ = get_csv_sample(csv_path)

# @observe()
def chat(response_text):
    return json.loads(response_text)  # Directly parse the JSON

def generate_code(question, column_info, sample_str, csv_path, model_name="gpt-4o"):
    """Asks OpenAI to generate Pandas code for a given question."""
    prompt = f"""You are a highly skilled Python data analyst with expert-level proficiency in Pandas. Your task is to write **concise, correct, and efficient** Pandas code to answer a specific question about data contained within a CSV file.  The code you generate must be self-contained, directly executable, and produce the correct numerical output or DataFrame structure.

**CSV File Information:**

*   **Path:** '{csv_path}'
*   **Column Information:** (This tells you the names and data types of the columns)
    ```
    {column_info}
    ```
*   **Sample Data:** (This gives you a glimpse of the data's structure. Note the European date format DD/MM/YYYY)
    ```
    {sample_str}
    ```

**Strict Requirements (Follow these EXACTLY):**
0. **Multi-part Questions:**
    * If the user asks a multi-part question, **reformat it** to process each part correctly while maintaining the original meaning. **Do not change the intent** of the question.
    * **For multi-part questions**, the code should reflect how each part of the question is handled. You must ensure that each part is processed and combined correctly at the end.
    * **Print a statement** explaining how you processed the multi-part question, e.g., `print("Question was split into parts for processing.")`.

1.  **Load Data and Parse Dates:**  Your code *MUST* begin with the following line to load the data, correctly parsing *ALL* potential date columns:
    ```python
    import pandas as pd
    df = pd.read_csv('{csv_path}', parse_dates=['Order Date'])
    ```
    Do *NOT* modify this line. The `parse_dates` argument is *critical* for correct date handling.

2.  **Imports:** Do *NOT* import any libraries other than pandas (which is already imported as `pd`). Do *NOT* use `numpy` or `datetime` directly, unless it is used within the context of parsing in read_csv.  Pandas is sufficient for all tasks.

3.  **Output:**
    *   Store your final answer in a variable named `result`.
    *   Print the `result` variable using `print(result)`.
    *   Do *NOT* use `display()`.
    *   The output must be a Pandas DataFrame, Series, or a single value, as appropriate for the question. If it's a DataFrame or Series, ensure the index is reset where appropriate (e.g., after a `groupby()` followed by `.size()`).

4.  **Conciseness and Style:**
    *   Write the *most concise* and efficient Pandas code possible.
    *   Use method chaining (e.g., `df.groupby(...).sum().sort_values().head()`) whenever possible and appropriate.
    *   Avoid unnecessary intermediate variables unless they *significantly* improve readability.
    *   Use clear and understandable variable names for filtered dataframes, (for example: df_2019, df_filtered etc)
    *   If calculating a percentage or distribution, combine operations efficiently, ideally in a single chained expression.

5.  **Correctness:** Your code *MUST* be syntactically correct Python and *MUST* produce the correct answer to the question. Double-check your logic, especially when grouping and aggregating. Pay close attention to the wording of the question.

6. **Date and Time Conditions (Implicit Filtering):**
    *   **Any question that refers to dates, time periods, months, years, or uses phrases like "issued in," "policies from," "between [dates]," etc., *MUST* filter the data using the `DATA_SEM_OFERTA` column.** This is the *implied* date column for policy issuance. Do *NOT* ask the user which column to use; assume `DATA_SEM_OFERTA`.
    * When filtering dates, use combined boolean conditions for efficiency, e.g., `df[(df['Order Date'].dt.year == 2019) & (df['Order Date'].dt.month == 12)]` rather than separate filtering steps.

7.  **Column Names:** Use the *exact* column names provided in the "CSV Column Information." Pay close attention to capitalization, spaces, and any special characters.

8.  **No Explanations:** Output *ONLY* the Python code. Do *NOT* include any comments, explanations, surrounding text, or markdown formatting (like ```python).  Just the code.

9. **Aggregation (VERY IMPORTANT):** When the question asks for:
    * "top N" or "first N"
    * "most frequent"
    *   "highest/lowest" (after grouping)
    * "average/sum/count per [group]"
    * **Calculate Percentage**: When percentage is asked, compute the correct percentage value

    You *MUST* perform a `groupby()` operation *BEFORE* sorting or selecting the top N values.  The correct order is:
    1.  Filter the DataFrame (if needed, using boolean indexing).
    2.  Group by the appropriate column(s) using `.groupby()`.
    3.  Apply an aggregation function (e.g., `.sum()`, `.mean()`, `.size()`, `.count()`, `.median()`).
    4.  *Then*, sort (if needed) using `.sort_values()` and/or select the top N (if needed) using `.nlargest()` or `.head()`.

10. **Error Handling:** Assume the CSV file exists and is correctly formatted.  You do *not* need to write any explicit error handling code.

11. **Clarity:** Use clear and meaningful variable names if you create intermediate dataframes, but prioritize conciseness.
**Column Usage Guidance:**

                                                        
13. primele means .nlargest and ultimele means .nsmallest
* Use *Product* when referring to specific items sold (e.g., "most popular product," "top-selling product").
* Use *City* when grouping or summarizing sales by location (e.g., "which city had the highest revenue?").
* Use *Order* Date for any time-based filtering (e.g., "sales in December," "transactions between January and March").
* Use *Sales* for financial aggregations (e.g., total revenue, average sale per transaction).
* Use *Quantity* Ordered when analyzing product demand (e.g., "most sold product in terms of units").
* Use *Hour* to analyze time-based trends (e.g., "which hour has the highest number of purchases?").

**Question:**
{question}
"""

    response = client.chat.completions.create(model=model_name,
    temperature=0,  # Keep temperature at 0 for consistent, deterministic code
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates Python code."},
        {"role": "user", "content": prompt}
    ])

    code_to_execute = response.choices[0].message.content.strip()
    code_to_execute = code_to_execute.replace("```python", "").replace("```", "").strip()

    return code_to_execute


def execute_code(generated_code, csv_path):
    """Executes the generated Pandas code and captures the output."""
    local_vars = {"pd": pd, "__file__": csv_path}
    exec(generated_code, {}, local_vars)
    return local_vars.get("result")

def generate_plot_code(question, dataframe, model_name="gpt-4o"):
    """Asks OpenAI to generate plotting code based on the question and dataframe."""
    
    # Convert dataframe to string representation
    df_str = dataframe.to_string(index=False)
    df_json = dataframe.to_json(orient="records")
    
    prompt = f"""You are a data visualization expert. Create Python code to visualize the data below based on the user's question. The visualizations must comprehensively represent *all* the information returned by the query to effectively answer the question.

**User Question:**
{question}

**Data (first few rows):**
```
{df_str}
```

**Data (JSON format):**
```json
{df_json}
```

**Requirements:**
1. Create 4-7 different, meaningful visualizations that collectively represent all aspects of the data returned by the query, ensuring no key information is omitted.
2. Ensure each visualization is simple, clear, and directly tied to a specific part of the data or question, while together they cover the full scope of the result.
3. Use ONLY Matplotlib and Seaborn (avoid Plotly to prevent compatibility issues).
4. Include proper titles, labels, and legends for clarity, reflecting the specific data being visualized.
5. Use appropriate color schemes that are visually appealing and accessible (e.g., colorblind-friendly palettes like Seaborn's 'colorblind').
6. Return a list of tuples containing the plot title and the base64-encoded image.
7. Make sure to close all plt figures with plt.close() after adding each to the plots list to prevent memory issues.
8. If the data includes categories (e.g., sucursale, produse, pachete), ensure these are fully represented across the plots (e.g., bar charts, pie charts, or grouped visuals).
9. If the data includes numerical values (e.g., sales, totals), use appropriate plot types (e.g., bar, line, or scatter) to show trends, comparisons, or distributions.
10. If the question involves time periods, ensure at least one visualization reflects the temporal aspect using the relevant date information.

**Output Format:**
Your code should ONLY include a function called `create_plots(data)` that takes a pandas DataFrame as input and returns a list of tuples containing the plot titles and the base64-encoded images.

Return only the function definition without any explanations, imports, or additional code. Do NOT include any Streamlit-specific code.
"""

    response = client.chat.completions.create(model=model_name,
    temperature=0.2,  # Slightly higher temperature for creative visualizations
    messages=[
        {"role": "system", "content": "You are a data visualization expert who creates Python code for plotting data."},
        {"role": "user", "content": prompt}
    ])

    plot_code = response.choices[0].message.content.strip()
    plot_code = plot_code.replace("```python", "").replace("```", "").strip()

    return plot_code

def execute_plot_code(plot_code, result_df):
    """Executes the generated plotting code and captures the outputs."""
    try:
        # Create a dictionary with all the necessary imports
        globals_dict = {
            "pd": pd,
            "plt": plt,
            "px": px,
            "sns": sns,
            "go": go,
            "io": io,
            "base64": base64,
            "np": __import__('numpy'),
            "plotly": __import__('plotly')
        }
        
        # Create a local variables dictionary with the data
        local_vars = {
            "data": result_df
        }
        
        # Define the helper functions first
        helper_code = """
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return img_str

def plotly_to_base64(fig):
    # For Plotly figures, convert to image bytes and then to base64
    img_bytes = fig.to_image(format="png", scale=2)
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return img_str
"""
        
        # Execute the helper functions first
        exec(helper_code, globals_dict, local_vars)
        
        # Then execute the plot code
        exec(plot_code, globals_dict, local_vars)
        
        # Get the plots from the create_plots function
        if "create_plots" in local_vars:
            plots = local_vars["create_plots"](result_df)
            return plots
        elif "plots" in local_vars:
            return local_vars["plots"]
        else:
            return []
    except Exception as e:
        st.error(f"Error executing plot code: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)

def generate_pdf(query, response_text, chat_response, plots):
    query = html.unescape(query)
    response_text = html.unescape(response_text)
    escaped_query = html.escape(query)
    escaped_response_text = html.escape(response_text)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ro">
    <head>
        <title>Data Analysis Report</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #1f77b4; text-align: center; }}
            h3 {{ color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
            h4 {{ color: #2980b9; }}
            p {{ line-height: 1.6; background-color: #fff; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            pre {{ background-color: #ecf0f1; padding: 10px; border-radius: 5px; font-size: 12px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; page-break-inside: avoid; }}
            th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            td {{ background-color: #fff; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; page-break-inside: avoid; }}
            .section {{ margin-bottom: 20px; }}
            .no-break {{ page-break-inside: avoid; }}
            .powered-by {{ text-align: center; margin-top: 20px; font-size: 10px; color: #777; }}
            .logo {{ height: 100px; }}
        </style>
    </head>
    <body>
    <h1>Data Analysis Agent Interface</h1>
    <div class="section no-break"><h3>Query</h3><p>{escaped_query}</p></div>
    <div class="section no-break"><h3>Response</h3><p>{escaped_response_text}</p></div>
    <div class="section no-break">
        <h3>Raw Structured Response</h3>
        <h4>Metadata</h4><pre>{json.dumps(chat_response["metadata"], indent=2, ensure_ascii=False)}</pre>
        <h4>Data</h4>{pd.DataFrame(chat_response["data"]).to_html(index=False, classes="no-break", escape=False)}
    </div>
    <div class="section"><h3>Plots</h3>{"".join([f'<div class="no-break"><h4>{name}</h4><img src="data:image/png;base64,{base64}"/></div>' for name, base64 in plots])}</div>
    <div class="powered-by">Powered by <img src="data:image/png;base64,{get_zega_logo_base64()}" class="logo"></div>
    </body></html>
    """

    html_file = "temp.html"
    sanitized_query = sanitize_filename(query)
    os.makedirs("./exported_pdfs", exist_ok=True)
    pdf_file = f"./exported_pdfs/{sanitized_query}.pdf"

    try:
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        options = {'encoding': "UTF-8", 'custom-header': [('Content-Type', 'text/html; charset=UTF-8')], 'no-outline': None}
        pdfkit.from_file(html_file, pdf_file, options=options)
        os.remove(html_file)
    except Exception as e:
        raise
    return pdf_file

def get_zega_logo_base64():
    try:
        with open("zega_logo.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    except Exception as e:
        raise
    
def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    try:
        css_path = Path(__file__).parent / file_name
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        # st.info(f"Loaded CSS: {file_name}") # Optional: uncomment for debugging
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_name}. Make sure it's in the same directory as app.py.")
    except Exception as e:
        st.error(f"Error loading CSS file {file_name}: {e}")
        
load_css("style.css")
        
# Streamlit Interface
st.title("Data Analysis Agent Interface")

st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        Powered by <img src="data:image/png;base64,{get_zega_logo_base64()}" style="height: 100px;">
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("Sample Questions")

sample_questions = [
   "Top 5 cities with the highest sales?",
    "Bottom 3 products by total sales?",
    "Top 10 products with reference to items sold?",
    "Top 10 products with reference to total sums sold?"
]

selected_question = st.sidebar.selectbox("Select a sample question:", sample_questions)


with open(csv_path, "rb") as f:
    st.sidebar.download_button(
        label="Download CSV",
        data=f,
        file_name="data.csv",
        mime="text/csv"
    )
    
user_query = st.text_area("Please write one question at a time.", value=selected_question, height=100)

def process_query():
    try:
        if len(user_query.strip()) == 0:
            st.error("Please enter a query.")
            return
        elif not re.match("^[a-zA-Z0-9 ]*$", user_query):
            st.error("Special characters are not allowed. Please use only letters and numbers.")
            return
        # Step 1: Generate and execute code to get the data
        generated_code = generate_code(user_query, column_info, sample_str, csv_path)
        result = execute_code(generated_code, csv_path)

        # Convert result to DataFrame if it's not already
        if isinstance(result, pd.DataFrame):
            result_df = result
        elif isinstance(result, pd.Series):
            result_df = result.reset_index()
        elif isinstance(result, list):
            if all(isinstance(item, dict) for item in result):
                result_df = pd.DataFrame(result)
            else:
                result_df = pd.DataFrame({"value": result})
        else:
            result_df = pd.DataFrame({"value": [result]})

        # Step 2: Generate and execute plotting code
        plot_code = generate_plot_code(user_query, result_df)
        plots = execute_plot_code(plot_code, result_df)

        # Prepare the chat response
        if isinstance(result, pd.DataFrame):
            chat_response = {
                "metadata": {"query": user_query, "unit": "", "plot_types": []},
                "data": result.to_dict(orient='records'),
                "csv_data": result.to_dict(orient='records'),
            }
        elif isinstance(result, pd.Series):
            result = result.reset_index()
            chat_response = {
                "metadata": {"query": user_query, "unit": "", "plot_types": []},
                "data": result.to_dict(orient='records'),
                "csv_data": result.to_dict(orient='records'),
            }
        elif isinstance(result, list):
            if all(isinstance(item, (int, float)) for item in result):
                chat_response = {
                    "metadata": {"query": user_query, "unit": "", "plot_types": []},
                    "data": [{"category": str(i), "value": v} for i, v in enumerate(result)],
                    "csv_data": [{"category": str(i), "value": v} for i, v in enumerate(result)],
                }
            elif all(isinstance(item, dict) for item in result):
                chat_response = {
                    "metadata": {"query": user_query, "unit": "", "plot_types": []},
                    "data": result,
                    "csv_data": result,
                }
            else:
                st.warning("Result is a list with mixed data types. Please inspect.")
                return
        else:
            chat_response = {
                "metadata": {"query": user_query, "unit": "", "plot_types": []},
                "data": [{"category": "Result", "value": result}],
                "csv_data": [{"category": "Result", "value": result}],
            }

        # Display the query and data
        st.markdown(f"<h3 style='color: #2e86de;'>Question:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #2e86de;'>{user_query}</p>", unsafe_allow_html=True)
        st.write("-" * 200)

        # Initially hide the code
        with st.expander("Show the generated data code"):
            st.code(generated_code, language="python")
        
        with st.expander("Show the generated plotting code"):
            st.code(plot_code, language="python")
        
        st.write("-" * 200)

        # Display the data
        st.markdown("### Data:")
        st.dataframe(result_df)
        st.write("-" * 200)

        # Display the plots
        st.markdown("### Visualizations:")
        for name, base64_img in plots:
            st.markdown(f"#### {name}")
            st.markdown(f'<img src="data:image/png;base64,{base64_img}" style="max-width:100%">', unsafe_allow_html=True)
            st.write("-" * 100)

        # Store the data for PDF generation
        st.session_state["query"] = user_query
        st.session_state["response_text"] = str(result)
        st.session_state["chat_response"] = chat_response
        st.session_state["plots"] = plots
        st.session_state["generated_code"] = generated_code
        st.session_state["plot_code"] = plot_code

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())

if st.button("Submit"):
    with st.spinner("Processing query..."):
        try:
            process_query()
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())

if "chat_response" in st.session_state:
    if st.button("Download PDF"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_file = generate_pdf(
                    st.session_state["query"],
                    st.session_state["response_text"],
                    st.session_state["chat_response"],
                    st.session_state["plots"]
                )
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                sanitized_query = sanitize_filename(st.session_state["query"])
                st.download_button(
                    label="Click Here to Download PDF",
                    data=pdf_data,
                    file_name=f"{sanitized_query}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")