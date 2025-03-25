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

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
csv_path = "asig_sales_31012025.csv"

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
    df = pd.read_csv('{csv_path}', parse_dates=['HIST_DATE', 'DATA_SEM_OFERTA', 'DATA_STARE_CERERE', 'DATA_IN_OFERTA', 'CTR_DATA_START', 'CTR_DATA_STATUS'], dayfirst=True)
    ```
    Do *NOT* modify this line. The `parse_dates` argument is *critical* for correct date handling, and `dayfirst=True` is absolutely required because dates are in European DD/MM/YYYY format.

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
    *   Use clear and understandable variable names for filtered dataframes, (for example: df_2010, df_filtered etc)
    *   If calculating a percentage or distribution, combine operations efficiently, ideally in a single chained expression.

5.  **Correctness:** Your code *MUST* be syntactically correct Python and *MUST* produce the correct answer to the question. Double-check your logic, especially when grouping and aggregating. Pay close attention to the wording of the question.

6. **Date and Time Conditions (Implicit Filtering):**
    *   **Any question that refers to dates, time periods, months, years, or uses phrases like "issued in," "policies from," "between [dates]," etc., *MUST* filter the data using the `DATA_SEM_OFERTA` column.** This is the *implied* date column for policy issuance. Do *NOT* ask the user which column to use; assume `DATA_SEM_OFERTA`.
    * When filtering dates, use combined boolean conditions for efficiency, e.g., `df[(df['DATA_SEM_OFERTA'].dt.year == 2010) & (df['DATA_SEM_OFERTA'].dt.month == 12)]` rather than separate filtering steps.

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
* Use `CTR_STATUS` when a concise or coded representation of the contract status is needed (e.g., for technical filtering or matching with system data).
* Use `CTR_DESCRIERE_STATUS` when a human-readable description is required (e.g., for distributions, summaries, or grouping by status type, such as "Activ", "Reziliat"). Default to `CTR_DESCRIERE_STATUS` for questions involving totals, distributions, or descriptive analysis unless the question specifies a coded status.
* Use `COD_SUCURSALA` for numerical branch identification (e.g., filtering or joining with other datasets); use `DENUMIRE_SUCURSALA` for human-readable branch names (e.g., grouping or summarizing by branch name).
* Use `COD_AGENTIE` for numerical agency identification; use `DENUMIRE_AGENTIE` for human-readable agency names, preferring the latter for summaries or rankings.
* Use `DATA_SEM_OFERTA` as the implied date column for policy issuance or time-based filtering (e.g., "issued in", "per month"), unless the question specifies another date column.
* Use `PBA_BAZA`, `PBA_ASIG_SUPLIM`, `PBA_TOTAL_SEMNARE_CERERE`, and `PBA_TOTAL_EMITERE_CERERE` for financial aggregations (e.g., sum, mean) based on the specific PBA type mentioned in the question.

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

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return img_str

def plotly_to_base64(fig):
    img_bytes = fig.to_image(format="png", scale=2)
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    return img_str

def generate_plots(metadata, categories, values):
    # Filter numeric values and categories
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    numeric_categories = [c for c, v in zip(categories, values) if isinstance(v, (int, float))]

    if not numeric_values:
        st.warning("No numeric data to plot for this query.")
        return []

    sorted_categories, sorted_values = zip(*sorted(zip(numeric_categories, numeric_values), key=lambda x: x[1], reverse=True))
    plots = []

    if all(isinstance(c, str) for c in categories) and all(isinstance(v, (int, float)) for v in values):
        sorted_categories, sorted_values = zip(*sorted(zip(categories, values), key=lambda x: x[1], reverse=True))

        # Bar Plot (Main plot for string categories and numeric values)
        fig_bar = px.bar(x=sorted_values, y=sorted_categories, orientation="h",
                         labels={"x": "Value", "y": "Category"},
                         title=f"{metadata['query']} (Bar Chart)",
                         color=sorted_values, color_continuous_scale="blues")
        fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_bar)
        plots.append(("Bar Chart (Plotly)", plotly_to_base64(fig_bar)))

    # Numeric plots (only if there are numeric values)
    if any(isinstance(v, (int, float)) for v in values):
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        numeric_categories = [c for c, v in zip(categories, values) if isinstance(v, (int, float))]

        if numeric_values:
            sorted_categories, sorted_values = zip(*sorted(zip(numeric_categories, numeric_values), key=lambda x: x[1], reverse=True))

            # Bar Plot (Plotly)
            fig1 = px.bar(x=sorted_categories, y=sorted_values, labels={"x": "Category", "y": metadata.get("unit", "Value")},
                          title=f"{metadata['query']} (Plotly Bar)", color=sorted_values, color_continuous_scale="blues")
            st.plotly_chart(fig1)
            plots.append(("Bar Plot (Plotly)", plotly_to_base64(fig1)))

            # Pie Chart
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            cmap = plt.get_cmap("tab20c")
            colors = [cmap(i) for i in range(len(sorted_categories))]
            wedges, texts = ax2.pie(sorted_values, labels=None, autopct=None, startangle=140, colors=colors, wedgeprops=dict(width=0.4))
            legend_labels = [f"{cat} ({val / sum(sorted_values):.1%})" for cat, val in zip(sorted_categories, sorted_values)]
            ax2.legend(wedges, legend_labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
            ax2.axis("equal")
            ax2.set_title(f"{metadata['query']} (Pie)", fontsize=16)
            st.pyplot(fig2)
            plots.append(("Pie Chart", fig_to_base64(fig2)))
            plt.close(fig2)

            # Histogram
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.hist(sorted_values, bins=10, color="skyblue", edgecolor="black")
            ax3.set_title(f"Distribution of {metadata['query']} (Histogram)", fontsize=16)
            st.pyplot(fig3)
            plots.append(("Histogram", fig_to_base64(fig3)))
            plt.close(fig3)

            # Heatmap
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            data_matrix = pd.DataFrame({metadata.get("unit", "Value"): sorted_values}, index=sorted_categories)
            sns.heatmap(data_matrix, annot=True, cmap="Blues", ax=ax4, fmt=".1f")
            ax4.set_title(f"{metadata['query']} (Heatmap)", fontsize=16)
            st.pyplot(fig4)
            plots.append(("Heatmap", fig_to_base64(fig4)))
            plt.close(fig4)

            # Scatter Plot
            fig5 = px.scatter(x=sorted_categories, y=sorted_values, title=f"{metadata['query']} (Scatter Plot)",
                              labels={"x": "Category", "y": metadata.get("unit", "Value")})
            st.plotly_chart(fig5)
            plots.append(("Scatter Plot (Plotly)", plotly_to_base64(fig5)))

            # Line Plot
            fig6 = px.line(x=sorted_categories, y=sorted_values, title=f"{metadata['query']} (Line Plot)",
                           labels={"x": "Category", "y": metadata.get("unit", "Value")})
            st.plotly_chart(fig6)
            plots.append(("Line Plot (Plotly)", plotly_to_base64(fig6)))

            # Box Plot
            fig7, ax7 = plt.subplots(figsize=(10, 6))
            ax7.boxplot(sorted_values, vert=False, tick_labels=["Data"], patch_artist=True)
            ax7.set_title(f"{metadata['query']} (Box Plot)", fontsize=16)
            st.pyplot(fig7)
            plots.append(("Box Plot", fig_to_base64(fig7)))
            plt.close(fig7)

            # Violin Plot
            fig8, ax8 = plt.subplots(figsize=(10, 6))
            ax8.violinplot(sorted_values, vert=False, showmeans=True, showextrema=True)
            ax8.set_title(f"{metadata['query']} (Violin Plot)", fontsize=16)
            st.pyplot(fig8)
            plots.append(("Violin Plot", fig_to_base64(fig8)))
            plt.close(fig8)

            # Area Chart
            fig9 = px.area(x=sorted_categories, y=sorted_values, title=f"{metadata['query']} (Area Chart)", labels={"x": "Category", "y": metadata.get("unit", "Value")})
            st.plotly_chart(fig9)
            plots.append(("Area Chart (Plotly)", plotly_to_base64(fig9)))

            # Radar Chart
            fig10 = go.Figure(data=go.Scatterpolar(r=sorted_values, theta=sorted_categories, fill='toself', name=metadata['query']))
            fig10.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title=f"{metadata['query']} (Radar Chart)")

            st.plotly_chart(fig10)
            plots.append(("Radar Chart (Plotly)", plotly_to_base64(fig10)))

    else:
        st.warning("No numeric data to plot for this query.")

    return plots

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
   "Da-mi top cinci sucursale cu vânzări în perioada 01.03.2024-01.04.2024.",
    "Da-mi vânzările defalcate pe produse pentru top cinci sucursale cu vânzări în perioada 01.03.2024-01.04.2024.",
    "Da-mi vânzările defalcate pe pachete pentru top cinci sucursale cu vânzări în perioada 01.03.2024-01.04.2024.",
]

selected_question = st.sidebar.selectbox("Select a sample question:", sample_questions)
user_query = st.text_area("Please write one question at a time.", value=selected_question, height=100)

def process_query():
    try:
        generated_code = generate_code(user_query, column_info, sample_str, csv_path)
        result = execute_code(generated_code, csv_path)

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

        st.markdown(f"<h3 style='color: #2e86de;'>Question:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #2e86de;'>{user_query}</p>", unsafe_allow_html=True)
        st.write("-" * 200)

        # Initially hide the code.
        with st.expander("Show the code"):
            st.code(generated_code, language="python")
        st.write("-" * 200)


        st.markdown("### Data:")
        st.dataframe(pd.DataFrame(chat_response["data"]))

        metadata = chat_response["metadata"]
        data = chat_response["data"]

        if data and isinstance(data, list) and isinstance(data[0], dict):
            if len(data[0]) == 1:
                categories = [item[list(item.keys())[0]] for item in data]
                values = categories
            else:
                categories = list(data[0].keys())
                if len(categories) == 1:
                    values = [item[categories[0]] for item in data]
                    categories = values
                else:
                    prioritized_columns = ["DENUMIRE_SUCURSALA", "NUMAR_CERERE", "size", "HIST_DATE", "COD_SUCURSALA", "COD_AGENTIE",
                                          "DENUMIRE_AGENTIE", "PRODUS", "DATA_SEM_OFERTA", "DATA_STARE_CERERE", "STATUS_CERERE",
                                          "DESCRIERE_STARE_CERERE", "DATA_IN_OFERTA", "PBA_BAZA", "PBA_ASIG_SUM",
                                          "PBA_TOTAL_SEMNARE_CERERE", "PBA_CTR_ASOC", "PBA_TOTAL_EMITERE_CERERE", "FRECVENTA_PLATA"]

                    for col in prioritized_columns:
                        if all(col in item for item in data):
                            categories = [str(item[col]) for item in data]
                            if col != "NUMAR_CERERE" and col != "size":
                                if all("NUMAR_CERERE" in item for item in data):
                                     values = [item.get("NUMAR_CERERE", 0) for item in data]
                                elif all("size" in item for item in data):
                                     values = [item.get("size", 0) for item in data]

                                else:
                                    numeric_col = next((c for c in data[0] if isinstance(data[0][c], (int, float))), None)
                                    if numeric_col:
                                        values = [item.get(numeric_col, 0) for item in data]
                                    else:
                                         values = [str(list(item.values())[1]) for item in data]
                            break
                    else:
                        values = [str(list(item.values())[1]) for item in data]

        elif isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
            categories = list(range(len(data)))
            values = data
        elif isinstance(data, (int, float, str)):
            categories = ["Result"]
            values = [data]
        else:
            categories = []
            values = []
            st.warning("Unexpected data format. Check the query and data.")

        plots = generate_plots(metadata, categories, values)

        st.session_state["query"] = user_query
        st.session_state["response_text"] = result
        st.session_state["chat_response"] = chat_response
        st.session_state["plots"] = plots
        st.session_state["generated_code"] = generated_code  # Store the generated code

    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Submit"):
    with st.spinner("Processing query..."):
        try:
            process_query()
        except Exception as e:
            st.error(f"An error occurred: {e}")

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
