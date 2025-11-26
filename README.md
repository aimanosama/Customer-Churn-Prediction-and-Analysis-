# Customer Churn Prediction & Analysis

A complete project that processes customer data, runs analysis in
notebooks, trains models, and provides an interactive UI using
Streamlit.

------------------------------------------------------------------------

## 1. Setup

1.  **Clone the repository**

``` bash
git clone Customer-Churn-Prediction-and-Analysis-
cd Customer-Churn-Prediction-and-Analysis-
```

2.  **Create and activate a virtual environment**

``` bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate      # Windows
```

3.  **Install dependencies**

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 2. Run Notebooks / Scripts

-   Open and run Jupyter notebooks inside the `notebooks/` folder:

``` bash
jupyter notebook
```

-   Run root-level scripts:

``` bash
python script_name.py
```

-   Outputs appear in:

```{=html}
<!-- -->
```
    results/ | reports/ | models/

------------------------------------------------------------------------

## 3. Serve Streamlit App

To launch the dashboard and prediction UI locally:

``` bash
streamlit run ui/app.py
```

-   Opens at: **http://localhost:8501**

------------------------------------------------------------------------

## Project Structure

    Customer-Churn-Prediction-and-Analysis-/
    ├── data/         # datasets
    ├── notebooks/    # analysis and experiments
    ├── models/       # saved model files
    ├── ui/           # Streamlit app scripts
    ├── results/      # outputs and charts
    ├── reports/      # summary files
    └── requirements.txt

------------------------------------------------------------------------

## Notes

-   Keep data files in `data/`
-   Train models once before using Streamlit UI
