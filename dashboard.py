import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("data/processed/churn_cleaned.csv")

required_columns = [
    "State", "Churn", "Account length", "Customer service calls",
    "Total day charge", "Total eve charge", "Total night charge", "Total intl charge",
    "International plan", "Voice mail plan", "Total day minutes", "Total eve minutes",
    "Total night minutes", "Total intl minutes"
]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    logger.error(f"Missing required columns: {missing_cols}")
    raise ValueError(f"Missing required columns: {missing_cols}")

# Ensure Churn column is numeric/0-1
df["Churn"] = df["Churn"].map({True: 1, False: 0, "Yes": 1, "No": 0, 1: 1, 0: 0})
if df["Churn"].isnull().any():
    logger.error("Churn column contains invalid or missing values")
    raise ValueError("Churn column contains invalid or missing values")

# Initialize Dash app
app = dash.Dash(__name__)

# Color palette
COLORS = {
    "bg": "#0a0e27",
    "card": "#1a1f3a",
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#3b82f6",
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("🚀 Customer Churn Analytics Dashboard", className="header-title"),
            html.P("Real-time Intelligence • Advanced Insights • Data-Driven Decisions", className="header-subtitle"),
        ], className="header-content"),
    ], className="header-section"),

    # Filters Section (Area code dropdown removed)
    html.Div([
        html.Div([
            html.Div("📍", className="filter-icon"),
            html.Div([
                html.Label("State", className="filter-label"),
                dcc.Dropdown(
                    id="state-dropdown",
                    options=[{"label": f"{state}", "value": state} for state in sorted(df["State"].unique())],
                    placeholder="All States",
                    className="custom-dropdown",
                    clearable=True
                ),
            ]),
        ], className="filter-box"),
    ], className="filters-section"),

    # KPIs Section
    html.Div([
        html.Div(id="kpi-total", className="kpi-box kpi-primary"),
        html.Div(id="kpi-churned", className="kpi-box kpi-danger"),
        html.Div(id="kpi-rate", className="kpi-box kpi-info"),
        html.Div(id="kpi-calls", className="kpi-box kpi-warning"),
        html.Div(id="kpi-revenue", className="kpi-box kpi-success"),
    ], className="kpi-section"),

    # Charts Grid
    html.Div([
        # Row 1 - Full Width
        html.Div([
            html.H3("📊 Account Length Distribution Analysis", className="chart-title"),
            dcc.Loading(dcc.Graph(id="account-length-histogram", className="chart"), type="circle"),
        ], className="chart-card chart-full"),

        # Row 2 - Two Columns
        html.Div([
            html.H3("☎️ Customer Service Calls Impact", className="chart-title"),
            dcc.Loading(dcc.Graph(id="service-calls-chart", className="chart"), type="circle"),
        ], className="chart-card chart-half"),

        html.Div([
            html.H3("🌍 International Plan Distribution", className="chart-title"),
            dcc.Loading(dcc.Graph(id="international-plan-pie-chart", className="chart"), type="circle"),
        ], className="chart-card chart-half"),

        # Row 3 - Two Columns
        html.Div([
            html.H3("📧 Voice Mail Plan Analysis", className="chart-title"),
            dcc.Loading(dcc.Graph(id="voice-mail-pie-chart", className="chart"), type="circle"),
        ], className="chart-card chart-half"),

        html.Div([
            html.H3("💰 Usage vs Charges Pattern", className="chart-title"),
            dcc.Loading(dcc.Graph(id="usage-scatter", className="chart"), type="circle"),
        ], className="chart-card chart-half"),

        # Row 4 - Full Width
        html.Div([
            html.H3("🌙 Time-Based Usage Analysis", className="chart-title"),
            dcc.Loading(dcc.Graph(id="time-usage-chart", className="chart"), type="circle"),
        ], className="chart-card chart-full"),

        # Row 5 - Two Columns
        html.Div([
            html.H3("📈 Charges Distribution", className="chart-title"),
            dcc.Loading(dcc.Graph(id="charges-box", className="chart"), type="circle"),
        ], className="chart-card chart-half"),

        html.Div([
            html.H3("🔥 Feature Correlation Heatmap", className="chart-title"),
            dcc.Loading(dcc.Graph(id="correlation-heatmap", className="chart"), type="circle"),
        ], className="chart-card chart-half"),
    ], className="charts-grid"),

    # Footer
    html.Div([
        html.P("© 2025 Churn Analytics Platform", className="footer-text")
    ], className="footer"),

], className="main-container")


# Callback
@app.callback(
    [
        Output("kpi-total", "children"),
        Output("kpi-churned", "children"),
        Output("kpi-rate", "children"),
        Output("kpi-calls", "children"),
        Output("kpi-revenue", "children"),
        Output("account-length-histogram", "figure"),
        Output("service-calls-chart", "figure"),
        Output("international-plan-pie-chart", "figure"),
        Output("voice-mail-pie-chart", "figure"),
        Output("usage-scatter", "figure"),
        Output("time-usage-chart", "figure"),
        Output("charges-box", "figure"),
        Output("correlation-heatmap", "figure"),
    ],
    Input("state-dropdown", "value"),
)
def update_dashboard(state):
    # Filter data
    filtered_df = df.copy()
    if state:
        filtered_df = filtered_df[filtered_df["State"] == state]

    # Handle empty filtered data
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(
            annotations=[dict(text="No data available for selected filters", showarrow=False)],
            plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["card"], font=dict(color="#ffffff")
        )
        empty_kpi = html.Div("No data", className="kpi-value")
        return (empty_kpi,) * 5 + (empty_fig,) * 8

    # Calculate KPIs
    total = len(filtered_df)
    churned = filtered_df["Churn"].sum()
    rate = (churned / total * 100) if total > 0 else 0
    avg_calls = filtered_df["Customer service calls"].mean()
    revenue = (filtered_df["Total day charge"].sum() +
               filtered_df["Total eve charge"].sum() +
               filtered_df["Total night charge"].sum() +
               filtered_df["Total intl charge"].sum())

    # KPI Cards
    kpi1 = html.Div([
        html.Div("👥", className="kpi-icon"),
        html.Div([
            html.H2(f"{total:,}", className="kpi-value"),
            html.P("Total Customers", className="kpi-label"),
        ])
    ])

    kpi2 = html.Div([
        html.Div("📉", className="kpi-icon"),
        html.Div([
            html.H2(f"{churned:,}", className="kpi-value"),
            html.P("Churned Customers", className="kpi-label"),
        ])
    ])

    kpi3 = html.Div([
        html.Div("📊", className="kpi-icon"),
        html.Div([
            html.H2(f"{rate:.1f}%", className="kpi-value"),
            html.P("Churn Rate", className="kpi-label"),
        ])
    ])

    kpi4 = html.Div([
        html.Div("📞", className="kpi-icon"),
        html.Div([
            html.H2(f"{avg_calls:.1f}", className="kpi-value"),
            html.P("Avg Service Calls", className="kpi-label"),
        ])
    ])

    kpi5 = html.Div([
        html.Div("💰", className="kpi-icon"),
        html.Div([
            html.H2(f"${revenue:,.0f}", className="kpi-value"),
            html.P("Total Revenue", className="kpi-label"),
        ])
    ])

    # Chart 1: Account Length - Histogram
    fig1 = go.Figure()
    for churn_val in [0, 1]:
        data = filtered_df[filtered_df["Churn"] == churn_val]["Account length"]
        fig1.add_trace(go.Histogram(
            x=data,
            name="Retained" if churn_val == 0 else "Churned",
            opacity=0.75,
            marker_color=COLORS["success"] if churn_val == 0 else COLORS["danger"],
            nbinsx=30
        ))
    fig1.update_layout(
        barmode="overlay",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        xaxis_title="Account Length (days)",
        yaxis_title="Number of Customers",
        hovermode="x unified",
        legend=dict(x=0.7, y=0.95, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 2: Service Calls - Line + Bar Combo
    service_data = filtered_df.groupby(["Customer service calls", "Churn"]).size().unstack(fill_value=0)
    fig2 = go.Figure()

    if 0 in service_data.columns:
        fig2.add_trace(go.Bar(
            x=service_data.index,
            y=service_data[0],
            name="Retained",
            marker_color=COLORS["success"]
        ))
    if 1 in service_data.columns:
        fig2.add_trace(go.Bar(
            x=service_data.index,
            y=service_data[1],
            name="Churned",
            marker_color=COLORS["danger"]
        ))

    # Add churn rate line
    churn_rates = []
    for calls in service_data.index:
        total_calls = filtered_df[filtered_df["Customer service calls"] == calls]
        if len(total_calls) > 0:
            churn_rates.append((total_calls["Churn"].sum() / len(total_calls)) * 100)
        else:
            churn_rates.append(0)

    fig2.add_trace(go.Scatter(
        x=service_data.index,
        y=churn_rates,
        name="Churn Rate %",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=COLORS["warning"], width=3),
        marker=dict(size=8)
    ))

    fig2.update_layout(
        barmode="group",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        xaxis_title="Number of Service Calls",
        yaxis_title="Customer Count",
        yaxis2=dict(title="Churn Rate %", overlaying="y", side="right", showgrid=False),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=60, t=40, b=40)
    )

    # Chart 3: International Plan - Sunburst
    intl_data = filtered_df.groupby(["International plan", "Churn"]).size().reset_index(name="count")
    intl_data["Churn_label"] = intl_data["Churn"].map({0: "Retained", 1: "Churned"})
    intl_data = intl_data.dropna(subset=["International plan", "Churn_label"])
    intl_data = intl_data[(intl_data["International plan"] != "") & (intl_data["Churn_label"] != "")]
    intl_data = intl_data.drop_duplicates(subset=["International plan", "Churn_label"])

    if intl_data.empty:
        fig3 = go.Figure().update_layout(
            annotations=[dict(text="No data for International Plan", showarrow=False)],
            plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["card"], font=dict(color="#ffffff")
        )
    else:
        fig3 = px.sunburst(
            intl_data,
            path=["International plan", "Churn_label"],
            values="count",
            color="Churn_label",
            color_discrete_map={"Retained": COLORS["success"], "Churned": COLORS["danger"]}
        )
        fig3.update_layout(
            plot_bgcolor=COLORS["bg"],
            paper_bgcolor=COLORS["card"],
            font=dict(color="#ffffff"),
            margin=dict(l=20, r=20, t=20, b=20)
        )

    # Chart 4: Voice Mail - Donut Chart (robust)
    vm_data = filtered_df.groupby(["Voice mail plan", "Churn"]).size().reset_index(name="count")

    def normalize_vm(x):
        if pd.isna(x):
            return "No"
        if isinstance(x, bool):
            return "Yes" if x else "No"
        s = str(x).strip().lower()
        if s in ("yes", "y", "true", "1", "t"):
            return "Yes"
        if s in ("no", "n", "false", "0", "f", "nan", "none", ""):
            return "No"
        return str(x)

    vm_data["Voice mail plan"] = vm_data["Voice mail plan"].apply(normalize_vm)
    vm_data["Churn_label"] = vm_data["Churn"].map({0: "Retained", 1: "Churned"}).fillna("Unknown")
    vm_data = vm_data.dropna(subset=["Voice mail plan", "Churn_label"])
    vm_data["label"] = vm_data["Voice mail plan"].astype(str) + " - " + vm_data["Churn_label"].astype(str)

    if vm_data.empty:
        fig4 = go.Figure().update_layout(
            annotations=[dict(text="No data for Voice Mail Plan", showarrow=False)],
            plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["card"], font=dict(color="#ffffff")
        )
    else:
        colors_vm = [COLORS["success"], COLORS["danger"], COLORS["info"], COLORS["warning"]]
        fig4 = go.Figure(data=[go.Pie(
            labels=vm_data["label"],
            values=vm_data["count"],
            hole=0.4,
            marker_colors=colors_vm[:len(vm_data)],
            textposition="inside",
            textinfo="percent+label"
        )])
        fig4.update_layout(
            plot_bgcolor=COLORS["bg"],
            paper_bgcolor=COLORS["card"],
            font=dict(color="#ffffff"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
            margin=dict(l=20, r=20, t=20, b=20)
        )

    # Chart 5: Usage vs Charges - Scatter
    fig5 = go.Figure()
    for churn_val in [0, 1]:
        data = filtered_df[filtered_df["Churn"] == churn_val]
        fig5.add_trace(go.Scatter(
            x=data["Total day minutes"],
            y=data["Total day charge"],
            mode="markers",
            name="Retained" if churn_val == 0 else "Churned",
            marker=dict(
                size=8,
                color=COLORS["success"] if churn_val == 0 else COLORS["danger"],
                opacity=0.6,
                line=dict(width=1, color="white")
            )
        ))
    fig5.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        xaxis_title="Total Day Minutes",
        yaxis_title="Total Day Charge ($)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 6: Time-based Usage - Grouped Bar
    time_data = pd.DataFrame({
        "Period": ["Day", "Evening", "Night", "International"],
        "Retained": [
            filtered_df[filtered_df["Churn"] == 0]["Total day minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 0]["Total eve minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 0]["Total night minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 0]["Total intl minutes"].mean()
        ],
        "Churned": [
            filtered_df[filtered_df["Churn"] == 1]["Total day minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 1]["Total eve minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 1]["Total night minutes"].mean(),
            filtered_df[filtered_df["Churn"] == 1]["Total intl minutes"].mean()
        ]
    })

    fig6 = go.Figure(data=[
        go.Bar(name="Retained", x=time_data["Period"], y=time_data["Retained"],
               marker_color=COLORS["success"]),
        go.Bar(name="Churned", x=time_data["Period"], y=time_data["Churned"],
               marker_color=COLORS["danger"])
    ])
    fig6.update_layout(
        barmode="group",
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        xaxis_title="Time Period",
        yaxis_title="Average Minutes",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 7: Charges - Box Plot
    fig7 = go.Figure()
    charge_cols = [
        ("Total day charge", "Day"),
        ("Total eve charge", "Evening"),
        ("Total night charge", "Night"),
        ("Total intl charge", "International")
    ]

    for col, name in charge_cols:
        for churn_val in [0, 1]:
            data = filtered_df[filtered_df["Churn"] == churn_val][col]
            fig7.add_trace(go.Box(
                y=data,
                name=f"{name} - {"Retained" if churn_val == 0 else "Churned"}",
                marker_color=COLORS["success"] if churn_val == 0 else COLORS["danger"]
            ))

    fig7.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        yaxis_title="Charge Amount ($)",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 8: Correlation Heatmap
    numeric_cols = ["Account length", "Total day minutes", "Total eve minutes",
                    "Total night minutes", "Total intl minutes",
                    "Customer service calls", "Churn"]
    corr = filtered_df[numeric_cols].corr()

    fig8 = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    fig8.update_layout(
        plot_bgcolor=COLORS["bg"],
        paper_bgcolor=COLORS["card"],
        font=dict(color="#ffffff"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return (kpi1, kpi2, kpi3, kpi4, kpi5,
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8)


if __name__ == "__main__":
    app.run(debug=True)


