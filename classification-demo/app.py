import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import utils.dash_reusable_components as drc
import utils.figures as figs

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Breast Cancer Classifier"
server = app.server

# Load data
df = pd.read_csv('breast-cancer.csv')
feature_names = ['radius_mean', 'texture_mean', 'compactness_mean', 'concavity_mean']

# Default parameters
DEFAULT_SAMPLE_SIZE = len(df)
DEFAULT_TEST_SIZE = 0.4
DEFAULT_MAX_ITER = 1000

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.Div(
                    className="container scalable",
                    children=[
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Breast Cancer Diagnosis Classifier",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.Div(
                            id="banner-subtitle",
                            children="CS 150, Community Action Computing | Allie Peterson"
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    children=[
                        html.Div(
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-sample-size",
                                            min=100,
                                            max=len(df),
                                            step=50,
                                            marks={
                                                str(i): str(i)
                                                for i in range(100, len(df) + 1, 100)
                                            },
                                            value=DEFAULT_SAMPLE_SIZE,
                                        ),
                                        drc.NamedSlider(
                                            name="Test Size (%)",
                                            id="slider-test-size",
                                            min=0.1,  # 10%
                                            max=0.9,  # 90%
                                            step=0.05,
                                            marks={
                                                i/10: f"{int(i*10)}"
                                                for i in range(1, 10, 1)
                                            },
                                            value=DEFAULT_TEST_SIZE,
                                        ),
                                        drc.NamedSlider(
                                            name="Max Iterations",
                                            id="slider-max-iter",
                                            min=100,
                                            max=2000,
                                            step=100,
                                            marks={
                                                str(i): str(i)
                                                for i in range(100, 2100, 500)
                                            },
                                            value=DEFAULT_MAX_ITER,
                                        ),
                                        drc.NamedDropdown(
                                            name="Feature 1 for Visualization",
                                            id="dropdown-feature-1",
                                            options=[{'label': f, 'value': f} for f in feature_names],
                                            value=feature_names[0],
                                            clearable=False
                                        ),
                                        drc.NamedDropdown(
                                            name="Feature 2 for Visualization",
                                            id="dropdown-feature-2",
                                            options=[{'label': f, 'value': f} for f in feature_names],
                                            value=feature_names[1],
                                            clearable=False
                                        ),
                                        html.Button(
                                            "Reset to Defaults",
                                            id="button-reset",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="right-column",
                            children=[
                                html.Div(
                                    id="graphs-container",
                                    children=[
                                        dcc.Graph(id="graph-roc-curve"),
                                        dcc.Graph(id="graph-confusion-matrix"),
                                        dcc.Graph(id="graph-feature-importance"),
                                        dcc.Graph(id="graph-decision-boundary"),
                                    ],
                                    style={
                                        'display': 'grid',
                                        'grid-template-columns': '1fr 1fr',
                                        'gap': '20px'
                                    }
                                ),
                                html.Div(
                                    id="metrics-display",
                                    style={
                                        'margin-top': '20px',
                                        'padding': '20px',
                                        'background': '#282b38',
                                        'border-radius': '5px'
                                    }
                                )
                            ],
                            style={'flex': 2}
                        )
                    ],
                    style={
                        'display': 'flex',
                        'flex-direction': 'row',
                        'gap': '20px'
                    }
                )
            ],
        ),
    ]
)

@app.callback(
    Output("slider-sample-size", "value"),
    Output("slider-test-size", "value"),
    Output("slider-max-iter", "value"),
    [Input("button-reset", "n_clicks")],
)
def reset_parameters(n_clicks):
    if n_clicks:
        return DEFAULT_SAMPLE_SIZE, DEFAULT_TEST_SIZE, DEFAULT_MAX_ITER
    return dash.no_update


@app.callback(
    Output("graph-roc-curve", "figure"),
    Output("graph-confusion-matrix", "figure"),
    Output("graph-feature-importance", "figure"),
    Output("graph-decision-boundary", "figure"),
    Output("metrics-display", "children"),
    [
        Input("slider-sample-size", "value"),
        Input("slider-test-size", "value"),
        Input("slider-max-iter", "value"),
        Input("dropdown-feature-1", "value"),
        Input("dropdown-feature-2", "value"),
    ],
)
def update_output(sample_size, test_size, max_iter, feature1, feature2):
    # Handle edge cases
    if test_size == 0:
        test_size = 0.01  # Minimum viable test size
    elif test_size == 1:
        test_size = 0.99  # Maximum viable test size

    # Rest of your callback remains the same...
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    X = df_sample[feature_names]
    y = df_sample['diagnosis'].map({'M': 1, 'B': 0})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=max_iter, class_weight='balanced')
    model.fit(X_train, y_train)

    # Get predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Generate figures using the reusable functions
    roc_fig = figs.serve_roc_curve(model, X_test, y_test)
    cm_fig = figs.serve_confusion_matrix(y_test, y_pred)
    feature_fig = figs.serve_feature_importance(model, feature_names)
    boundary_fig = figs.serve_prediction_plot(
        model,
        pd.DataFrame(X_train, columns=feature_names),
        pd.DataFrame(X_test, columns=feature_names),
        y_train,
        y_test,
        feature1,
        feature2
    )

    # Get metrics
    metrics = figs.serve_metrics(y_test, y_pred, y_prob)

    # Create metrics display
    metrics_display = html.Div([
        html.H4("Model Performance Metrics"),
        html.P(f"Accuracy: {metrics['accuracy']:.4f}"),
        html.P(f"Precision: {metrics['precision']:.4f}"),
        html.P(f"Recall: {metrics['recall']:.4f}"),
        html.P(f"F1 Score: {metrics['f1']:.4f}"),
        html.P(f"ROC AUC: {metrics['roc_auc']:.4f}"),
        html.Br(),
        html.P(f"Sample Size: {len(df_sample)}"),
        html.P(f"Test Size: {test_size:.2f}"),
        html.P(f"Max Iterations: {max_iter}"),
    ], style={'padding': '20px'})

    return roc_fig, cm_fig, feature_fig, boundary_fig, metrics_display

if __name__ == "__main__":
    app.run_server(debug=True)