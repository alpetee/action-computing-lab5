from textwrap import dedent
import dash_core_components as dcc
import dash_html_components as html

# Display utility functions
def _merge(a, b):
    return dict(a, **b)

def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}

# Custom Display Components
def Card(children, **kwargs):
    """A styled container for grouping related content"""
    return html.Section(
        className="card",
        children=children,
        style={
            "margin": "10px",
            "padding": "15px",
            "borderRadius": "5px",
            "backgroundColor": "#282b38",
            "boxShadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2)"
        },
        **_omit(["style"], kwargs)
    )

def FormattedSlider(**kwargs):
    """Slider with consistent styling"""
    return html.Div(
        style=kwargs.get("style", {"padding": "0px 10px"}),
        children=dcc.Slider(**_omit(["style"], kwargs))
    )

def NamedSlider(name, **kwargs):
    """Slider with a title label"""
    return html.Div(
        style={"padding": "15px 10px 20px 4px"},
        children=[
            html.P(
                children=f"{name}:",
                style={
                    "marginBottom": "5px",
                    "fontWeight": "bold",
                    "color": "#a5b1cd"
                }
            ),
            html.Div(
                style={"marginLeft": "6px"},
                children=dcc.Slider(
                    **kwargs,
                    tooltip={"always_visible": False, "placement": "bottom"}
                )
            ),
        ],
    )

def NamedDropdown(name, **kwargs):
    """Dropdown with a title label"""
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            html.P(
                children=f"{name}:",
                style={
                    "marginBottom": "5px",
                    "fontWeight": "bold",
                    "color": "#a5b1cd"
                }
            ),
            dcc.Dropdown(
                **kwargs,
                style={
                    "backgroundColor": "#282b38",
                    "color": "#a5b1cd"
                }
            ),
        ],
    )

def NamedRadioItems(name, **kwargs):
    """Radio buttons with a title label"""
    return html.Div(
        style={"padding": "15px 10px 20px 4px"},
        children=[
            html.P(
                children=f"{name}:",
                style={
                    "marginBottom": "10px",
                    "fontWeight": "bold",
                    "color": "#a5b1cd"
                }
            ),
            dcc.RadioItems(
                **kwargs,
                labelStyle={
                    "display": "inline-block",
                    "marginRight": "10px",
                    "color": "#a5b1cd"
                }
            ),
        ],
    )

def MetricCard(title, value, color="#13c6e9"):
    """Styled card for displaying a single metric"""
    return html.Div(
        className="metric-card",
        style={
            "border": f"1px solid {color}",
            "borderRadius": "5px",
            "padding": "10px",
            "margin": "10px",
            "textAlign": "center",
            "backgroundColor": "#282b38"
        },
        children=[
            html.Div(
                children=title,
                style={
                    "fontSize": "0.9em",
                    "color": "#a5b1cd",
                    "marginBottom": "5px"
                }
            ),
            html.Div(
                children=value,
                style={
                    "fontSize": "1.2em",
                    "fontWeight": "bold",
                    "color": color
                }
            )
        ]
    )

def DemoDescription(filename, strip=False):
    """Component for displaying markdown descriptions from files"""
    with open(filename, "r") as file:
        text = file.read()

    if strip:
        text = text.split("<Start Description>")[-1]
        text = text.split("<End Description>")[0]

    return html.Div(
        className="row",
        style={
            "padding": "15px 30px 27px",
            "margin": "45px auto 45px",
            "width": "80%",
            "max-width": "1024px",
            "borderRadius": 5,
            "border": "thin lightgrey solid",
            "font-family": "Roboto, sans-serif",
        },
        children=dcc.Markdown(dedent(text)),
    )