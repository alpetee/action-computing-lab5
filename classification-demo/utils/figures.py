import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn import metrics

# Define color scales
BRIGHT_CSCALE = [[0, "#ff3700"], [1, "#0b8bff"]]
CSCALE = [
    [0.0000000, "#ff744c"],
    [0.1428571, "#ff916d"],
    [0.2857143, "#ffc0a8"],
    [0.4285714, "#ffe7dc"],
    [0.5714286, "#e5fcff"],
    [0.7142857, "#c8feff"],
    [0.8571429, "#9af8ff"],
    [1.0000000, "#20e6ff"],
]

def serve_roc_curve(model, X_test, y_test):
    """
    Generate ROC curve figure for binary classifier with square aspect ratio
    Args:
        model: trained classifier
        X_test: test features
        y_test: test labels
    Returns:
        plotly figure object
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    auc_score = metrics.roc_auc_score(y_test, y_prob)

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='#13c6e9', width=2),
    )
    trace1 = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='#ff744c', dash='dash', width=1)
    )

    layout = go.Layout(
        title=f'ROC Curve (AUC = {auc_score:.3f})',
        xaxis=dict(
            title='False Positive Rate',
            gridcolor='#2f3445',
            range=[0, 1],
            constrain='domain',
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            title='True Positive Rate',
            gridcolor='#2f3445',
            range=[0, 1],
            constrain='domain'
        ),
        legend=dict(x=0.5, y=0.15, orientation='h'),
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=60, r=30, t=60, b=60),
        plot_bgcolor='#282b38',
        paper_bgcolor='#282b38',
        font={'color': '#a5b1cd'},
    )

    return go.Figure(data=[trace0, trace1], layout=layout)

def serve_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix heatmap with TP/TN/FP/FN labels
    Args:
        y_true: true labels
        y_pred: predicted labels
    Returns:
        plotly figure object
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Reorder the confusion matrix to show TP in top-left
    cm_reordered = np.array([[tp, fp],
                             [fn, tn]])

    # Create labels with counts and type
    labels = [
        [f"TP: {tp}", f"FP: {fp}"],
        [f"FN: {fn}", f"TN: {tn}"]
    ]

    heatmap = go.Heatmap(
        z=cm_reordered,
        x=['Predicted Positive', 'Predicted Negative'],
        y=['Actual Positive', 'Actual Negative'],
        text=labels,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    )

    layout = go.Layout(
        title='Confusion Matrix',
        xaxis=dict(
            constrain='domain',
            side='top'
        ),
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='#282b38',
        paper_bgcolor='#282b38',
        font={'color': '#a5b1cd'},
    )

    return go.Figure(data=[heatmap], layout=layout)

def serve_metrics(y_true, y_pred, y_prob=None):
    """
    Generate metrics display
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities (optional)
    Returns:
        Dictionary of metrics
    """
    metrics_dict = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred),
        'f1': metrics.f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true, y_prob)

    return metrics_dict

def serve_feature_importance(model, feature_names):
    """
    Generate feature importance plot
    Args:
        model: trained classifier with coef_ attribute
        feature_names: list of feature names
    Returns:
        plotly figure object
    """
    if hasattr(model, 'coef_'):
        importance = model.coef_[0]
    else:
        return None

    sorted_idx = np.argsort(np.abs(importance))
    importance = importance[sorted_idx]
    feature_names = np.array(feature_names)[sorted_idx]

    trace = go.Bar(
        x=importance,
        y=feature_names,
        orientation='h',
        marker=dict(color='#13c6e9')
    )

    layout = go.Layout(
        title='Feature Importance',
        xaxis=dict(title='Coefficient Value'),
        yaxis=dict(title='Feature'),
        margin=dict(l=150, r=30, t=40, b=30),
        plot_bgcolor='#282b38',
        paper_bgcolor='#282b38',
        font={'color': '#a5b1cd'}
    )

    return go.Figure(data=[trace], layout=layout)


def serve_prediction_plot(model, X_train, X_test, y_train, y_test, feature1, feature2, mesh_step=0.02):
    """
    Create a 2D decision boundary plot for two selected features
    """
    # Get the selected features
    x_min, x_max = X_train[feature1].min() - 0.5, X_train[feature1].max() + 0.5
    y_min, y_max = X_train[feature2].min() - 0.5, X_train[feature2].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    # Create a temporary dataframe with the mesh grid values
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Initialize DataFrame with all features in correct order
    temp_df = pd.DataFrame(columns=X_train.columns)

    # Set the selected features
    temp_df[feature1] = grid[:, 0]
    temp_df[feature2] = grid[:, 1]

    # Fill other features with their mean values
    for col in X_train.columns:
        if col not in [feature1, feature2]:
            temp_df[col] = X_train[col].mean()

    # Ensure columns are in same order as training data
    temp_df = temp_df[X_train.columns]

    # Get predictions
    Z = model.predict_proba(temp_df)[:, 1]
    Z = Z.reshape(xx.shape)

    # Get accuracy scores
    train_score = metrics.accuracy_score(y_train, model.predict(X_train))
    test_score = metrics.accuracy_score(y_test, model.predict(X_test))

    # Create contour plot
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=CSCALE,
        opacity=0.7,
    )

    # Plot training data
    trace1 = go.Scatter(
        x=X_train[feature1],
        y=X_train[feature2],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(
            size=8,
            color=y_train,
            colorscale=BRIGHT_CSCALE,
            line=dict(width=1, color='white')
        )
    )

    # Plot test data
    trace2 = go.Scatter(
        x=X_test[feature1],
        y=X_test[feature2],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(
            size=8,
            symbol="triangle-up",
            color=y_test,
            colorscale=BRIGHT_CSCALE,
            line=dict(width=1, color='white')
        )
    )

    layout = go.Layout(
        title=f"Decision Boundary ({feature1} vs {feature2})",
        xaxis=dict(title=feature1),
        yaxis=dict(title=feature2),
        hovermode="closest",
        legend=dict(x=0, y=-0.15, orientation="h"),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    return go.Figure(data=[trace0, trace1, trace2], layout=layout)