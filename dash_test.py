import os
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import pickle
import numpy as np


app = dash.Dash('Make Vehicles Great Again!!')

app.layout = html.Div([
        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': '0.5', 'value': 0.5},
                {'label': '0.7', 'value': 0.7},
                {'label': '0.9', 'value': 0.9}
            ],
            value=0.5
        ),
        dcc.Graph(id='precision-recall')], style={'width': '500'})


def prcurve(cache):

    name = os.path.splitext(os.path.basename(cache))[0]
    with open(cache, 'rb') as pf:
        data = pickle.load(pf)

    pr = data['prediction']
    n_gt_bboxes = data["n_gt_bboxes"]

    true_positives = np.array([p['correct'] for p in pr], dtype=np.int32)
    true_positives = np.cumsum(true_positives)

    recall = true_positives / n_gt_bboxes
    precision = true_positives / (np.arange(len(true_positives))+1)

    return go.Scatter(x=recall,
                      y=precision,
                      opacity=0.7,
                      mode='lines',
                      # marker={'size': 2, 'line': {'width': 1}},
                      name=name)


@app.callback(Output('precision-recall', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    print (selected_dropdown_value)

    cache = '/home/phantom/projects/benchmark/ssd_incep1_wide480.pickle'
    prcurve1 = prcurve(cache)
    cache = '/home/phantom/projects/benchmark/ssd_wide480a.pickle'
    prcurve2 = prcurve(cache)
    data = [prcurve1, prcurve2]

    layout = go.Layout(
                width=1200,
                height=1100,
                xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
                yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
                margin={'l': 40, 'b': 40, 't': 150, 'r': 40},
                # legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
    figure = {'data': data, 'layout': layout}
    return figure


if __name__ == '__main__':


    app.run_server()
