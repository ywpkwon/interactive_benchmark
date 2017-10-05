import os
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from plotly import tools
import glob
import plotly.graph_objs as go
import pickle
import numpy as np
import collections
from flask import Flask
from util import AP
from scipy.interpolate import interp1d

gt_path = 'gt.txt'

colors = {
    'font': dict(color='black'),
    'titlefont': dict(color='black', size='14'),
    'plot_bgcolor': "#e6e6e6",
    'paper_bgcolor': "white",
}

target_files = glob.glob('*.pickle')

with open(gt_path, 'r') as f:
    lines = f.readlines()
    gt = []
    for line in lines:
        name, x1, y1, x2, y2, cl = line.split()
        x1 = max(0, float(x1)); y1 = max(0, float(y1))
        x2 = min(1, float(x2)); y2 = min(1, float(y2))
        gt.append({'name': name,
                   'bbox': [x1, y1, x2, y2],
                   'class': cl,
                   'detected': False})

class_instances = [g['class'] for g in gt]
class_counter = collections.Counter(class_instances)
class_keys = list(class_counter)

pie_data = [{
        'type': 'pie',
        'labels': class_keys,
        'values': [class_counter[k] for k in class_keys],
        'name': 'Class Distribution',
        'hole': 0.2,
        'opacity': 0.7,
        # 'marker': dict(colors=['#fac1b7', '#a9bb95', '#92d8d8']),
        'hoverinfo': "text+value+percent",
        'textinfo': "label+value+name",
        }
]
pie_layout = go.Layout(
        title="Class Distribution",
        # width=500,
        height=600,
        margin={'l': 60, 'b': 40, 't': 80, 'r': 60},
        plot_bgcolor=colors["plot_bgcolor"],
        # paper_bgcolor=colors["paper_bgcolor"],
        font=colors["font"],
        titlefont=colors["titlefont"],
        hovermode='closest'
)
main_layout = go.Layout(
        title="Precision - Recall",
        # width=700,
        height=600,
        xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
        yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
        margin={'l': 40, 'b': 40, 't': 50, 'r': 40},
        plot_bgcolor=colors["plot_bgcolor"],
        # paper_bgcolor=colors["paper_bgcolor"],
        font=colors["font"],
        titlefont=colors["titlefont"],
        hovermode='closest')

individual_layouts = [go.Layout(
        title=class_name,
        height=600,
        xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
        yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
        margin={'l': 40, 'b': 40, 't': 80, 'r': 40},
        showlegend=False,
        plot_bgcolor=colors["plot_bgcolor"],
        # paper_bgcolor=colors["paper_bgcolor"],
        font=colors["font"],
        titlefont=colors["titlefont"],
        hovermode='closest') for class_name in class_keys]


app = dash.Dash('Make Vehicles Great Again!!')

STATIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# @app.server.route('/static/<resource>')
# def serve_static(resource):
    # return flask.send_from_directory(STATIC_PATH, resource)


# dcc._css_dist[0]['relative_package_path'].append('mycss.css')
# app.css.append_css({'external_url': '/static/mycss.css'})  # noqa: E501#
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501


# encoded_image = base64.b64encode(open("home/paul/projects/interactive_benchmark/phantom.png", 'rb').read())


app.layout = html.Div([

    html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src="/static/phantom.png",
                        className='two columns',
                        style={
                            'height': '60',
                            'width': '160',
                            'float': 'left',
                            'position': 'relative',
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    html.H1(
                        'PHANTOM AI Performance Lab',
                        className='eleven columns',
                        style={'text-align': 'center'}
                    ),
                    # html.P(''),
                    html.H6(
                        '"If you can\'t measure it, you can\'t manage it."',
                        className='eleven columns',
                        style={'text-align': 'center', 'color': 'gray'}
                    ),
                ], className='eleven columns')
        ], style={'text-align': 'center', 'margin-top': '50', 'margin-bottom': '50'}, className='row'
    ),
    html.Div(
        [
            html.Div(
                [
                    html.P('Evaluation set:'),
                    dcc.Dropdown(
                        id='evaluation-set-dropdown',
                        options=[
                            {'label': 'random', 'value': 'random'},
                            {'label': 'balanced', 'value': 'balanced'},
                            {'label': 'human', 'value': 'human'},
                            {'label': 'night', 'value': 'night'}
                        ],
                        value='random'
                    ),
                ], className='four columns'),
            html.Div(
                [
                    html.P('IoU Thresholds:'),  # noqa: E501
                    dcc.Dropdown(
                        id='my-dropdown',
                        options=[
                            {'label': '0.5', 'value': 0.5},
                            {'label': '0.7', 'value': 0.7},
                            {'label': '0.9', 'value': 0.9}
                        ],
                        value=0.5
                    ),
                ], className='four columns'),
        ], style={'text-align': 'center', 'margin-top': '10', 'margin-bottom': '30'}, className='row'),
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id='precision-recall'),
                ],
                className='eight columns',
                style={'margin-top': '10'}
            ),
            html.Div(
                [
                    dcc.Graph(id='distribution'),
                ],
                className='four columns',
                style={'margin-top': '10'}
            ),
        ], className='row'),
    html.Div(
        [
            html.Div(
                [
                    dcc.Graph(id=class_name),
                ],
                className='four columns',
                style={'margin-top': '10'}
            ) for class_name in class_keys
        ], className='row')
    ])


def prcurve(cache, category="all", threshold=0.5):

    category = category.lower()
    name = os.path.splitext(os.path.basename(cache))[0]
    with open(cache, 'rb') as pf:
        data = pickle.load(pf)

    # if threshold not in data: return name, [], []
    # data = data[threshold]
    gt = data['gt']
    pr = data['prediction']

    n_total = 0
    for fname in gt:
        for g in gt[fname]:
            if category != "all" and g['class'].lower() != category: continue
            if not g['valid']: continue
            n_total += 1

    assert(n_total > 0)

    true_positives = []
    for p in pr:
        if category != "all" and p['class'].lower() != category: continue
        if p['correct']:    true_positives.append(1)
        else:               true_positives.append(0)

    true_positives = np.array(true_positives)
    true_positives = np.cumsum(true_positives)
    recall = true_positives / n_total
    precision = true_positives / (np.arange(len(true_positives))+1)
    ap = AP(recall, precision)

    resolution = 1000
    if len(recall) > resolution:
        # reduce the amount of points to plot
        f = interp1d(recall, precision)
        x = np.linspace(np.min(recall), np.max(recall), num=1000, endpoint=True)
        y = f(x)
        recall = x
        precision = y
    name = '%0.03f-' % ap + name
    # return name, recall, precision
    return name, recall, precision


@app.callback(Output('precision-recall', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    print (selected_dropdown_value)
    data = []
    for target_file in target_files:
        name, recall, precision = prcurve(target_file, threshold=selected_dropdown_value)
        data.append(go.Scatter(x=recall, y=precision, opacity=0.7, mode='lines', name=name))
    figure = {'data': data, 'layout': main_layout}
    return figure


@app.callback(Output('distribution', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    figure = {'data': pie_data, 'layout': pie_layout}
    return figure


@app.callback(Output(class_keys[0], 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    data = []
    for target_file in target_files:
        name, recall, precision = prcurve(target_file, class_keys[0])
        data.append(go.Scatter(x=recall, y=precision, opacity=0.7, mode='lines', name=name))
    figure = {'data': data, 'layout': individual_layouts[0]}
    return figure


@app.callback(Output(class_keys[1], 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    data = []
    for target_file in target_files:
        name, recall, precision = prcurve(target_file, class_keys[1])
        data.append(go.Scatter(x=recall, y=precision, opacity=0.7, mode='lines', name=name))
    figure = {'data': data, 'layout': individual_layouts[1]}
    return figure


@app.callback(Output(class_keys[2], 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):

    data = []
    for target_file in target_files:
        name, recall, precision = prcurve(target_file,  class_keys[2])
        data.append(go.Scatter(x=recall, y=precision, opacity=0.7, mode='lines', name=name))
    figure = {'data': data, 'layout': individual_layouts[2]}
    return figure




# @app.callback(Output('individual_graphs', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown_value):


#     import matplotlib.pyplot as plt
#     import plotly.tools as tls

#     target_files = ['ssd_incep1_wide480.pickle', 'ssd_wide480a.pickle']
#     n = len(target_files)
#     fig = plt.figure()

#     for i, cls in enumerate(class_keys):
#         ax = fig.add_subplot(1, n+1, i+1)
#         t = np.arange(0.01, 5.0, 0.01)
#         s1 = np.sin(2*np.pi*t)
#         ax.plot(t, s1)
#         ax.set_title(cls)
#     fig = tls.mpl_to_plotly(fig)

#     # target_files = ['ssd_incep1_wide480.pickle', 'ssd_wide480a.pickle']
#     # fig = tools.make_subplots(rows=1, cols=len(target_files))
#     # for i, target in enumerate(target_files):
#     #     trace = go.Scatter(
#     #         x=[1, 2, 3],
#     #         y=[4, 5, 6],
#     #         name=target,
#     #         xaxis='x%d' % (i+1),
#     #         yaxis='y%d' % (i+1)
#     #     )
#     #     fig.append_trace(trace, 1, i+1)

#     # n = len(target_files)
#     # fig['layout'].update(height=400, width=1200, title='individuals')
#     # for i, target in enumerate(target_files):
#     #     import pdb; pdb.set_trace()
#     #     fig['layout'].update(xaxis=dict(domain=[i/n, (i+1)/n], anchor='x%d' % (i+1)),
#     #                          yaxis=dict(domain=[0, 1], anchor='y%d' % (i+1)))

#     #                      # xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
#     #                      # yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
#     #                      # margin={'l': 40, 'b': 40, 't': 50, 'r': 40},
#     #                      # # legend={'x': 0, 'y': 1},
#     #                      # plot_bgcolor=colors["plot_bgcolor"],
#     #                      # paper_bgcolor=colors["paper_bgcolor"],

#     return fig


# @app.callback(Output('vehicle', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown_value):

#     print (selected_dropdown_value)

#     cache = 'ssd_incep1_wide480.pickle'
#     prcurve1 = prcurve(cache)
#     cache = 'ssd_wide480a.pickle'
#     prcurve2 = prcurve(cache)
#     data = [prcurve1, prcurve2]

#     layout = go.Layout(
#                 title="Precision - Recall",
#                 width=450,
#                 height=350,
#                 xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
#                 yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
#                 margin={'l': 40, 'b': 40, 't': 50, 'r': 40},
#                 # legend={'x': 0, 'y': 1},
#                 plot_bgcolor=colors["plot_bgcolor"],
#                 paper_bgcolor=colors["paper_bgcolor"],
#                 hovermode='closest'
#             )
#     figure = {'data': data, 'layout': layout}
#     return figure


# @app.callback(Output('motocycle', 'figure'), [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown_value):

#     print (selected_dropdown_value)

#     cache = 'ssd_incep1_wide480.pickle'
#     prcurve1 = prcurve(cache)
#     cache = 'ssd_wide480a.pickle'
#     prcurve2 = prcurve(cache)
#     data = [prcurve1, prcurve2]

#     layout = go.Layout(
#                 title="Precision - Recall",
#                 width=450,
#                 height=350,
#                 xaxis={"autorange": False, "range": [0, 1], 'title': 'recall'},
#                 yaxis={"autorange": False, "range": [0, 1], 'title': 'precision'},
#                 margin={'l': 40, 'b': 40, 't': 50, 'r': 40},
#                 # legend={'x': 0, 'y': 1},
#                 plot_bgcolor=colors["plot_bgcolor"],
#                 paper_bgcolor=colors["paper_bgcolor"],
#                 hovermode='closest'
#             )
#     figure = {'data': data, 'layout': layout}
#     return figure




if __name__ == '__main__':

    app.run_server()
    # app.run_server(host='0.0.0.0')
