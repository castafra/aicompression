import dash
import sys
import os
from pathlib import Path
home_path = Path.home()
sys.path.append(f"{home_path}\\Documents\\GitHub\\aicompression")
from src.pipeline import *
from subprocess import call
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import datetime
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go
from PIL import Image
import requests


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server  # Expose the server variable for deployments

#Useful functions

#for the first slide, display the input slide





# declare table contents
# ******************************************
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Upload(id='upload-image',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '0',
                            'color': 'white'
                        },
                        # Don't allow multiple files to be uploaded
                        multiple=False
                    ),
            dbc.Col(html.Div(id='image-uploaded-1'), width='6'),
            dbc.Col(html.Div(id='object-detection-image'), width='6')
        ]
    ),
    className="font-weight-light",
)

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row("Please add a slide on sheet 1"),
            dbc.Col(html.Div(id='image-uploaded-2'), width='12')
        ]
    ),
    className="mt-3",
)

tab3_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row("Please add a slide on sheet 1", className="btn-white font-weight-bold")
        ]
    ),
    className="mt-3",
)

tab4_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row("Please add a slide on sheet 1", className="btn-white font-weight-bold")
        ]
    ),
    className="mt-3",
)

#Final Dashboard
# ******************************************
app.config['suppress_callback_exceptions'] = True

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("AI Compression", className="text-center text-light font-weight-light, mb-4"),
                width=12),
        dbc.Col(html.Img(src=app.get_asset_url("logo_cs.png"),
                style={"float": "right", "height": 70}))
    ]),

    dbc.Tabs(
        [
            dbc.Tab(tab1_content, label="Object detection",
                    labelClassName="btn btn-dark btn-lg active font-weight-bold, w-20", tab_id="tab-1"),
            dbc.Tab(tab2_content, label="Background Retrieval",
                    labelClassName="font-weight-bold, w-20", tab_id="tab-2"),
            dbc.Tab(tab3_content, label="Text recognition",
                    labelClassName="btn-white font-weight-bold, w-20", disabled=True
            ),
            dbc.Tab(tab4_content, label="Performance",
                    labelClassName="btn-white font-weight-bold, w-20", disabled=True
            ),
            dbc.Tab(
                "Allez l'OM", label="About", labelClassName="btn-white font-weight-bold, w-20", disabled=True
            ),
        ],
        className= "font-weight-light, w-100, h-100",
        active_tab="tab-1"
    ),
])

# Callback
# ********************************************************************


# display input image in first sheet

@app.callback(Output('image-uploaded-1','children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
              State('upload-image', 'last_modified')])

def parse_contents_1(contents, filename, date):
    return dbc.Col([
        html.H5(filename),
        dbc.CardImg(src=contents, style={'height':'50%', 'width': '50%'}),
        dbc.Row(dbc.Button(id='run-object-detection', children='Run Object Detection', color='success'))
    ], width={'size':12})


def update_output_1(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents_1(list_of_contents, list_of_names, list_of_dates) ]
        return children

#run object detection after clicking on the button

@app.callback(Output('object-detection-image', 'children'),
             [Input('run-object-detection', 'n_clicks'),Input('upload-image','filename')],
             [State('upload-image','contents')])

def run_script_onClick(n_clicks,filename,contents):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate
    
    # Load your output file with "some code"
    compression = compressor(PATH_TO_OD_LABELS=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels", PATH_TO_OD_MODEL_DIR=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection")
    print(filename)
    output_content = compression.detect_objects(f"{home_path}\\Documents\\GitHub\\aicompression\\App\\input_images\\{filename}")
    viz_od = compression.visualize_detections()
    # Now return.
    return Image.fromarray(viz_od)


#display input image in the second sheet

@app.callback(Output('image-uploaded-2', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
              State('upload-image', 'last_modified')])

def parse_contents_2(contents, filename, date):
    return dbc.Col([
        html.H5(filename),
        dbc.CardImg(src=contents, style={'height':'50%', 'width': '50%'}),
        dbc.Row(dbc.Button(id='run-background-retrieval', children='Run background retrieval', color='success'))
    ], width={'size':12})

def update_output_2(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents_2(list_of_contents, list_of_names, list_of_dates) ]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)