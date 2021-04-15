import dash
import sys
import os
from pathlib import Path
from dash_bootstrap_components._components.Card import Card

from dash_bootstrap_components._components.CardBody import CardBody
from dash_bootstrap_components._components.CardImg import CardImg
from numpy.core.fromnumeric import compress
home_path = Path.home()
sys.path.append(f"{home_path}\\Documents\\GitHub\\aicompression")
from src.pipeline import *
from subprocess import call
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import datetime
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from plotly.subplots import make_subplots
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import requests


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR], suppress_callback_exceptions = True)
server = app.server  # Expose the server variable for deployments

#Useful functions

#for the first slide, display the input slide





# declare table contents
# ******************************************
tab1_content = dbc.Card([
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
        ],
        className="font-weight-light"),
    dbc.Row([
        dbc.Col(html.Div(id='image-uploaded-1'), width='5'),
        dbc.Col([
            dbc.CardImg(id='object-detection-image', src=[]),
            dbc.Row(html.H1(" ")
            )], 
        width='6')
        ],
    justify="center"
    ),
    dbc.Row(html.H1(" "))
])

tab2_content = dbc.Card([
    dbc.CardBody(
    [
        dbc.Row("Please add a slide on sheet 1", className="btn-white font-weight-bold"),
        dbc.Row([
            dbc.Col(html.Div(id='image-uploaded-2'), width=5),
            dbc.Col([
                dbc.CardImg(id='background-image', src=[]),
                dbc.Row(html.H1(" ")
                )],
            width=6),
        ],
        justify="center"
        ),
        dbc.Row(html.H1(" "))
    ])
])

tab3_content = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row("Please add a slide on sheet 1", className="btn-white font-weight-bold"),
            dbc.Row(html.Div(id='text-box-button')),
            dbc.Col(html.Div(children=[],id='text-boxes'), width=12),
        ])
    ),


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
                    labelClassName="font-weight-bold, w-20", tab_id="tab-1"),
            dbc.Tab(tab2_content, label="Background Retrieval",
                    labelClassName="font-weight-bold, w-20", tab_id="tab-2"),
            dbc.Tab(tab3_content, label="Text recognition",
                    labelClassName="btn-white font-weight-bold, w-20", tab_id="tab-3"
            ),
            dbc.Tab(tab4_content, label="Performance",
                    labelClassName="btn-white font-weight-bold, w-20", tab_id='tab-4'
            ),
            dbc.Tab(
                "This project has been developed by François Castagnos and Mehdi Arsaoui, under the supervision of Marine Picot", 
                label="About", labelClassName="btn-white font-weight-bold, w-20", tab_id="tab-5"
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
              [State('upload-image', 'filename')])

def parse_contents_1(contents, filename):
    if not contents:
        raise PreventUpdate
    return dbc.Col([
        dbc.Row(html.H5(filename), justify='center', align='start'),
        dbc.Row(dbc.CardImg(src=contents, style={'height': '50%', 'width': '50%'}), justify='center', align='center'),
        dbc.Row(dbc.Button(id='run-object-detection', children='Run Object Detection', color='success', size='sm'), justify='center', align='end')
    ], width=12)


def update_output_1(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents_1(list_of_contents, list_of_names) ]
        return children

#run object detection after clicking on the button

@app.callback(Output('object-detection-image', 'src'),
             [Input('run-object-detection', 'n_clicks'),
             Input('upload-image','filename')],
             [State('upload-image','contents')])

def run_script_onClick(n_clicks, filename, contents):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate
    
    # Load your output file with "some code"
    global compression 
    compression = compressor(PATH_TO_OD_LABELS=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels", PATH_TO_OD_MODEL_DIR=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection")
    print(filename)
    output_content = compression.detect_objects(f"{home_path}\\Documents\\GitHub\\aicompression\\App\\input_images\\{filename}")
    viz_od = compression.visualize_detections()
    
    img_out = Image.fromarray(viz_od)
    img_out.save(f"{home_path}\\Documents\\GitHub\\aicompression\\App\\output_images\\{filename}")
    return img_out


#display input image in the second sheet

@app.callback(Output('image-uploaded-2', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])

def parse_contents_2(contents, filename):
    if not contents:
        raise PreventUpdate
    return dbc.Col([
        dbc.Row(html.H5(filename), justify='center', align='start'),
        dbc.Row(dbc.CardImg(src=contents, style={'height': '50%', 'width': '50%'}), justify='center', align='center'),
        dbc.Row(dbc.Button(id='run-background-retrieval', children='Run background retrieval', color='success', size='sm'), justify='center', align='end')
    ], width=12)

def update_output_2(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents_2(list_of_contents, list_of_names)]
        return children

#run background retrieve after clicking on the button

@app.callback(Output('background-image', 'src'),
             [Input('run-background-retrieval', 'n_clicks'),
             Input('upload-image','filename')],
             [State('upload-image','contents')])

def run_script_onClick(n_clicks, filename, contents):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate
    
    # Load your output file with "some code"
    #global compression 
    #compression = compressor(PATH_TO_OD_LABELS=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels", PATH_TO_OD_MODEL_DIR=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection")
    
    #PIL image with retrieved background
    background_img = compression.background_retrieve()
    background_img.save(f"{home_path}\\Documents\\GitHub\\aicompression\\App\\output_images\\{filename}")
    return background_img


# display run text recognition button in first sheet

@app.callback(Output('text-box-button','children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])

def parse_contents_3(contents, filename):
    if not contents:
        raise PreventUpdate
    return dbc.Col([
        dbc.Row(dbc.Button(id='run-text-box', children='Run Text Recognition', color='success', size='sm'), justify='center', align='end')
    ], width=12)

def update_output_3(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents_2(list_of_contents, list_of_names)]
        return children

#run text recognition after clicking on the button
@app.callback(Output('text-boxes', 'children'),
             [Input('run-text-box', 'n_clicks'),
             Input('upload-image','filename')],
             [State('upload-image','contents')])

def run_script_onClick(n_clicks, filename, contents):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate
    
    # Load your output file with "some code"
    #global compression 
    #compression = compressor(PATH_TO_OD_LABELS=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels", PATH_TO_OD_MODEL_DIR=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection")
    
    #PIL image with retrieved background
    img=Image.fromarray(compression.image_np)
    text_boxes = compression.perform_ocr()
    fig=[dbc.Row(id='text_box_0')]
    for i in range(len(text_boxes)):
        box=text_boxes[i].get("box")
        box_crop = (box[1],box[0],box[3],box[2])
        #crop = img[box[1]:box[3], box[0]:box[2]]
        img_cropped=img.crop(box_crop)
        text=text_boxes[i].get("text")
        if len(text.replace(" ", "")) > 0 :
            fig.append(dbc.Row([
                dbc.Col(html.Img(src=img_cropped, id='text_box_' + str(i+1), height=100), width= 7),
                dbc.Col(dbc.Textarea(id= 'text_' + str(i+1), placeholder=text,
                className="text-center text-light font-weight-light, mb-4"), width=5)
            ]))
        #ax.text(3, 8, text, style='italic',
        #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    return fig

'''
@app.callback(Output('text-boxes', 'figure'),
             [Input('run-text-box', 'n_clicks'),
             Input('upload-image','filename')],
             [State('upload-image','contents')])

def run_script_onClick(n_clicks, filename, contents):
    # Don't run unless the button has been pressed...
    if not n_clicks:
        raise PreventUpdate
    
    # Load your output file with "some code"
    #global compression 
    #compression = compressor(PATH_TO_OD_LABELS=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection\\labels", PATH_TO_OD_MODEL_DIR=f"{home_path}\\Documents\\GitHub\\aicompression\\models\\object_detection")
    
    #PIL image with retrieved background
    img=compression.image_np
    text_boxes = compression.perform_ocr()
    fig=make_subplots(rows=len(text_boxes), cols=2)
    for i in range(len(text_boxes)):
        box=text_boxes[i].get("box")
        #box_crop = (box[1],box[0],box[3],box[2])
        crop = img[box[1]:box[3], box[0]:box[2]]
        #img_cropped=img.crop(box_crop)
        text=text_boxes[i].get("text")
        fig.add_trace(go.Image(z=crop), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=[0],y=[0],text=text, row=i+1, col=2))
        #ax.text(3, 8, text, style='italic',
        #bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    return fig'''

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False) 

