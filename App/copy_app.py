import dash
from dash.dependencies import Input, Output, State
import datetime
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from PIL import Image
import requests

# Dash component wrappers
def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("logo_cs.png"), style={"float": "right", "height": 50}
    )
    link = html.A(logo, href="https://www.centralesupelec.fr/")
    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=0)])

def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)

#Useful functions


#get & display image infos
def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content')

    ])



# Start Dash
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

app.layout = html.Div(className='container', children=[
    Header("AI Compression", app),

html.Div([
    dcc.Tabs(id="menu", value='tab-1', children=[
        dcc.Tab(label='Object Detection', value='tab-1'),
        dcc.Tab(label='Background Retrieval', value='tab-2'),
        dcc.Tab(label='NLP', value='tab-3'),
        dcc.Tab(label='Performance', value='tab-4'),
        dcc.Tab(label='Final Results', value='tab-5'),
        dcc.Tab(label='About', value='tab-4'),
    ], colors={
        "border": 'gold',
        "primary": "blue",
        "background": "#4C78A8"
    }, vertical=False),
    html.Div(id='menu-content')
]),

    Row(html.P("Upload Image:")),
    Row([
        Column(width=8, children=[
            dcc.Upload(
                id='upload-image',
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
                    'margin': '10px'
                },
                # Don't allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-image-upload')
        ]),
    #html.Button('Upload Image', width = 2)
    ]),



])


#Actions


#display image

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
              State('upload-image', 'last_modified')])
def update_output(content, name, date):
    if content is not None:
        children = [
            parse_contents(content, name, date) ]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
