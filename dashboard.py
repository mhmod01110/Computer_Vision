import base64
import io
import os

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load and preprocess image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((512, 512))  # Resize image as needed
    img = np.array(img) / 255.0  # Normalize pixel values
    return img


loaded_model = load_model('./Fine_tuned_U_NET_Model.h5')

# Function to perform segmentation
def segment_image(image):
    segmentation = loaded_model.predict(image)
    return segmentation

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    html.H1("Retinal Vessel Segmentation"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Image')
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
                multiple=False
            ),
            html.Div(id='output-image-upload'),
            html.Button('Predict', id='predict-button', n_clicks=0, className='btn btn-primary', style={'margin': '10px'}),
        ], width=5),
        dbc.Col([
            html.Div(id='output-image-segmentation')
        ], width=7)
    ])
], fluid=True)

# Callback to upload image and display
@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(content, filename):
    if content is not None:
        return html.Div([
            html.H5(filename),
            html.Img(src=content, style={'width': '100%'})
        ])

# Callback to perform segmentation and display result
@app.callback(
    Output('output-image-segmentation', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('output-image-upload', 'children')]
)
def predict_segmentation(n_clicks, image_div):
    if n_clicks == 0:
        raise PreventUpdate

    if image_div is None:
        return html.Div("Please upload an image first.")

    content_str = image_div[1]['props']['children'][1]['props']['src'].split(",")[-1]
    image_bytes = base64.b64decode(content_str)
    img = preprocess_image(image_bytes)
    mask = segment_image(img)

    # Convert mask to image
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img_bytes = io.BytesIO()
    mask_img.save(mask_img_bytes, format='PNG')
    encoded_mask = base64.b64encode(mask_img_bytes.getvalue()).decode('utf-8')

    return html.Div([
        html.H5("Predicted Mask"),
        html.Img(src=f"data:image/png;base64,{encoded_mask}", style={'width': '100%'})
    ])

if __name__ == '__main__':
    app.run_server(debug=True)

