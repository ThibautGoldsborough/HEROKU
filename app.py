from dash import Dash, dcc, html, Input, Output
import os
import numpy as np

from scipy.interpolate import interpn
from scipy.interpolate import interpn   

import io
import base64

from PIL import Image
import dash
import flask
from dash import dcc, html, Input, Output, no_update
#import plotly.graph_objects as go
import plotly.graph_objs as go
import pandas as pd

basepath = os.getcwd()+"/DATA"
readpath = basepath
outpath = basepath


channel_dict={'Ch11': 'AF647 PDC (Photoreceptors)','Ch10': 'CA10-BV605', 'Ch07': "SYTO40 (DNA)"}

def density_scatter( x , y, sort = True, bins = 20 )   :
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    z[np.where(np.isnan(z))] = 0.0
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    return x,y,z


def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


latent_df=pd.read_csv(outpath+"/Resnet0.35477188.csv")
namesdf=latent_df["Cell_ID"].to_numpy()
display_images=np.load(outpath+"/Ch1.npy")



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets,update_title=None,title="Single-cell analyst")

server = app.server

x=latent_df["SCALED_Intensity_MC_"+"Ch11"].to_numpy()
y=latent_df["PRED_Intensity_MC_"+"Ch11"].to_numpy()
x,y,z=density_scatter( x , y, sort = False, bins =[40,40])

         
u=np.vstack((x,y)).T
colors=z


dim=np.shape(u)[1]

if np.min(display_images)<0:
    display_images=np.array((np.array(display_images)-np.min(display_images)))

if str(type(display_images[0][0][0]))=="<class 'numpy.float64'>":
    display_images=display_images.astype(np.uint8)


fig = go.Figure(data=[go.Scatter(
    x=u[:, 0],
    y=u[:, 1],
    hovertemplate='<extra></extra>',
    mode='markers',
    marker=dict(
        size=3,
        color=colors,
        colorscale="Spectral_r"
    )
)])

fig.add_trace(go.Scatter(x=u[:, 1], y=u[:, 1],
                    mode='lines',
                    name='lines',
                    hovertemplate='<extra></extra>'))

fig.update_layout(
    title="AF647 PDC (Photoreceptors)",
    xaxis_title="Actual fluorescence",
    yaxis_title="Predicted fluorescence")

fig.update(layout_showlegend=False)
app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(style={'height': '800px'}, id="graph-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ], style={'width': '80%' , 'display': 'inline-block', 'vertical-align': 'middle'}
) 

@app.callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)

def display_hover(hoverData):
    print(hoverData)
    if hoverData is None:
        return False, no_update, no_update


    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = display_images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "100px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("Cell ID: " + str(int(namesdf[num])))#, style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children
    
if __name__ == '__main__':
    app.run_server(debug=True)

