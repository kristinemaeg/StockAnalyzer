#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the libraries

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import math
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from datetime import date
from datetime import datetime
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import requests
import io
import plotly.express as px
import plotly.graph_objs as go
from techindicators import *

# Produce list of companies
url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))
companies["ticker_name"] = companies["Symbol"] + ": " + companies["Company Name"]

# Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Build the layout
colors = {
    'background': '#111111',
    'text': 'firebrick'
}

app.layout = html.Div([
    html.H4("Stock Analyzer",style={
        'textAlign': 'left-center',
        'color': colors['text'],
        'fontWeight': 'bold',
        }),
    dcc.Dropdown(
        id='stockTicker',
        options=[
            {'label':i, 'value':i} for i in companies['ticker_name'].unique()
        ],
        placeholder = "Select a Ticker",
        style={'width': 400},
    ),
    html.Br(),
    dcc.Dropdown(
        id='dataType',
        options=[
            {'label': 'Open', 'value': 'Open'},
            {'label': 'High', 'value': 'High'},
            {'label': 'Low', 'value': 'Low'},
            {'label': 'Close', 'value': 'Close'},
            {'label': 'Volume', 'value': 'Volume'}
        ],
        placeholder = "Select a Data Type",
        style={'width': 300}
    ),
    html.Br(),
    html.P('Select a Date Range'),
    dcc.DatePickerRange(
    id='dateRange',
    start_date_placeholder_text="Start Period",
    end_date_placeholder_text="End Period",
    calendar_orientation='vertical',
    display_format='Y-MM-DD',
    ),
    html.Br(),
    html.Br(),
    dcc.Dropdown(
        id='techInd',
        options=[
            {'label': 'Moving Averages', 'value': '1'},
            {'label': 'Accumulation/Distribution Line', 'value': '2'},
            {'label': 'Latest Values', 'value': '3'},
            {'label': 'Moving Average Convergence/Divergence', 'value': '4'},
            {'label': 'Percentage Price Oscillator', 'value': '5'},
            {'label': 'TRIX', 'value': '6'},
            {'label': 'Keltner Channels', 'value': '7'},
            {'label': 'Bollinger Bands', 'value': '8'},
            {'label': 'Stochastic Oscillator', 'value': '9'},
            {'label': 'Vortex Indicator', 'value': '10'},
            {'label': 'Average Directional Index (ADX)', 'value': '11'},
            {'label': 'Aroon Oscillator', 'value': '12'},
            {'label': 'Chandelier Exits', 'value': '13'},
            {'label': 'Coppock Curve', 'value': '14'},
            {'label': 'Force Index', 'value': '15'},
            {'label': 'Chaikin Money Flow (CMF)', 'value': '16'},
            {'label': 'Chaikin Oscillator', 'value': '17'},
            {'label': 'Ease of Movement (EMV)', 'value': '18'},
            {'label': 'Mass Index', 'value': '19'},
            {'label': 'Money Flow Index (MFI)', 'value': '20'},
            {'label': 'Negative Volume Index (NVI)', 'value': '21'},
            {'label': 'On Balance Volume (OBV)', 'value': '22'},
            {'label': 'Percentage Volume Oscillator', 'value': '23'},
            {'label': "Pring's Know Sure Thing (KST)", 'value': '24'}
        ],
        placeholder = "Select a Technical Indicator",
        style={'width': 400}
    ),
    html.P(['If the graphs do not show up, the date range may be too small for the selected Technical Indicator.',
           html.Br(), 'Expand the date range or select a different Technical Indicator to view the results.']),
    html.Br(),
    html.Button('Analyze', id='button', n_clicks=0),
    html.Div(id='td-output-container')
])


@app.callback(
    dash.dependencies.Output('td-output-container', 'children'),
    dash.dependencies.Input('button', 'n_clicks'),
    dash.dependencies.Input('stockTicker', 'value'),
    dash.dependencies.Input('dataType', 'value'),
    dash.dependencies.Input('dateRange', 'start_date'),
    dash.dependencies.Input('dateRange', 'end_date'),
    dash.dependencies.Input('techInd', 'value')
    )
def update_output(n_clicks,stockTicker, dataType, start_date, end_date, techInd):
    if n_clicks > 0:
        stockTicker = stockTicker.split(':')[0] 
        if start_date is not None:
            start_date_object = date.fromisoformat(start_date)
        if end_date is not None:
            end_date_object = date.fromisoformat(end_date)

        #Historical Data Line Graph
        df = yf.download(stockTicker,
                      start=start_date, 
                      end=end_date, 
                      progress=False)
        fig = px.line(df[dataType])
        
        if dataType == 'Volume':
            fig.update_layout(
                title="Historical Data",
                xaxis_title="Date",
                yaxis_title="Total",
                legend_title="Variable"
            )
        else:
            fig.update_layout(
                title="Historical Data",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
        histData = html.Div([dcc.Graph(figure=fig)])
        
        
        # Technical Indicator Graphs
        df = yf.download(stockTicker,
                      start=start_date, 
                      end=end_date, 
                      progress=False)

        df.to_csv('stockprice.csv', header=None)
        
        stockdata = np.genfromtxt('stockprice.csv', delimiter=',')
        sd_open = stockdata[:,1] # Open
        sd_high = stockdata[:,2] # High
        sd_low = stockdata[:,3] # Low
        sd_close = stockdata[:,4] # Close
        sd_adjclose = stockdata[:,5] # Adj Volume
        sd_volume = stockdata[:,6] # Volume
        sd_dates = np.loadtxt('stockprice.csv', delimiter=',', usecols=(0), dtype='datetime64[D]') # Dates
        tradedays = np.arange(len(sd_close)) # Array of number of trading days
        
        if techInd == "1":
        # Calculate the Simple Moving Average
            sma50 = sma(sd_close,50) # calculate 50 day SMA of closing price
            wma50 = wma(sd_close,50) # calculated 50 day WMA of closing price
            ema20 = ema(sd_close,20) # calculate 20 day EMA of closing price
            kama_sd = kama(sd_close,10,2,30) # calculate standard Kaufman adaptive moving average
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(sd_dates.astype(datetime)), y=list(sd_close), name="Close"
                    ),
                    go.Scatter(
                        x=list(sd_dates[len(sd_dates)-len(sma50):].astype(datetime)), y=list(sma50), 
                                name="50 Day Simple Moving Average"
                    ),
                    go.Scatter(
                        x=list(sd_dates[len(sd_dates)-len(wma50):].astype(datetime)), y=list(wma50), 
                                name="50 Day Weighted Moving Average"
                    ),
                    go.Scatter(
                        x=list(sd_dates[len(sd_dates)-len(ema20):].astype(datetime)), y=list(ema20), 
                                name="50 Day Exponential Moving Average"
                    ),
                     go.Scatter(
                        x=list(sd_dates[len(sd_dates)-len(kama_sd):].astype(datetime)), y=list(kama_sd), 
                                name="Kaufman's Adaptive Moving Average"
                    )
                ])
            
            fig.update_layout(
                title="Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        elif techInd == "2":
        # Calculate the Accumulation/Distribution line
            adl_sd = adl(sd_high,sd_low,sd_close,sd_volume)
        
            fig = px.line(x = tradedays, y = adl_sd)
            
            if dataType == 'Volume':
                fig.update_layout(
                    title="Accumulation/Distribution Line",
                    xaxis_title="Trading Days in Time Period",
                    yaxis_title="Total",
                    legend_title="Variable"
            )
            else:
                fig.update_layout(
                    title="Accumulation/Distribution Line",
                    xaxis_title="Trading Days in Time Period",
                    yaxis_title="Price",
                    legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        elif techInd == "3":
        # Display the Latest Values
            cci20 = cci(sd_high,sd_low,sd_close,20) # 20-day commodity channel index
            atr14 = atr(sd_high,sd_low,sd_close,14) # 14-day average true range
            rsi14 = rsi(sd_close,14) # 14-day relative strength index
            rstd10 = rstd(sd_close,10) # 10-day rolling standard deviation
            roc12 = roc(sd_close,12) # 12-day rate of change
            
            var_cci = "{:.2f}".format(cci20[-1])
            var_atr = "{:.2f}".format(atr14[-1])
            var_rsi = "{:.2f}".format(rsi14[-1])
            var_rstd = "{:.2f}".format(rstd10[-1])
            var_roc = "{:.2f}".format(roc12[-1])
            
            techIndGraph = html.Div([html.H6("Latest Values",style={'textAlign': 'left-center',}), html.Br(),
                               html.P(['The 20-day Commodity Channel Index (CCI) was: ' + var_cci,
                                       html.Br(), 'The 14-day Average True Range (ATR) was: ' + var_atr,
                                       html.Br(), 'The 14-day Relative Strength Index (RSI) was: ' + var_rsi,
                                       html.Br(), 'The 10-day Rolling Standard Deviation was: ' + var_rstd,
                                       html.Br(), 'The 12-day Rate of Change was: ' + var_roc,
                                       html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br()], 
                                      style={'margin':'2%'}),])

        
        elif techInd == "4":
        # Calculate Moving Average Convergence/Divergence
            macd_line_sd = macd(sd_close,12,26,9)[0]
            macd_signal_sd = macd(sd_close,12,26,9)[1]
            macd_histogram_sd = macd_line_sd[len(macd_line_sd)-len(macd_signal_sd):]-macd_signal_sd
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(macd_line_sd):]), y=list(macd_line_sd), name="Line"
                    ),
                     go.Scatter(
                        x=list(tradedays[len(tradedays)-len(macd_signal_sd):]), y=list(macd_signal_sd), 
                                name="Signal"
                    )
                ])
            
            fig.update_layout(
                title="Moving Average Convergence/Divergence",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        elif techInd == "5":
        # Calculate Percentage Price Oscillator
            ppo_line_sd = ppo(sd_close,12,26,9)[0]
            ppo_signal_sd = ppo(sd_close,12,26,9)[1]
            ppo_histogram_sd = ppo_line_sd[len(ppo_line_sd)-len(ppo_signal_sd):]-ppo_signal_sd
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(ppo_line_sd):]), y=list(ppo_line_sd), name="Line"
                    ),
                     go.Scatter(
                        x=list(tradedays[len(tradedays)-len(ppo_signal_sd):]), y=list(ppo_signal_sd), 
                                name="Signal"
                    )
                ])
            
            fig.update_layout(
                title="Percentage Price Oscillator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "6":
        # Calculate TRIX
            trix_line_sd = trix(sd_close,15,9)[0]
            trix_signal_sd = trix(sd_close,15,9)[1]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(trix_line_sd):]), y=list(trix_line_sd), name="Line"
                    ),
                     go.Scatter(
                        x=list(tradedays[len(tradedays)-len(trix_signal_sd):]), y=list(trix_signal_sd), 
                                name="Signal"
                    )
                ])
            
            fig.update_layout(
                title="TRIX",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "7":
        # Calculate Keltner Channels    
            kelt_sd = kelt(sd_high,sd_low,sd_close,20,2.0,10) # Kelter Channel calculated with standard parameters
            lowl = kelt_sd[0] # lower line
            cenl = kelt_sd[1] # center line
            uppl = kelt_sd[2] # upper line
        
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=list(tradedays),
                        open=list(sd_open),
                        high=list(sd_high),
                        low=list(sd_low),
                        close=list(sd_close),
                        name="Candlestick",
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(lowl):]), y=list(lowl), name="Keltner Channels (20,2,10)"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(uppl):]), y=list(uppl), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(cenl):]), y=list(cenl), name="", 
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Keltner Channels",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
                
        
        elif techInd == "8":
        # Calculate Bollinger Bands    
            boll_sd = boll(sd_close,20,2.0,20) # Bollinger Bands calculated with standard parameters
            lowlb = boll_sd[0] # lower line
            cenlb = boll_sd[1] # center line
            upplb = boll_sd[2] # upper line
        
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=list(tradedays),
                        open=list(sd_open),
                        high=list(sd_high),
                        low=list(sd_low),
                        close=list(sd_close),
                        name="Candlestick",
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(lowlb):]), y=list(lowlb), name="Bollinger Bands (20,2,20)"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(upplb):]), y=list(upplb), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(cenlb):]), y=list(cenlb), name="", 
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Bollinger Bands",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
    
    
        
        elif techInd == "9":
        # Calculate Stochastic Oscillator
            stoch_sd = stoch(sd_high,sd_low,sd_close,14,3,3) # Full stochastics calculated with standard parameters
            stoch_k = stoch_sd[0] # %K parameter
            stoch_d = stoch_sd[1] # %D parameter
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(stoch_k):]), y=list(stoch_k), name="%K"
                    ),
                     go.Scatter(
                        x=list(tradedays[len(tradedays)-len(stoch_d):]), y=list(stoch_d), 
                                name="%D"
                    )
                ])
            
            fig.update_layout(
                title="Stochastic Oscillator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "10":
        # Calculate Vortex Indicator
            vort_sd = vortex(sd_high,sd_low,sd_close,14)
            vort_p_sd = vort_sd[0]
            vort_n_sd = vort_sd[1]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(vort_p_sd):]), y=list(vort_p_sd), name="+VM"
                    ),
                     go.Scatter(
                        x=list(tradedays[len(tradedays)-len(vort_n_sd):]), y=list(vort_n_sd), 
                                name="$-$VM"
                    )
                ])
            
            fig.update_layout(
                title="Vortex Indicator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "11":
        # Calculate Average Directional Index (ADX)
            adx_sd = adx(sd_high,sd_low,sd_close,14)
            adx_pdm = adx_sd[0]
            adx_ndm = adx_sd[1]
            adx_line = adx_sd[2]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(adx_pdm):]), y=list(adx_pdm), name="$+$DI"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(adx_ndm):]), y=list(adx_ndm), name="$-$DI"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(adx_line):]), y=list(adx_line), 
                                name="ADX"
                    )
                ])
            
            fig.update_layout(
                title="Average Directional Index (ADX)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "12":
        # Calculate Aroon Oscillator
            aroon_sd = aroon(sd_high,sd_low,25)
            aroon_up = aroon_sd[0]
            aroon_down = aroon_sd[1]
            aroon_osc = aroon_sd[2]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(aroon_up):]), y=list(aroon_up), name="Up"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(aroon_down):]), y=list(aroon_down), 
                                name="Down"
                    )
                ])
            
            fig.update_layout(
                title="Aroon Oscillator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "13":
        # Calculate Chandelier Exits    
            chand_long = chand(sd_high,sd_low,sd_close,22,3,'long')
            chand_short = chand(sd_high,sd_low,sd_close,22,3,'short')
        
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=list(tradedays),
                        open=list(sd_open),
                        high=list(sd_high),
                        low=list(sd_low),
                        close=list(sd_close),
                        name="Candlestick",
                    ),
                    go.Scatter(
                        x=list(tradedays[len(sd_dates)-len(chand_long):]), y=list(chand_long), name="Chandelier Exit (22,3)"
                    )
                ])
            
            fig.update_layout(
                title="Chandelier Exits",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        elif techInd == "14":
        # Calculate Coppock Curve
            copp_sd = copp(sd_close,14,11,10)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(copp_sd):]), y=list(copp_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(copp_sd):]), y=list(np.zeros(len(copp_sd))), name="", 
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Coppock Curve",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        elif techInd == "15":
        # Calculate Force Index
            force_sd = force(sd_close,sd_volume,13)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(force_sd):]), y=list(force_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(force_sd):]), y=list(np.zeros(len(force_sd))), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Force Index",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "16":
        # Calculate Chaikin Money Flow (CMF)
            cmf_sd = cmf(sd_high,sd_low,sd_close,sd_volume,20)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(cmf_sd):]), y=list(cmf_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(cmf_sd):]), y=list(np.zeros(len(cmf_sd))), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Chaikin Money Flow (CMF)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        elif techInd == "17":
        # Calculate Chaikin Oscillator
            chosc_sd = chosc(sd_high,sd_low,sd_close,sd_volume,3,10)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(chosc_sd):]), y=list(chosc_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(chosc_sd):]), y=list(np.zeros(len(chosc_sd))), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Chaikin Oscillator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "18":
        # Calculate Ease of Movement (EMV)
            emv_sd = emv(sd_high,sd_low,sd_volume,14)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(emv_sd):]), y=list(emv_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(emv_sd):]), y=list(np.zeros(len(emv_sd))), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Ease of Movement (EMV)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "19":
        # Calculate Mass Index
            mindx_sd = mindx(sd_high,sd_low,25)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mindx_sd):]), y=list(mindx_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mindx_sd):]), y=list(np.zeros(len(mindx_sd))+27), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Mass Index",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "20":
        # Calculate Money Flow Index (MFI)
            mfi_sd = mfi(sd_high,sd_low,sd_close,sd_volume,14)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mfi_sd):]), y=list(mfi_sd), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mfi_sd):]), y=list(np.zeros(len(mfi_sd))+50), name=""
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mfi_sd):]), y=list(np.zeros(len(mfi_sd))+20), name="",
                        line=dict(dash="dot")
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(mfi_sd):]), y=list(np.zeros(len(mfi_sd))+80), name="",
                        line=dict(dash="dot")
                    )
                ])
            
            fig.update_layout(
                title="Money Flow Index (MFI)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Price",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
            
        elif techInd == "21":
        # Calculate Negative Volume Index (NVI)
            nvi_sd = nvi(sd_close,sd_volume,50)
            nvi_line = nvi_sd[0]
            nvi_signal = nvi_sd[1]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(nvi_line):]), y=list(nvi_line), name="NVI"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(nvi_signal):]), y=list(nvi_signal), 
                                name="Signal (50 day EMA)"
                    )
                ])
            
            fig.update_layout(
                title="Negative Volume Index (NVI)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Total",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
            
        
        elif techInd == "22":
        # Calculate On Balance Volume (OBV)
            obv_sd = obv(sd_close,sd_volume)
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays), y=list(obv_sd), name=""
                    )
                ])
            
            fig.update_layout(
                title="On Balance Volume (OBV)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Total",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)]) 
            
            
        elif techInd == "23":
        # Calculate Percentage Volume Oscillator
        # DID NOT INCLUDE HISTOGRAM
            pvo_line_sd = pvo(sd_close,12,26,9)[0]
            pvo_signal_sd = pvo(sd_close,12,26,9)[1]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(pvo_line_sd):]), y=list(pvo_line_sd), name="Line"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(pvo_signal_sd):]), y=list(pvo_signal_sd), 
                                name="Signal"
                    )
                ])
            
            fig.update_layout(
                title="Percentage Volume Oscillator",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Total",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        elif techInd == "24":
        # Calculate Pring's Know Sure Thing (KST)
            kst_sd = kst(sd_close,10,15,20,30,10,10,10,15,9)
            kst_line = kst_sd[0]
            kst_signal = kst_sd[1]
            
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(kst_line):]), y=list(kst_line), name="Line"
                    ),
                    go.Scatter(
                        x=list(tradedays[len(tradedays)-len(kst_signal):]), y=list(kst_signal), 
                                name="Signal"
                    )
                ])
            
            fig.update_layout(
                title="Pring's Know Sure Thing (KST)",
                xaxis_title="Trading Days in Time Period",
                yaxis_title="Total",
                legend_title="Variable"
            )
        
            techIndGraph = html.Div([dcc.Graph(figure=fig)])
        
        
        # Display Graphs
        allCharts = html.Div([ 
            html.Div(histData),
            html.Div(techIndGraph)
        ])
        
    return allCharts

if __name__ == '__main__':
    app.run_server(debug=False)

