#!/usr/bin/env python3

import time
import datetime
import matplotlib
# import mpld3
import maya
import json
import pprint as pp
import pandas as pd
import numpy as np
from fbprophet import Prophet
from decimal import Decimal

# Read the CSV with all the data.
data = pd.read_csv("./snp500historical.csv")

# The 'y' field should be the closing price.
data['y'] = data['Close']

# Add the timestamp to the 'ds' field.
data['ds'] = data['Date']
timestamps = [time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d").timetuple()) \
              for d in data['Date']]

# Create the prophet model.
model = Prophet()

# And then fit it with the data.
model.fit(data)

# This generates the future dataframe.
future = model.make_future_dataframe(periods=26, freq="w")

# And then, based on that dataframe we generate a forecast.
forecast = model.predict(future)

# Finally, we create the matplot figure.
figure = model.plot(forecast, xlabel='Date', ylabel='Points')

# Get the D3 HTML graph.
# with open("test.html", "w") as file:
#     fig = pd.Series(figure).to_json()
#     html = mpld3.fig_to_html(fig)
#     file.write(html)

# And then save it to an image file.
matplotlib.pyplot.savefig("out.png")
pp.pprint(figure)

# Components figure.
figure_components = model.plot_components(forecast)

# And then save it to an image file.
matplotlib.pyplot.savefig("out-components.png")
pp.pprint(figure)

with open("predictions.json", "w") as file:
    forecast_data_orig = forecast
    #forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
    forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
    forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])

    # print(forecast_data_orig)
    d = forecast_data_orig['yhat'].to_dict()
    predictions = []

    for i, k in enumerate(list(d.keys())[-30:]):
        w = maya.when(f'{i+1} days from now')
        predictions.append({
            'when': w.slang_time(),
            'timestamp': w.iso8601(),
            'usd': d[k]
        })

    # Dump the JSON into the file.
    json.dump(predictions, file)
