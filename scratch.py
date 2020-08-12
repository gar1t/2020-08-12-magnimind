import pandas as pd

import plotly.offline as py
import plotly.io as pio

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

pio.renderers.default = "png"

df = pd.read_csv("example_wp_log_peyton_manning.csv")
m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

#py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)
fig.show()

"""
import matplotlib.pyplot as plt

x = 2

plt.plot([4, 7, 9, 15])
plt.ylabel('some numbers')
plt.savefig("scatch.png")
"""
