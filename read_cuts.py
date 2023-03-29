import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = pd.read_csv("compare.csv")
print(df)
df_last = df.loc[df.name == "8m"].copy()
df = df.loc[df.name == "6m"].copy()
print(df_last)
print(df)
def get_stored(df, lg):
    return (np.sum(df.consumed_stock) * lg) - np.sum(df.total_linear)+ np.sum(df.total_waste)

fig = go.Figure()
fig.add_trace(go.Bar(x=["Total linear","Stock Consumed", "In Storage", "Waste"],
                    y = [np.sum(df.total_linear),np.sum(df.consumed_linear), get_stored(df, 6), np.sum(df.total_waste)],
                    name = df.name.iloc[0]))
fig.add_trace(go.Bar(x=["Total linear","Stock Consumed", "In Storage", "Waste"],
                    y = [np.sum(df.total_linear),np.sum(df_last.consumed_linear), get_stored(df_last, 8), np.sum(df_last.total_waste)],
                    name = df_last.name.iloc[0]))

# input()

fig.show()

pie = make_subplots(1,2, specs=[[{"type": "pie"}, {"type": "pie"}]])
pie.add_trace(go.Pie(labels = ["Total linear", "Linear waste", "Consumed linear", "Stored"],
                    values=[np.sum(df.total_linear), np.sum(df.total_waste), np.sum(df.consumed_linear), get_stored(df)],
                    domain=dict(x=[0, 0.5]),
                    title="6m"),1,1)
pie.add_trace(go.Pie(labels = ["Total linear", "Linear waste", "Consumed linear", "Stored"],
                    values=[np.sum(df.total_linear),np.sum(df_last.total_waste), np.sum(df_last.consumed_linear), get_stored(df_last,8)],
                    domain=dict(x=[0.5, 1]),
                    title="8m"),1,2)


pie.show()
