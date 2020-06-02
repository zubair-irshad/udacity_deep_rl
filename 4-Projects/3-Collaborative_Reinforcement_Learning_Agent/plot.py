import plotly
import plotly.graph_objs as go
import pandas as pd 
import numpy as np


def plot_results(benchmark_score=0.5):
    score_data = pd.read_csv('./logging/ma_ddpg_agent_score_1276.csv')
    rolling_avg_data = pd.read_csv('./logging/ma_ddpg_agent_rolling_avg_1276.csv')

    score_data= score_data.iloc[:].to_numpy().squeeze()
    iterations = np.linspace(1, len(score_data), len(score_data))

    rolling_avg_data= rolling_avg_data.iloc[:].to_numpy().squeeze()
    rolling_avg_iterations = np.linspace(1, len(rolling_avg_data), len(rolling_avg_data))

    #baseline
    baseline = np.ones(len(score_data))*benchmark_score
    trace1 = go.Scatter(x=iterations, y=score_data,mode = "lines",name = "scores", marker = dict(color = 'rgba(65, 131, 215, 1)'))
    trace2 = go.Scatter(x=rolling_avg_iterations , y=rolling_avg_data,mode = "lines",name = "moving average", marker = dict(color = 'rgba(102, 51, 153, 1)'))
    trace3 = go.Scatter(x=iterations, y=baseline,mode = "lines",line={'dash': 'dash'}, name = "Benchmark score", marker = dict(color = 'rgba(1, 50, 67, 1)'))

    fig = go.Figure()

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.update_layout(template='plotly_white')
    fig.update_layout(
        title="Average rewards",
        xaxis_title="Number of episodes",
        yaxis_title="Average Reward",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    plotly.io.write_image(fig,'./logging/plot.jpg','jpg')
    fig.show()
