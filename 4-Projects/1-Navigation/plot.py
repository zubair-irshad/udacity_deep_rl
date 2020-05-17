import plotly
import plotly.graph_objs as go
import pandas as pd 
import numpy as np


def plot_results(baseline_score):
    df = pd.read_csv('/home/mirshad7/udacity_deep_rl/4-Projects/1- Navigation/logging/nav_dqn_scores_20200517-174735.csv')
    data= df.iloc[:].to_numpy().squeeze()
    iterations = np.linspace(1, len(df), len(df))
    N=4
    cumsum, moving_aves = [0], []
    #moving average
    for i, x in enumerate(data,1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)


    iterations_avg = np.linspace(1, len(moving_aves), len(moving_aves))
    #baseline
    baseline = np.ones(len(df))*baseline_score
    trace1 = go.Scatter(x=iterations, y=df.iloc[:].to_numpy().squeeze(),mode = "lines",name = "scores", marker = dict(color = 'rgba(65, 131, 215, 1)'))
    trace2 = go.Scatter(x=iterations_avg, y=moving_aves,mode = "lines",name = "moving average", marker = dict(color = 'rgba(102, 51, 153, 1)'))
    trace3 = go.Scatter(x=iterations, y=baseline,mode = "lines",line={'dash': 'dash'}, name = "Benchmark score", marker = dict(color = 'rgba(1, 50, 67, 1)'))

    fig = go.Figure()

    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.update_layout(template='plotly_white')
    plotly.io.write_image(fig,'/home/mirshad7/deep-reinforcement-learning/p1_navigation/logging/plot.jpg','jpg')
    fig.show()
