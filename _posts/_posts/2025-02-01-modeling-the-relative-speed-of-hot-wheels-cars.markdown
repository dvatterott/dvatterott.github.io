---
layout: post
title: "Modeling the relative speed of Hot Wheels Cars"
date: 2025-02-01 08:49:00 -0600
comments: true
categories: [open source, data science, statistics, python]
---

I recently came across a really interesting modeling problem when playing with my kids. It all started when my 5yr old received this [Hot Wheels T-Rex Transporter](https://www.amazon.com/Hot-Wheels-Ultimate-Transforms-Stomping/dp/B0BN15GCB7/).

<div>
  <div float:left>
    <img src="{{ root_url }}/images/misc/hot_wheels_dino1.jpeg" />
  </div>
  <div float:left>
    <img src="{{ root_url }}/images/misc/hot_wheels_dino2.jpeg" />
  </div>
</div>


The transporter comes with a race-track where the winner triggers a trap door, dropping the loser to the ground. Obviously our first task when playing with this toy was determining which car was fastest. I figured the system would be relatively deterministic and finding the fastest car would be a simple matter of racing each car and passing the winner of a given race onto the next race. The final winner should be the fastest car.

I noticed pretty quickly that the race outcomes were much more stochastic than I anticipated (even when controlling for race-lane and censoring bad starts). Undeterred, I realized I would need to model the race outcomes in order to estimate the fastest car.

In this modeling problem there's a couple constraints that make the problem a little more interesting:
 1) I'm only going to record the outcome of races a limited number of times. I have limited patience. My 5yr old is similar. My 3yr old, has less.
 2) I have limited control over which cars my kids choose to race.

The easiest way to handle this problem would be to race each car against each competitor N times (balancing for race-lane across trials). I could estimate N by racing two cars against each other repeatedly to estimate the race-to-race variance.

Needless to say, this wasn't going to happen. My 5yr old is going to choose the fastest car that she's willing to risk losing. My 3yr old is going to choose her favorite car. Repeatedly.

Another interesting feature of this problem is the race cars compete against each other. This means in a given race, I can learn about both the cars. I wanted to create a model that took advantage of this feature.

Okay, before I get too deep into the problem, I should introduce the 12 cars. Here they are.

<img src="{{ root_url }}/images/misc/hot_wheels_cars.jpeg" />

We had the patience to record 37 races (not bad!). That said, let's get into it.


{% codeblock lang:python %}
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
register_matplotlib_converters()

pct_formatter = FuncFormatter(lambda prop, _: "{:.1%}".format(prop))
params = {"ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k",
          "figure.figsize": (10,6),
          "axes.facecolor": [0.9, 0.9, 0.9],
          "axes.grid": True,
          "axes.labelsize": 10,
          "axes.titlesize": 10,
          "xtick.labelsize": 8,
          "ytick.labelsize": 8,
          "legend.fontsize": 8,
          "legend.facecolor": "w"}

%matplotlib inline
plt.rcParams.update(params)

# helper funcs
def getData():
    return (pd.read_csv('./data/HotWheels.csv')
            .assign(Winner=lambda x: x['Winner'] - 1)
            )

def removeLaneCoding(tmp, num):
    return (tmp[[f'Lane{num}', 'Winner']]
            .assign(won=lambda x: x['Winner'] == (num-1))
            .rename(columns={f'Lane{num}': 'racer'})
            .drop(columns=['Winner'])
            )

def explodeRows():
    tmp = getData()
    tmp = pd.concat([
        removeLaneCoding(tmp, 1),
        removeLaneCoding(tmp, 2)
    ])
    return tmp
{% endcodeblock %}

Here's a quick look at the data. I originally encoded the car in lane 1, the car in lane 2, and which lane won. I updated the final column to whether lane2 won which will make modeling a little easier.

{% codeblock lang:python %}
df = getData()
print(len(df))
df.head()
{% endcodeblock %}

    37

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>&#xa0;</th>
<th>Lane1</th>
<th>Lane2</th>
<th>Winner</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Spikey</td>
<td>Mom</td>
<td>1</td>
</tr>

<tr>
<td>1</td>
<td>Whitey</td>
<td>Orange</td>
<td>1</td>
</tr>

<tr>
<td>2</td>
<td>BlackBlur</td>
<td>CandyCane</td>
<td>0</td>
</tr>

<tr>
<td>3</td>
<td>Reddy</td>
<td>Nick</td>
<td>0</td>
</tr>
<tr>
<td>4</td>
<td>Yellow</td>
<td>Spidey</td>
<td>1</td>
</tr>
</tbody>
</table>
</div>

One of my first questions is which lane is faster. Lane2 won 21 of 37 races (57%). Hard to draw too much of a conclusion here though because I made no attempt to counterbalance cars across the lanes.

{% codeblock lang:python %}
(getData()
 .assign(_=lambda x: 'Lane2')
 .groupby('_')
 .agg({'Winner': ['sum', 'count', 'mean']})
 .reset_index()
 )
{% endcodeblock %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>&#xa0;</th>
<th>_</th>
<th>Winner</th>
<th>&#xa0;</th>
<th>&#xa0;</th>
</tr>
</thead>
<tbody>
<tr>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>sum</td>
<td>count</td>
<td>mean</td>
</tr>

<tr>
<td>0</td>
<td>Lane2</td>
<td>21</td>
<td>37</td>
<td>0.567568</td>
</tr>
</tbody>
</table>
</div>

A natural next question is to look at the performance of the individual cars. To do this, I need to transform the data such that each race is two rows representing the outcome from each car's perspective.

{% codeblock lang:python %}
df = explodeRows()
print(len(df))
df.head()
{% endcodeblock %}
    74

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>&#xa0;</th>
<th>racer</th>
<th>won</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Spikey</td>
<td>False</td>
</tr>

<tr>
<td>1</td>
<td>Whitey</td>
<td>False</td>
</tr>

<tr>
<td>2</td>
<td>BlackBlur</td>
<td>True</td>
</tr>

<tr>
<td>3</td>
<td>Reddy</td>
<td>True</td>
</tr>

<tr>
<td>4</td>
<td>Yellow</td>
<td>False</td>
</tr>
</tbody>
</table>
</div>

Now to group by car at look at the number of races, number of wins, and proportion of wins.

{% codeblock lang:python %}
(explodeRows()
 .groupby('racer')
 .agg({'won': ['sum', 'count', 'mean']})
 .reset_index()
 .sort_values('racer')
 )
{% endcodeblock %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>&#xa0;</th>
<th>racer</th>
<th>won</th>
<th>&#xa0;</th>
<th>&#xa0;</th>
</tr>
</thead>
<tbody>
<tr>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>sum</td>
<td>count</td>
<td>mean</td>
</tr>

<tr>
<td>0</td>
<td>BigChain</td>
<td>4</td>
<td>5</td>
<td>0.800000</td>
</tr>

<tr>
<td>1</td>
<td>BlackBlur</td>
<td>5</td>
<td>8</td>
<td>0.625000</td>
</tr>

<tr>
<td>2</td>
<td>Bluey</td>
<td>6</td>
<td>9</td>
<td>0.666667</td>
</tr>

<tr>
<td>3</td>
<td>CandyCane</td>
<td>5</td>
<td>15</td>
<td>0.333333</td>
</tr>

<tr>
<td>4</td>
<td>Mom</td>
<td>3</td>
<td>4</td>
<td>0.750000</td>
</tr>

<tr>
<td>5</td>
<td>Nick</td>
<td>3</td>
<td>7</td>
<td>0.428571</td>
</tr>

<tr>
<td>6</td>
<td>Orange</td>
<td>7</td>
<td>10</td>
<td>0.700000</td>
</tr>

<tr>
<td>7</td>
<td>Reddy</td>
<td>1</td>
<td>3</td>
<td>0.333333</td>
</tr>

<tr>
<td>8</td>
<td>Spidey</td>
<td>2</td>
<td>4</td>
<td>0.500000</td>
</tr>

<tr>
<td>9</td>
<td>Spikey</td>
<td>0</td>
<td>2</td>
<td>0.000000</td>
</tr>

<tr>
<td>10</td>
<td>Whitey</td>
<td>0</td>
<td>2</td>
<td>0.000000</td>
</tr>

<tr>
<td>11</td>
<td>Yellow</td>
<td>1</td>
<td>5</td>
<td>0.200000</td>
</tr>
</tbody>
</table>
</div>

To create the model, I need to numerically encode the cars' identities. The only tricky thing here is not all cars have raced in both lanes.

{% codeblock lang:python %}
df = getData()
racer_set = set(df['Lane1'].tolist() + df['Lane2'].tolist())
racers = {x: i for i, x in enumerate(racer_set)}
lane1_idx = np.array(df['Lane1'].map(racers))
lane2_idx = np.array(df['Lane2'].map(racers))
coords = {'racer': racers}
print(coords)
{% endcodeblock %}

{'racer': {'Whitey': 0, 'Yellow': 1, 'Bluey': 2, 'BigChain': 3, 'CandyCane': 4, 'Spidey': 5, 'Nick': 6, 'Spikey': 7, 'Mom': 8, 'Orange': 9, 'BlackBlur': 10, 'Reddy': 11}}


Now to design a model. The outcome is whether lane 2 won the race. The inputs are an intercept term (which I name lane2) and the quality of the two cars in the race. I figured these cars would be compared by subtracting one from another. I think this makes more sense than something like a ratio.....but feel free to let me know if you think otherwise.

EQUATION

{% codeblock lang:python %}
import pymc as pm
import arviz as az

with pm.Model(coords=coords) as model:
    # global model parameters
    lane2 = pm.Normal('lane2', mu=0, sigma=2)
    sd_racer = pm.HalfNormal('sd_racer', sigma=1)

    # team-specific model parameters
    mu_racers = pm.Normal('mu_racer', mu=0, sigma=sd_racer, dims='racer')

    # model
    μ = lane2 + mu_racers[lane2_idx] - mu_racers[lane1_idx]
    θ = pm.math.sigmoid(μ)
    yl = pm.Bernoulli('yl', p=θ, observed=df['Winner'])
    trace = pm.sample(2000, chains=4)
{% endcodeblock %}

{% codeblock lang:python %}
az.plot_trace(trace, var_names=["lane2"], compact=False);
{% endcodeblock %}

<img src="{{ root_url }}/images/misc/hot_wheels_trace.png" />

{% codeblock lang:python %}
az.summary(trace, kind="diagnostics")
{% endcodeblock %}

Onto results. First, is lane 2 faster. Here's a table with of lane2's (the intercept's) posterior.

{% codeblock lang:python %}
(az.hdi(trace)["lane2"]
 .sel({"hdi": ["lower", "higher"]})
 .to_dataframe()
 .reset_index()
 .assign(_='lane2')
 .pivot(index='_', columns='hdi', values='lane2')
 .reset_index()
 .assign(median=trace.posterior["lane2"].median(dim=("chain", "draw")).to_numpy())
 .assign(width=lambda x: x['higher'] - x['lower'])
 [['lower', 'median', 'higher', 'width']]
 )
{% endcodeblock %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>hdi</th>
<th>lower</th>
<th>median</th>
<th>higher</th>
<th>width</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>-0.292225</td>
<td>0.546044</td>
<td>1.481515</td>
<td>1.77374</td>
</tr>
</tbody>
</table>
</div>

Let's look at the portion of the intercept's posterior that is greater than 0. The model is 90% sure that lane 2 is faster.

{% codeblock lang:python %}
(trace
 .posterior["lane2"]
 .to_dataframe()
 .reset_index()
 .assign(greater_0=lambda x: x['lane2'] > 0)['greater_0'].mean()
 )
{% endcodeblock %}

    0.899875

Next, onto the measurement of the cars. Candy Cane is my 3yr olds favorite so lots of trials there. It also pretty consistently loses. The model picks up on this. Not surprised to see Orange near the top. Big Chain also seems quite fast. I was not aware of how well Bluey was performing.

I'm not sure how to measure this (maybe it's the widthe of hdi interval) but I'm curious which cars I'd learn the most about given 1 additional race.

{% codeblock lang:python %}
(az.hdi(trace)["mu_racer"]
 .sel({"hdi": ["lower", "higher"]})
 .to_dataframe()
 .reset_index()
 .pivot(index='racer', columns='hdi', values='mu_racer')
 .reset_index()
 .merge(trace.posterior["mu_racer"].median(dim=("chain", "draw")).to_dataframe('median'),
        on='racer',
        how='left')
 .sort_values(by='median')
 .assign(width=lambda x: x['higher'] - x['lower'])
 [['racer', 'lower', 'median', 'higher', 'width']]
 )
{% endcodeblock %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>&#xa0;</th>
<th>racer</th>
<th>lower</th>
<th>median</th>
<th>higher</th>
<th>width</th>
</tr>
</thead>
<tbody>
<tr>
<td>3</td>
<td>CandyCane</td>
<td>-1.630189</td>
<td>-0.477509</td>
<td>0.385684</td>
<td>2.015874</td>
</tr>

<tr>
<td>11</td>
<td>Yellow</td>
<td>-1.732217</td>
<td>-0.281237</td>
<td>0.735534</td>
<td>2.467751</td>
</tr>

<tr>
<td>9</td>
<td>Spikey</td>
<td>-1.861409</td>
<td>-0.230835</td>
<td>0.939880</td>
<td>2.801289</td>
</tr>

<tr>
<td>10</td>
<td>Whitey</td>
<td>-1.798544</td>
<td>-0.181691</td>
<td>0.958850</td>
<td>2.757394</td>
</tr>

<tr>
<td>5</td>
<td>Nick</td>
<td>-1.182705</td>
<td>-0.082876</td>
<td>0.941209</td>
<td>2.123914</td>
</tr>

<tr>
<td>8</td>
<td>Spidey</td>
<td>-1.248341</td>
<td>-0.037978</td>
<td>1.116357</td>
<td>2.364698</td>
</tr>

<tr>
<td>7</td>
<td>Reddy</td>
<td>-1.215164</td>
<td>0.021394</td>
<td>1.376155</td>
<td>2.591319</td>
</tr>

<tr>
<td>4</td>
<td>Mom</td>
<td>-1.041651</td>
<td>0.127130</td>
<td>1.415593</td>
<td>2.457243</td>
</tr>

<tr>
<td>1</td>
<td>BlackBlur</td>
<td>-0.747250</td>
<td>0.189628</td>
<td>1.351777</td>
<td>2.099027</td>
</tr>

<tr>
<td>0</td>
<td>BigChain</td>
<td>-0.742637</td>
<td>0.298039</td>
<td>1.720762</td>
<td>2.463398</td>
</tr>

<tr>
<td>2</td>
<td>Bluey</td>
<td>-0.606904</td>
<td>0.338542</td>
<td>1.570626</td>
<td>2.177530</td>
</tr>

<tr>
<td>6</td>
<td>Orange</td>
<td>-0.519564</td>
<td>0.375396</td>
<td>1.543926</td>
<td>2.063490</td>
</tr>
</tbody>
</table>
</div>

Next I wanted to get a feel for how the cars compared in terms of how often I expect one car to win against another. I compared this by looking at the share of a car's posterior (racer_x) that was greater than the median of each other car's posterior (racer_y).

Only 2% of Candy Cane's posterior is greater than Orange's median!

{% codeblock lang:python %}
results = trace.posterior["mu_racer"].to_dataframe().reset_index()
means = results.groupby('racer')['mu_racer'].median().reset_index()
comparisons = (results
               .assign(key=0)
               .merge(means.assign(key=0), on='key', how='inner')
               .drop(columns='key')
               .assign(racerx_win_prob=lambda x: x['mu_racer_x'] > x['mu_racer_y'])
               .groupby(['racer_x', 'racer_y'])['racerx_win_prob']
               .mean()
               .reset_index()
               )
(comparisons
 .pivot(index='racer_x', columns='racer_y', values='racerx_win_prob')
 )
{% endcodeblock %}

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1">
<thead>
<tr>
<th>racer<sub>y</sub></th>
<th>BigChain</th>
<th>BlackBlur</th>
<th>Bluey</th>
<th>CandyCane</th>
<th>Mom</th>
<th>Nick</th>
<th>Orange</th>
<th>Reddy</th>
<th>Spidey</th>
<th>Spikey</th>
<th>Whitey</th>
<th>Yellow</th>
</tr>
</thead>
<tbody>
<tr>
<td>racer<sub>x</sub></td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
<td>&#xa0;</td>
</tr>

<tr>
<td>BigChain</td>
<td>0.499875</td>
<td>0.581875</td>
<td>0.472875</td>
<td>0.952750</td>
<td>0.631125</td>
<td>0.803625</td>
<td>0.446500</td>
<td>0.714750</td>
<td>0.768125</td>
<td>0.888375</td>
<td>0.866375</td>
<td>0.908000</td>
</tr>

<tr>
<td>BlackBlur</td>
<td>0.415125</td>
<td>0.500000</td>
<td>0.388125</td>
<td>0.936250</td>
<td>0.555625</td>
<td>0.749125</td>
<td>0.359500</td>
<td>0.660375</td>
<td>0.711875</td>
<td>0.847125</td>
<td>0.819625</td>
<td>0.872375</td>
</tr>

<tr>
<td>Bluey</td>
<td>0.529875</td>
<td>0.628000</td>
<td>0.500000</td>
<td>0.970000</td>
<td>0.674250</td>
<td>0.840500</td>
<td>0.470500</td>
<td>0.763625</td>
<td>0.808125</td>
<td>0.913375</td>
<td>0.892625</td>
<td>0.930125</td>
</tr>

<tr>
<td>CandyCane</td>
<td>0.028625</td>
<td>0.052250</td>
<td>0.023000</td>
<td>0.500000</td>
<td>0.070875</td>
<td>0.193000</td>
<td>0.019250</td>
<td>0.122000</td>
<td>0.157750</td>
<td>0.313250</td>
<td>0.275625</td>
<td>0.355000</td>
</tr>

<tr>
<td>Mom</td>
<td>0.355750</td>
<td>0.445250</td>
<td>0.329125</td>
<td>0.883625</td>
<td>0.500000</td>
<td>0.669250</td>
<td>0.303750</td>
<td>0.585000</td>
<td>0.630875</td>
<td>0.779250</td>
<td>0.749625</td>
<td>0.807500</td>
</tr>

<tr>
<td>Nick</td>
<td>0.184750</td>
<td>0.257625</td>
<td>0.165375</td>
<td>0.790625</td>
<td>0.308625</td>
<td>0.500000</td>
<td>0.148625</td>
<td>0.397375</td>
<td>0.458500</td>
<td>0.633000</td>
<td>0.585500</td>
<td>0.670375</td>
</tr>

<tr>
<td>Orange</td>
<td>0.559625</td>
<td>0.652250</td>
<td>0.529375</td>
<td>0.979875</td>
<td>0.708500</td>
<td>0.856250</td>
<td>0.500000</td>
<td>0.786000</td>
<td>0.824625</td>
<td>0.927875</td>
<td>0.906625</td>
<td>0.943250</td>
</tr>

<tr>
<td>Reddy</td>
<td>0.289250</td>
<td>0.361250</td>
<td>0.265875</td>
<td>0.832625</td>
<td>0.406000</td>
<td>0.588250</td>
<td>0.247750</td>
<td>0.500000</td>
<td>0.549625</td>
<td>0.704125</td>
<td>0.665750</td>
<td>0.735625</td>
</tr>

<tr>
<td>Spidey</td>
<td>0.228875</td>
<td>0.297000</td>
<td>0.205625</td>
<td>0.806250</td>
<td>0.347250</td>
<td>0.543000</td>
<td>0.189500</td>
<td>0.442125</td>
<td>0.500000</td>
<td>0.664750</td>
<td>0.624750</td>
<td>0.696875</td>
</tr>

<tr>
<td>Spikey</td>
<td>0.131750</td>
<td>0.190250</td>
<td>0.116875</td>
<td>0.653750</td>
<td>0.226000</td>
<td>0.380875</td>
<td>0.104375</td>
<td>0.300375</td>
<td>0.344625</td>
<td>0.500000</td>
<td>0.465625</td>
<td>0.539500</td>
</tr>

<tr>
<td>Whitey</td>
<td>0.161500</td>
<td>0.208625</td>
<td>0.144375</td>
<td>0.693625</td>
<td>0.246875</td>
<td>0.418750</td>
<td>0.133125</td>
<td>0.332250</td>
<td>0.379125</td>
<td>0.538500</td>
<td>0.500000</td>
<td>0.572250</td>
</tr>

<tr>
<td>Yellow</td>
<td>0.095625</td>
<td>0.142375</td>
<td>0.083375</td>
<td>0.624250</td>
<td>0.176000</td>
<td>0.340250</td>
<td>0.073125</td>
<td>0.246250</td>
<td>0.300000</td>
<td>0.462000</td>
<td>0.426375</td>
<td>0.500000</td>
</tr>
</tbody>
</table>
</div>

I visually depict each car's posterior in the next two plots. Not much sticks out to be beyond... I have some more playing to do!

{% codeblock lang:python %}
_, ax = plt.subplots()

trace_hdi = az.hdi(trace)
ax.scatter(racers.values(), trace.posterior["mu_racer"].median(dim=("chain", "draw")), color="C0", alpha=1, s=100)
ax.vlines(
    racers.values(),
    trace_hdi["mu_racer"].sel({"hdi": "lower"}),
    trace_hdi["mu_racer"].sel({"hdi": "higher"}),
    alpha=0.6,
    lw=5,
    color="C0",
)
ax.set_xticks(list(racers.values()), racers.keys())
ax.set_xlabel("Car")
ax.set_ylabel("Posterior Car Speed")
ax.set_title("HDI of Car Speed");
{% endcodeblock %}

<img src="{{ root_url }}/images/misc/hot_wheels_output1.png" />

{% codeblock lang:python %}
class TeamLabeller(az.labels.BaseLabeller):
    def make_label_flat(self, var_name, sel, isel):
        sel_str = self.sel_to_str(sel, isel)
        return sel_str
ax = az.plot_forest(trace, var_names=["mu_racer"], labeller=TeamLabeller())
ax[0].set_title("Car Speed");
{% endcodeblock %}

<img src="{{ root_url }}/images/misc/hot_wheels_output2.png" />
