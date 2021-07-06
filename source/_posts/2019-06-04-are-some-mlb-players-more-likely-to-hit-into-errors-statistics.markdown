---
layout: post
title: "Are Some MLB Players More Likely To Hit Into Errors: Statistics"
date: 2019-06-04 20:04:31 -0500
comments: true
categories: [open source, data science, statistics, python]
---

In a [previous post](https://danvatterott.com/blog/2019/04/19/are-some-mlb-players-more-likely-to-hit-into-errors-munging/), I described how to download and clean data for understanding how likely a baseball player is to hit into an error given that they hit the ball into play.

This analysis will statistically demonstrate that some players are more likely to hit into errors than others.

Errors are uncommon, so players hit into errors very infrequently. Estimating the likelihood of an infrequent event is hard and requires lots of data. To acquire as much data as possible, I wrote a bash script that will download data for all players between 1970 and 2018.

This data enables me to use data from multiple years for each player, giving me more data when estimating how likely a particular player is to hit into an error.


{% codeblock lang:bash %}
%%bash

for i in {1970..2018}; do
    echo "YEAR: $i"
    ../scripts/get_data.sh ${i};
done

find processed_data/* -type f -name 'errors_bip.out' | \
    xargs awk '{print $0", "FILENAME}' | \
    sed s1processed_data/11g1 | \
    sed s1/errors_bip.out11g1 > \
        processed_data/all_errors_bip.out
{% endcodeblock %}

The data has 5 columns: playerid, playername, errors hit into, balls hit into play (BIP), and year. The file does not have a header.


{% codeblock lang:bash %}
%%bash
head ../processed_data/all_errors_bip.out
{% endcodeblock %}

    aaroh101, Hank Aaron, 8, 453, 1970
    aarot101, Tommie Aaron, 0, 53, 1970
    abert101, Ted Abernathy, 0, 10, 1970
    adaij101, Jerry Adair, 0, 24, 1970
    ageet101, Tommie Agee, 12, 480, 1970
    akerj102, Jack Aker, 0, 10, 1970
    alcal101, Luis Alcaraz, 1, 107, 1970
    alleb105, Bernie Allen, 1, 240, 1970
    alled101, Dick Allen, 4, 341, 1970
    alleg101, Gene Alley, 6, 356, 1970


I can load the data into pandas using the following command.


{% codeblock lang:python %}
import pandas as pd

DF = pd.read_csv('../processed_data/all_errors_bip.out',
                 header=None,
                 names=['playerid', 'player_name', 'errors', 'bip', 'year'])
{% endcodeblock %}


{% codeblock lang:python %}
DF.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerid</th>
      <th>player_name</th>
      <th>errors</th>
      <th>bip</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aaroh101</td>
      <td>Hank Aaron</td>
      <td>8</td>
      <td>453</td>
      <td>1970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aarot101</td>
      <td>Tommie Aaron</td>
      <td>0</td>
      <td>53</td>
      <td>1970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abert101</td>
      <td>Ted Abernathy</td>
      <td>0</td>
      <td>10</td>
      <td>1970</td>
    </tr>
    <tr>
      <th>3</th>
      <td>adaij101</td>
      <td>Jerry Adair</td>
      <td>0</td>
      <td>24</td>
      <td>1970</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ageet101</td>
      <td>Tommie Agee</td>
      <td>12</td>
      <td>480</td>
      <td>1970</td>
    </tr>
  </tbody>
</table>
</div>




{% codeblock lang:python %}
len(DF)
{% endcodeblock %}




    38870



I have almost 39,000 year, player combinations.... a good amount of data to play with.

While exploring the data, I noticed that players hit into errors less frequently now than they used to. Let's see how the probability that a player hits into an error has changed across the years.


{% codeblock lang:python %}
%matplotlib inline

YEAR_DF = (DF
           .groupby("year")
           .agg({
               "errors": "sum",
               "bip": "sum"
           })
           .assign(prop_error=lambda x: x["errors"] / x["bip"])
          )

YEAR_DF["prop_error"].plot(style="o-");
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/error_year.png" />

Interestingly, the proportion of errors per BIP [has been dropping over time](https://www.pinstripealley.com/2013/8/16/4623050/mlb-errors-trends-statistics). I am not sure if this is a conscious effort by MLB score keepers, a change in how hitters hit, or improved fielding (but I suspect it's the score keepers). It looks like this drop in errors per BIP leveled off around 2015. Zooming in.


{% codeblock lang:python %}
YEAR_DF[YEAR_DF.index > 2010]["prop_error"].plot(style="o-");
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/zoom_error_year.png" />

I explore this statistically in [a jupyter notebook on my github](https://github.com/dvatterott/mlb_errors/blob/master/notebook/PYMC%20-%20Hierarchical%20Beta%20Binomial%20YEAR.ipynb).

Because I don't want year to confound the analysis, I remove all data before 2015.


{% codeblock lang:python %}
DF = DF[DF["year"] >= 2015]
{% endcodeblock %}


{% codeblock lang:python %}
len(DF)
{% endcodeblock %}




    3591



This leaves me with 3500 year, player combinations.

Next I combine players' data across years.


{% codeblock lang:python %}
GROUPED_DF = DF.groupby(["playerid", "player_name"]).agg({"errors": "sum", "bip": "sum"}).reset_index()
{% endcodeblock %}


{% codeblock lang:python %}
GROUPED_DF.describe()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>errors</th>
      <th>bip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1552.000000</td>
      <td>1552.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.835052</td>
      <td>324.950387</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.073256</td>
      <td>494.688755</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>69.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>437.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.000000</td>
      <td>2102.000000</td>
    </tr>
  </tbody>
</table>
</div>



I want an idea for how likely players are to hit into errors.


{% codeblock lang:python %}
TOTALS = GROUPED_DF.agg({"errors": "sum", "bip": "sum"})
ERROR_RATE = TOTALS["errors"] / TOTALS["bip"]
ERROR_RATE
{% endcodeblock %}




    0.011801960251664112



Again, errors are very rare, so I want know how many "trials" (BIP) I need for a reasonable estimate of how likely each player is to hit into an error.

I'd like the majority of players to have at least 5 errors. I can estimate how many BIP that would require.


{% codeblock lang:python %}
5. /ERROR_RATE
{% endcodeblock %}




    423.65843413978496



Looks like I should require at least 425 BIP for each player. I round this to 500.


{% codeblock lang:python %}
GROUPED_DF = GROUPED_DF[GROUPED_DF["bip"] > 500]
{% endcodeblock %}


{% codeblock lang:python %}
GROUPED_DF = GROUPED_DF.reset_index(drop=True)
{% endcodeblock %}


{% codeblock lang:python %}
GROUPED_DF.head()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerid</th>
      <th>player_name</th>
      <th>errors</th>
      <th>bip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abrej003</td>
      <td>Jose Abreu</td>
      <td>20</td>
      <td>1864</td>
    </tr>
    <tr>
      <th>1</th>
      <td>adamm002</td>
      <td>Matt Adams</td>
      <td>6</td>
      <td>834</td>
    </tr>
    <tr>
      <th>2</th>
      <td>adrie001</td>
      <td>Ehire Adrianza</td>
      <td>2</td>
      <td>533</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aguij001</td>
      <td>Jesus Aguilar</td>
      <td>2</td>
      <td>551</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ahmen001</td>
      <td>Nick Ahmed</td>
      <td>12</td>
      <td>1101</td>
    </tr>
  </tbody>
</table>
</div>




{% codeblock lang:python %}
GROUPED_DF.describe()
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>errors</th>
      <th>bip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>354.000000</td>
      <td>354.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12.991525</td>
      <td>1129.059322</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.447648</td>
      <td>428.485467</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>503.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.000000</td>
      <td>747.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>1112.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17.000000</td>
      <td>1475.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.000000</td>
      <td>2102.000000</td>
    </tr>
  </tbody>
</table>
</div>



I've identified 354 players who have enough BIP for me to estimate how frequently they hit into errors. 

Below, I plot how the likelihood of hitting into errors is distributed.


{% codeblock lang:python %}
%matplotlib inline

GROUPED_DF["prop_error"] = GROUPED_DF["errors"] / GROUPED_DF["bip"]
GROUPED_DF["prop_error"].hist();
{% endcodeblock %}


<img src="{{ root_url }}/images/mlb/error_dist.png" />

The question is whether someone who has hit into errors in 2% of their BIP is more likely to hit into an error than someone who has hit into errors in 0.5% of their BIP (or is this all just random variation).

To try and estimate this, I treat each BIP as a Bernoulli trial. Hitting into an error is a "success". I use a Binomial distribution to model the number of "successes". I would like to know if different players are more or less likely to hit into errors. To do this, I model each player as having their own Binomial distribution and ask whether *p* (the probability of success) differs across players. 

To answer this question, I could use a [chi square contingency test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html#scipy.stats.chi2_contingency) but this would only tell me whether players differ at all and not which players differ.

The traditional way to identify which players differ is to do pairwise comparisons, but this would result in TONS of comparisons making [false positives all but certain](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).

Another option is to harness Bayesian statistics and build a [Hierarchical Beta-Binomial model](http://sl8r000.github.io/ab_testing_statistics/use_a_hierarchical_model/). The intuition is that each player's probability of hitting into an error is drawn from a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution). I want to know whether these Beta distributions are different. I then assume I can best estimate a player's Beta distribution by using that particular player's data AND data from all players together.

The model is built so that as I accrue data about a particular player, I will trust that data more and more, relying less and less on data from all players. This is called partial pooling. [Here's](https://dsaber.com/2016/08/27/analyze-your-experiment-with-a-multilevel-logistic-regression-using-pymc3%E2%80%8B/) a useful explanation.

I largely based my analysis on [this](https://docs.pymc.io/notebooks/hierarchical_partial_pooling.html) tutorial. Reference the tutorial for an explanation of how I choose my priors. I ended up using a greater lambda value (because the model sampled better) in the Exponential prior, and while this did lead to more extreme estimates of error likelihood, it didn't change the basic story.


{% codeblock lang:python %}
import pymc3 as pm
import numpy as np
import theano.tensor as tt

with pm.Model() as model:
 
    phi = pm.Uniform('phi', lower=0.0, upper=1.0)
    
    kappa_log = pm.Exponential('kappa_log', lam=25.)
    kappa = pm.Deterministic('kappa', tt.exp(kappa_log))

    rates = pm.Beta('rates', alpha=phi*kappa, beta=(1.0-phi)*kappa, shape=len(GROUPED_DF))

    trials = np.array(GROUPED_DF["bip"])
    successes = np.array(GROUPED_DF["errors"])
 
    obs = pm.Binomial('observed_values', trials, rates, observed=successes)
    trace = pm.sample(2000, tune=1000, chains=2, cores=2, nuts_kwargs={'target_accept': .95})
{% endcodeblock %}

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [rates, kappa_log, phi]
    Sampling 2 chains: 100%|██████████| 6000/6000 [01:47<00:00, 28.06draws/s] 


Check whether the model converged.


{% codeblock lang:python %}
max(np.max(score) for score in pm.gelman_rubin(trace).values())
{% endcodeblock %}




    1.0022635936332533




{% codeblock lang:python %}
bfmi = pm.bfmi(trace)
max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(trace).values())
(pm.energyplot(trace, figsize=(6, 4)).set_title("BFMI = {}\nGelman-Rubin = {}".format(bfmi, max_gr)));
{% endcodeblock %}


<img src="{{ root_url }}/images/mlb/energy.png" />

The most challenging parameter to fit is *kappa* which modulates for the variance in the likelihood to hit into an error. I take a look at it to make sure things look as expected.


{% codeblock lang:python %}
pm.summary(trace, varnames=["kappa"])
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>kappa</th>
      <td>927.587178</td>
      <td>141.027597</td>
      <td>4.373954</td>
      <td>657.066554</td>
      <td>1201.922608</td>
      <td>980.288914</td>
      <td>1.000013</td>
    </tr>
  </tbody>
</table>
</div>




{% codeblock lang:python %}
pm.traceplot(trace, varnames=['kappa']);
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/kappa.png" />

I can also look at *phi*, the estimated global likelihood to hit into an error.


{% codeblock lang:python %}
pm.traceplot(trace, varnames=['phi']);
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/phi.png" />

Finally, I can look at how all players vary in their likelihood to hit into an error.


{% codeblock lang:python %}
pm.traceplot(trace, varnames=['rates']);
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/rate_trace.png" />

Obviously, the above plot is a lot to look at it, so let's order players by how likely the model believes they are to hit in an error.


{% codeblock lang:python %}
from matplotlib import pyplot as plt

rate_means = trace['rates', 1000:].mean(axis=0)
rate_se = trace['rates', 1000:].std(axis=0)

mean_se = [(x, y, i) for i, x, y in zip(GROUPED_DF.index, rate_means, rate_se)]
sorted_means_se = sorted(mean_se, key=lambda x: x[0])
sorted_means = [x[0] for x in sorted_means_se]
sorted_se = [x[1] for x in sorted_means_se]

x = np.arange(len(sorted_means))

plt.plot(x, sorted_means, 'o', alpha=0.25);

for x_val, m, se in zip(x, sorted_means, sorted_se):
    plt.plot([x_val, x_val], [m-se, m+se], 'b-', alpha=0.5)
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/players_ranked.png" />

Now, the ten players who are most likely to hit into an error.


{% codeblock lang:python %}
estimated_mean = pm.summary(trace, varnames=["rates"]).iloc[[x[2] for x in sorted_means_se[-10:]]]["mean"]

GROUPED_DF.loc[[x[2] for x in sorted_means_se[-10:]], :].assign(estimated_mean=estimated_mean.values).iloc[::-1]
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerid</th>
      <th>player_name</th>
      <th>errors</th>
      <th>bip</th>
      <th>prop_error</th>
      <th>estimated_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71</th>
      <td>corrc001</td>
      <td>Carlos Correa</td>
      <td>30</td>
      <td>1368</td>
      <td>0.021930</td>
      <td>0.017838</td>
    </tr>
    <tr>
      <th>227</th>
      <td>myerw001</td>
      <td>Wil Myers</td>
      <td>27</td>
      <td>1214</td>
      <td>0.022241</td>
      <td>0.017724</td>
    </tr>
    <tr>
      <th>15</th>
      <td>andre001</td>
      <td>Elvis Andrus</td>
      <td>37</td>
      <td>1825</td>
      <td>0.020274</td>
      <td>0.017420</td>
    </tr>
    <tr>
      <th>258</th>
      <td>plawk001</td>
      <td>Kevin Plawecki</td>
      <td>14</td>
      <td>528</td>
      <td>0.026515</td>
      <td>0.017200</td>
    </tr>
    <tr>
      <th>285</th>
      <td>rojam002</td>
      <td>Miguel Rojas</td>
      <td>21</td>
      <td>952</td>
      <td>0.022059</td>
      <td>0.017001</td>
    </tr>
    <tr>
      <th>118</th>
      <td>garca003</td>
      <td>Avisail Garcia</td>
      <td>28</td>
      <td>1371</td>
      <td>0.020423</td>
      <td>0.016920</td>
    </tr>
    <tr>
      <th>244</th>
      <td>pench001</td>
      <td>Hunter Pence</td>
      <td>22</td>
      <td>1026</td>
      <td>0.021442</td>
      <td>0.016875</td>
    </tr>
    <tr>
      <th>20</th>
      <td>baezj001</td>
      <td>Javier Baez</td>
      <td>23</td>
      <td>1129</td>
      <td>0.020372</td>
      <td>0.016443</td>
    </tr>
    <tr>
      <th>335</th>
      <td>turnt001</td>
      <td>Trea Turner</td>
      <td>23</td>
      <td>1140</td>
      <td>0.020175</td>
      <td>0.016372</td>
    </tr>
    <tr>
      <th>50</th>
      <td>cainl001</td>
      <td>Lorenzo Cain</td>
      <td>32</td>
      <td>1695</td>
      <td>0.018879</td>
      <td>0.016332</td>
    </tr>
  </tbody>
</table>
</div>



And the 10 players who are least likely to hit in an error. 


{% codeblock lang:python %}
estimated_mean = pm.summary(trace, varnames=["rates"]).iloc[[x[2] for x in sorted_means_se[:10]]]["mean"]

GROUPED_DF.loc[[x[2] for x in sorted_means_se[:10]], :].assign(estimated_mean=estimated_mean.values)
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerid</th>
      <th>player_name</th>
      <th>errors</th>
      <th>bip</th>
      <th>prop_error</th>
      <th>estimated_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>226</th>
      <td>murpd006</td>
      <td>Daniel Murphy</td>
      <td>4</td>
      <td>1680</td>
      <td>0.002381</td>
      <td>0.005670</td>
    </tr>
    <tr>
      <th>223</th>
      <td>morrl001</td>
      <td>Logan Morrison</td>
      <td>4</td>
      <td>1241</td>
      <td>0.003223</td>
      <td>0.006832</td>
    </tr>
    <tr>
      <th>343</th>
      <td>vottj001</td>
      <td>Joey Votto</td>
      <td>8</td>
      <td>1724</td>
      <td>0.004640</td>
      <td>0.007112</td>
    </tr>
    <tr>
      <th>239</th>
      <td>panij002</td>
      <td>Joe Panik</td>
      <td>7</td>
      <td>1542</td>
      <td>0.004540</td>
      <td>0.007245</td>
    </tr>
    <tr>
      <th>51</th>
      <td>calhk001</td>
      <td>Kole Calhoun</td>
      <td>9</td>
      <td>1735</td>
      <td>0.005187</td>
      <td>0.007413</td>
    </tr>
    <tr>
      <th>55</th>
      <td>carpm002</td>
      <td>Matt Carpenter</td>
      <td>8</td>
      <td>1566</td>
      <td>0.005109</td>
      <td>0.007534</td>
    </tr>
    <tr>
      <th>142</th>
      <td>hamib001</td>
      <td>Billy Hamilton</td>
      <td>8</td>
      <td>1476</td>
      <td>0.005420</td>
      <td>0.007822</td>
    </tr>
    <tr>
      <th>289</th>
      <td>rosae001</td>
      <td>Eddie Rosario</td>
      <td>8</td>
      <td>1470</td>
      <td>0.005442</td>
      <td>0.007855</td>
    </tr>
    <tr>
      <th>275</th>
      <td>renda001</td>
      <td>Anthony Rendon</td>
      <td>9</td>
      <td>1564</td>
      <td>0.005754</td>
      <td>0.007966</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alony001</td>
      <td>Yonder Alonso</td>
      <td>8</td>
      <td>1440</td>
      <td>0.005556</td>
      <td>0.008011</td>
    </tr>
  </tbody>
</table>
</div>



It looks to me like players who hit more ground balls are more likely to hit into an error than players who predominately hits fly balls and line-drives. This makes sense since infielders make more errors than outfielders.

Using the posterior distribution of estimated likelihoods to hit into an error, I can assign a probability to whether Carlos Correa is more likely to hit into an error than Daniel Murphy.


{% codeblock lang:python %}
np.mean(trace['rates', 1000:][:, 71] <= trace['rates', 1000:][:, 226])
{% endcodeblock %}




    0.0



The model believes Correa is much more likely to hit into an error than Murphy!

I can also plot these players' posterior distributions.


{% codeblock lang:python %}
import seaborn as sns

sns.kdeplot(trace['rates', 1000:][:, 226], shade=True, label="Daniel Murphy");
sns.kdeplot(trace['rates', 1000:][:, 71], shade=True, label="Carlos Correa");
sns.kdeplot(trace['rates', 1000:].flatten(), shade=True, label="Overall");
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/play_comparison.png" />

Finally, I can look exclusively at how the posterior distributions of the ten most likely and 10 least likely players to hit into an error compare.


{% codeblock lang:python %}
sns.kdeplot(trace['rates', 1000:][:, [x[2] for x in sorted_means_se[-10:]]].flatten(), shade=True, label="10 Least Likely");
sns.kdeplot(trace['rates', 1000:][:, [x[2] for x in sorted_means_se[:10]]].flatten(), shade=True, label="10 Most Likely");
{% endcodeblock %}

<img src="{{ root_url }}/images/mlb/top10.png" />

All in all, this analysis makes it obvious that some players are more likely to hit into errors than other players. This is probably driven by how often players hit ground balls.