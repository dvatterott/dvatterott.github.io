---
layout: post
title: "PCA Tutorial"
date: 2016-11-06 13:33:50 -0500
comments: true
categories: [python, open source, tutorial, pca]
---

[Principal Component Analysis](http://setosa.io/ev/principal-component-analysis/) (PCA) is an important method for dimensionality reduction and data cleaning. I have used PCA in the past on this blog for estimating the latent variables that underlie player statistics. For example, I might have two features: average number of offensive rebounds and average number of defensive rebounds. The two features are highly correlated because a latent variable, the player's *rebounding ability*, explains common variance in the two features. PCA is a method for extracting these latent variables that explain common variance across features.

In this tutorial I generate fake data in order to help gain insight into the mechanics underlying PCA.

Below I create my first feature by sampling from a normal distribution. I create a second feature by adding a noisy normal distribution to the first feature multiplied by two. Because I generated the data here, I know it's composed to two latent variables, and PCA should be able to identify these latent variables.

I generate the data and plot it below.


{% codeblock lang:python %}
import numpy as np, matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

np.random.seed(1) #make sure we're all working with the same numbers

X = np.random.normal(0.0,2.0,[100,1])
X = [X,X*2+np.random.normal(0.0,8.0,[100,1])]
X = np.squeeze(X)

plt.plot(X[0],X[1],'o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Raw Data')
plt.axis([-6,6,-30,30]);
{% endcodeblock %}


<img src="{{ root_url }}/images/PCA/original_data.png" />


The first step before doing PCA is to normalize the data. This centers each feature (each feature will have a mean of 0) and divides data by its standard deviation (changing the standard deviation to 1). Normalizing the data puts all features on the same scale. Having features on the same scale is important because features might be more or less variable because of measurement rather than the latent variables producing the feature. For example, in basketball, points are often accumulated in sets of 2s and 3s, while rebounds are accumulated one at a time. The nature of basketball puts points and rebounds on a different scales, but this doesn't mean that the latent variables *scoring ability* and *rebounding ability* are more or less variable.

Below I normalize and plot the data.


{% codeblock lang:python %}
import scipy.stats as stats

X = stats.mstats.zscore(X,axis=1)

plt.plot(X[0],X[1],'o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Standardized Data')
plt.axis([-4,4,-4,4]);
{% endcodeblock %}


<img src="{{ root_url }}/images/PCA/stand_data.png" />


After standardizing the data, I need to find the [eigenvectors and eigenvalues](http://setosa.io/ev/eigenvectors-and-eigenvalues/). The eigenvectors point in the direction of a component and eigenvalues represent the amount of variance explained by the component. Below, I plot the standardized data with the eigenvectors ploted with their eigenvalues as the vectors distance from the origin.

As you can see, the blue eigenvector is longer and points in the direction with the most variability. The purple eigenvector is shorter and points in the direction with less variability.

As expected, one component explains far more variability than the other component (becaus both my features share variance from a single latent gaussian distribution).


{% codeblock lang:python %}
C = np.dot(X,np.transpose(X))/(np.shape(X)[1]-1);
[V,PC] = np.linalg.eig(C)

plt.plot(X[0],X[1],'o')
plt.plot([0,PC[0,0]*V[0]],[0,PC[1,0]*V[0]],'o-')
plt.plot([0,PC[0,1]*V[1]],[0,PC[1,1]*V[1]],'o-')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Standardized Data with Eigenvectors')
plt.axis([-4,4,-4,4]);
{% endcodeblock %}


<img src="{{ root_url }}/images/PCA/eigen_data.png" />


Next I order the eigenvectors according to the magnitude of their eigenvalues. This orders the components so that the components that explain more variability occur first. I then transform the data so that they're axis aligned. This means the first component explain variability on the x-axis and the second component explains variance on the y-axis.


{% codeblock lang:python %}
indices = np.argsort(-1*V)
V = V[indices]
PC = PC[indices,:]

X_rotated = np.dot(X.T,PC)

plt.plot(X_rotated.T[0],X_rotated.T[1],'o')
plt.plot([0,PC[1,0]*V[0]],[0,0],'o-')
plt.plot([0,0],[0,PC[1,1]*V[1]],'o-')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Projected into PC space')
plt.axis([-4,4,-4,4]);
{% endcodeblock %}


<img src="{{ root_url }}/images/PCA/trans_data.png" />


Finally, just to make sure the PCA was done correctly, I will call PCA from the sklearn library, run it, and make sure it produces the same results as my analysis.


{% codeblock lang:python %}
from sklearn.decomposition import PCA

pca = PCA() #create PCA object
test = pca.fit_transform(X.T) #pull out principle components

print(stats.stats.pearsonr(X_rotated.T[0],test.T[0]))
print(stats.stats.pearsonr(X_rotated.T[1],test.T[1]))
{% endcodeblock %}

    (-1.0, 0.0)
    (-1.0, 0.0)
