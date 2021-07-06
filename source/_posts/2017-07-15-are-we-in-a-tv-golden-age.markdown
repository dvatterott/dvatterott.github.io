---
layout: post
title: "Are we in a TV golden age?"
date: 2017-07-15 16:12:26 -0400
comments: true
categories: [tv, movies, data visualization, plotting, open source]
---

I recently found myself in a argument with my wife regarding whether TV was better now than previously. I believed that TV was better now than 20 years ago. My wife contended that there was simply more TV content being produced, and that this led to more good shows, but shows are not inherently any better.

This struck me as a great opportunity to do some quick data science. For this post, I scraped the names (from wikipedia) and ratings (from [TMDb](https://www.themoviedb.org/)) of all American TV shows. I did the same for major American movies, so that I could have a comparison group (maybe all content is better or worse). The ratings are given by TMDb's users and are scores between 1 and 10 (where 10 is a great show/movie and 1 is a lousy show/movie).

All the code for this post can be found [on my github](https://github.com/dvatterott/tv_vs_movies).

I decided to operationalize my "golden age of TV" hypothesis as the average TV show is better now than previously. This would be expressed as a positive slope (beta coefficient) when building a linear regression that outputs the rating of a show given the date on which the show first aired. My wife predicted a slope near zero or negative (shows are no better or worse than previously).

Below, I plot the ratings of TV shows and movies across time. Each show is a dot in the scatter plot. Show rating (average rating given my TMBb) is on the y-axis. The date of the show's first airing is on the x-axis. When I encountered shows with the same name, I just tacked a number onto the end. For instance, show "x" would become show "x_1." The size of each point in the scatter plot is the show's "popularity", which is a bit of a black box, but it's given by TMBb's API. TMDb does not give a full description of how they calculate popularity, but they do say its a function of how many times an item is viewed on TMDb, how many times an item is rated, and how many times the item has been added to watch or favorite list. I decided to depict it here just to give the figures a little more detail. The larger the dot, the more popular the show.

Here's a plot of all TV shows across time.

<iframe src="{{ root_url }}/images/tv_movies/index_tv.html" marginwidth="0" marginheight="0" scrolling="no" width="800" height="500"></iframe>

To test the "golden age of TV" hypothesis, I coded up a linear regression in javascript (below). I put the regression's output as a comment at the end of the code.
Before stating whether the hypothesis was rejected or not, I should note that that I removed shows with less than 10 votes because these shows had erratic ratings.

As you can see, there is no evidence that TV is better now that previously. In fact, if anything, this dataset says that TV is worse (but more on this later).

{% codeblock lang:javascript %}
function linearRegression(y,x){

    var lr = {};
    var n = y.length;
    var sum_x = 0;
    var sum_y = 0;
    var sum_xy = 0;
    var sum_xx = 0;
    var sum_yy = 0;

    for (var i = 0; i < y.length; i++) {

        sum_x += x[i];
        sum_y += y[i];
        sum_xy += (x[i]*y[i]);
        sum_xx += (x[i]*x[i]);
        sum_yy += (y[i]*y[i]);
    }

    lr['slope'] = (n * sum_xy - sum_x * sum_y) / (n*sum_xx - sum_x * sum_x);
    lr['intercept'] = (sum_y - lr.slope * sum_x)/n;
    lr['r2'] = Math.pow((n*sum_xy - sum_x*sum_y)/Math.sqrt((n*sum_xx-sum_x*sum_x)*(n*sum_yy-sum_y*sum_y)),2);

    return lr;

};

var yval = data
    .filter(function(d) { return d.vote_count > 10 })
    .map(function (d) { return parseFloat(d.vote_average); });
var xval = data
    .filter(function(d) { return d.vote_count > 10 })
    .map(function (d) { return d.first_air_date.getTime() / 1000; });
var lr = linearRegression(yval,xval);
// Object { slope: -3.754543948800799e-10, intercept: 7.0808230581192815, r2: 0.038528573017115 }

{% endcodeblock %}

I wanted to include movies as a comparison to TV. Here's a plot of all movies across time.

<iframe src="{{ root_url }}/images/tv_movies/index_movie.html" marginwidth="0" marginheight="0" scrolling="no" width="800" height="500"></iframe>

It's important to note that I removed all movies with less than 1000 votes. This is completely 100% unfair, BUT I am very proud of my figures here and things get a little laggy when including too many movies in the plot. Nonetheless, movies seem to be getting worse over time! More dramatically than TV shows!


{% codeblock lang:javascript %}
var yval = data
    .filter(function(d) { return d.vote_count > 1000 })
    .map(function (d) { return parseFloat(d.vote_average); });
var xval = data
    .filter(function(d) { return d.vote_count > 1000 })
    .map(function (d) { return d.first_air_date.getTime() / 1000; });
var lr = linearRegression(yval,xval);
// Object { slope: -8.11645196776367e-10, intercept: 7.659366705415847, r2: 0.16185069580043676 }
{% endcodeblock %}

Okay, so this was a fun little analysis, but I have to come out and say that I wasn't too happy with my dataset and the conclusions we can draw from this analysis are only as good as the dataset.

The first limitation is that recent content is much more likely to receive a rating than older content, which could systematically bias the ratings of older content (e.g., only good shows from before 2000 receive ratings). It's easy to imagine how this would lead us to believing that all older content is better than it actually was.

Also, TMDb seems to have IMDB type tastes by which I mean its dominated by young males. For instance, while I don't like the show "Keeping up the Kardashians," it's definitely not the worst show ever. Also, "Girls" is an amazing show which gets no respect here. The quality of a show is in the eye of the beholder, which in this case seems to be boys.

I would have used Rotten Tomatoes' API, but they don't provide access to TV ratings.

Even with all these caveats in mind, it's hard to defend my "golden age of TV" hypothesis. Instead, it seems like there is just more content being produced, which leads to more good shows (yay!), but the average show is no better or worse than previously.
