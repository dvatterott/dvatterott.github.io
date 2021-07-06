---
layout: post
title: "Looking Towards the Future of Automated machine-learning"
date: 2018-11-03 08:27:07 -0500
comments: true
categories: [python, machine-learning, data science, open source]
---

I recently gave a [presentation](https://vencafstl.org/event/where-automated-machine-learning-fits-in-your-data-science-toolbox-prepare-ai) at [Venture Cafe](https://vencafstl.org/) describing how I see automation changing python, machine-learning workflows in the near future.

In this post, I highlight the presentation's main points. You can find the slides [here](https://danvatterott.com/presentations/automation_presentation/).

From [Ray Kurzweil's](https://en.wikipedia.org/wiki/Ray_Kurzweil) excitement about a [technological singularity](https://en.wikipedia.org/wiki/Technological_singularity) to Elon Musk's warnings about an [A.I. Apocalypse](https://www.vanityfair.com/news/2017/03/elon-musk-billion-dollar-crusade-to-stop-ai-space-x), automated machine-learning evokes strong feelings. Neither of these futures will be true in the near-term, but where will automation fit in your machine-learning workflows?

Our existing machine-learning workflows might look a little like the following (please forgive the drastic oversimplification of a purely sequential progression across stages!).

<img src="{{ root_url }}/presentations/automation_presentation/slides/data_science_pipeline/ds_pipeline.png" style="background-color:white;"/>

Where does automation exist in this workflow? Where can automation improve this workflow?

Not all these stages are within the scope of machine-learning. For instance, while you should automate gathering data, I view this as a data engineering problem. In the image below, I depict the stages that I consider ripe for automation, and the stages I consider wrong for automation. For example, data cleaning is too idiosyncratic to each dataset for true automation. I "X" out model evaluation as wrong for automation. In retrospect, I believe this is a great place for automation, but I don't know of any existing python packages handling it.

<img src="{{ root_url }}/presentations/automation_presentation/slides/libraries_pipeline/tpot_pipeline.png" style="background-color:white;"/>

I depict feature engineering and model selection as the most promising areas for automation. I consider feature engineering as the stage where advances in automation can have the largest impact on your model performance. In the presentation, I include a [strong quote](https://www.quora.com/What-generally-improves-a-models-score-more-on-average-feature-engineering-or-hyperparameter-tuning) from a Quora user saying that [hyper-parameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization) (a part of model selection) "hardly matters at all." I agree with the sentiment of this quote, but it's not true. Choosing roughly the correct hyper-parameter values is *VERY* important, and choosing the very best hyper-parameter values can be equally important depending on how your model is used. I highlight feature engineering over model selection because automated model selection is largely solved. For example [grid-search](http://scikit-learn.org/stable/modules/grid_search.html) automates model selection. It's not a fast solution, but given infinite time, it will find the best hyper-parameter values!

There are many python libraries automating these parts of the workflow. I highlight three libraries that automate feature engineering.

<img src="{{ root_url }}/presentations/automation_presentation/slides/feature_automation/tpot-logo.jpg" width="200" style="background-color:white;"/>

The first is [teapot](https://github.com/EpistasisLab/tpot). Teapot (more or less) takes all the different operations and models available in [scikit-learn](http://scikit-learn.org/stable/), and allows you to assemble these operations into a pipeline. Many of these operations (e.g., [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)) are forms of feature engineering. Teapot measures which operations lead to the best model performance. Because Teapot enables users to assemble *SO MANY* different operations, it utilizes a [genetic search algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) to search through the different possibilities more efficiently than grid-search would.

The second is [auto_ml](https://github.com/ClimbsRocks/auto_ml). In auto_ml users simply pass a dataset to the software and it will do model selection and hyper-parameter tuning for you. Users can also [ask the software to train a deep learning model that will learn new features](https://auto-ml.readthedocs.io/en/latest/deep_learning.html#feature-learning) from your dataset. The authors claim this approach can improve model accuracy by 5%.  

<img src="{{ root_url }}/presentations/automation_presentation/slides/feature_automation/featuretools.png" width="400" style="background-color:white;"/>

The third is [feature tools](https://github.com/Featuretools/featuretools). Feature Tools is the piece of automation software whose future I am most excited about. I find this software exciting because users can feed it pre-aggregated data. Most machine-learning models expect that for each value of the [response variable](https://www.quora.com/What-is-a-response-variable-in-statistics), you supply a vector of explanatory variables. This is an example of aggregated data. Teapot and auto_ml both expect users to supply aggregated data. Lots of important information is lost in the aggregation process, and allowing automation to thoroughly explore different aggregations will lead to predictive features that we would not have created otherwise (any many believe this is why deep learning is so effective). Feature tools explores different aggregations all while creating easily interpreted variables (in contrast to deep learning). While I am excited about the future of feature tools, it is a new piece of software and has a ways to go before I use it in my workflows. Like most automation machine-learning software it's very slow/resource intensive. Also, the software is not very intuitive. That said, I created a [binder notebook](https://mybinder.org/v2/gh/dvatterott/explore_feature_automation/master) demoing feature tools, so check it out yourself!

We should always keep in mind the possible dangers of automation and machine-learning. Removing humans from decisions accentuates biases baked into data and algorithms. These accentuated biases can have dangerous effects. We should carefully choose which decisions we're comfortable automating and what safeguards to build around these decisions. Check out [Cathy O'Neil's](https://en.wikipedia.org/wiki/Cathy_O%27Neil) amazing [Weapons for Math Destruction](https://en.wikipedia.org/wiki/Weapons_of_Math_Destruction) for an excellent treatment of the topic.  

This post makes no attempt to give an exhaustive view of automated machine-learning. This is my single view point on where I think automated machine-learning can have an impact on your python workflows in the near-term. For a more thorough view of automated machine-learning, check out this [presentation](https://twitter.com/randal_olson/status/992105498894831616) by [Randy Olson](http://www.randalolson.com/) (the creator of teapot).
