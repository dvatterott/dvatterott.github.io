---
layout: post
title: "Introducing Predeval"
date: 2019-01-29 20:27:57 -0600
comments: true
categories: [python, machine-learning, data science, open source]
---

[Predeval](https://predeval.readthedocs.io/en/latest/) is software designed to help you identify changes in a model's output.

For instance, you might be tasked with building a model to predict churn. When you deploy this model in production, you have to wait to learn which users churned in order to know how your model performed. While Predeval will not free you from this wait, it can provide initial signals as to whether the model is producing reasonable (i.e., expected) predictions. Unexpected predictions *might* reflect a poor performing model. They also *might* reflect a change in your input data. Either way, something has changed and you will want to investigate further.

Using predeval, you can detect changes in model output ASAP. You can then use python's libraries to build a surrounding alerting system that will signal a need to investigate. This system should give you additional confidence that your model is performing reasonably. Here's a [post](https://danvatterott.com/blog/2018/06/02/random-weekly-reminders/) where I configure an alerting system using python, mailutils, and postfix (although the alerting system is not built around predeval).

Predeval operates by forming expectations about what your model's outputs will look like. For example, you might give predeval the model's output from a validation dataset. Predeval will then compare new outputs to the outputs produced by the validation dataset, and will report whether it detects a difference.

Predeval works with models producing both categorical and continuous outputs.

Here's an [example](https://predeval.readthedocs.io/en/latest/usage.html#categoricalevaluator) of predeval with a model producing categorical outputs. Predeval will (by default) check whether all expected output categories are present, and whether the output categories occur at their expected frequencies (using a [Chi-square test of independence of variables in a contingency table](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html)).

Here's an [example](https://predeval.readthedocs.io/en/latest/usage.html#continuousevaluator) of predeval with a model producing continuous outputs. Predeval will (by default) check whether the new output have a minimum lower than expected, a maximum greater than expected, a different mean, a different standard deviation, and whether the new output are distributed as expected (using a [Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp))

I've tried to come up with reasonable defaults for determining whether data are different, but you can also [set these thresholds yourself](https://predeval.readthedocs.io/en/latest/usage.html#updating-test-parameters). You can also [choose what comparison tests to run](https://predeval.readthedocs.io/en/latest/usage.html#changing-evaluation-tests) (e.g., checking the minimum, maximum etc.).

You will likely need to save your predeval objects so that you can apply them to future data. Here's an [example](https://predeval.readthedocs.io/en/latest/usage.html#saving-and-loading-your-evaluator) of saving the objects.

Documentation about how to install predeval can be found [here](https://predeval.readthedocs.io/en/latest/installation.html#installation).

If you have comments about improvements or would like to [contribute](https://predeval.readthedocs.io/en/latest/contributing.html), please reach out!
