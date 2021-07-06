---
layout: post
title: "Custom Email Alerts in Airflow"
date: 2018-08-29 18:19:42 -0500
comments: true
categories: [data science, data engineering, open source]
---

[Apache Airflow](https://airflow.apache.org/) is great for coordinating automated jobs, and it provides a simple interface for sending email alerts when these jobs fail. Typically, one can request these emails by setting <code>email_on_failure</code> to <code>True</code> in your operators.

These email alerts work great, but I wanted to include additional links in them (I wanted to include a link to my spark cluster which can be grabbed from the [Databricks Operator](https://airflow.incubator.apache.org/_modules/airflow/contrib/operators/databricks_operator.html#DatabricksSubmitRunOperator)). Here's how I created a custom email alert on job failure.

First, I set <code>email_on_failure</code> to <code>False</code> and use the operators's <code>on_failure_callback</code>. I give <code>on_failure_callback</code> the function described below.

{% codeblock lang:python %}
from airflow.utils.email import send_email

def notify_email(contextDict, **kwargs):
    """Send custom email alerts."""

    # email title.
    title = "Airflow alert: {task_name} Failed".format(**contextDict)

    # email contents
    body = """
    Hi Everyone, <br>
    <br>
    There's been an error in the {task_name} job.<br>
    <br>
    Forever yours,<br>
    Airflow bot <br>
    """.format(**contextDict)

    send_email('you_email@address.com', title, body)
{% endcodeblock %}

<code>send_email</code> is a function imported from Airflow. <code>contextDict</code> is a dictionary given to the callback function on error. Importantly, <code>contextDict</code> contains lots of relevant information. This includes the Task Instance (key='ti') and Operator Instance (key='task') associated with your error. I was able to use the Operator Instance, to grab the relevant cluster's address and I included this address in my email (this exact code is not present here).

To use the <code>notify_email</code>, I set <code>on_failure_callback</code> equal to <code>notify_email</code>.

I write out a short example airflow dag below.

{% codeblock lang:python %}
from airflow.models import DAG
from airflow.operators import PythonOperator
from airflow.utils.dates import days_ago

args = {
  'owner': 'me',
  'description': 'my_example',
  'start_date': days_ago(1)
}

# run every day at 12:05 UTC
dag = DAG(dag_id='example_dag', default_args=args, schedule_interval='0 5 * * *')

def print_hello():
  return 'hello!'

py_task = PythonOperator(task_id='example',
                         python_callable=print_hello,
                         on_failure_callback=notify_email,
                         dag=dag)

py_task
{% endcodeblock %}

Note where set <code>on_failure_callback</code> equal to <code>notify_email</code> in the <code>PythonOperator</code>.

Hope you find this helpful! Don't hesitate to reach out if you have a question.
