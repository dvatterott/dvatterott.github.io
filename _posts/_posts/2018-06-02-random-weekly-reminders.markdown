---
layout: post
title: "Random Weekly Reminders"
date: 2018-06-02 17:58:52 -0500
comments: true
categories: [automation, python]
---

I constantly use google calendar to schedule reminder emails, but I want some of my reminders to be stochastic!

Google calendar wants all their events to occur on a regular basis (e.g., every Sunday), but I might want a weekly reminder email which occurs on a random day each the week.

I wrote a quick set of [python scripts](https://github.com/dvatterott/reminder_email) which handle this situation.

The script [find_days.py](https://github.com/dvatterott/reminder_email/blob/master/find_days.py) chooses a random day each week (over a month) on which a reminder email should be sent. These dates are piped to a text file (dates.txt). The script [send_email.py](https://github.com/dvatterott/reminder_email/blob/master/send_email.py) reads this text file and sends a reminder email to me if the current date matches one of the dates in dates.txt.

I use [cron](https://help.ubuntu.com/community/CronHowto) to automatically run these scripts on a regular basis. Cron runs find_days.py on the first of each month and runs send_email.py every day. I copied my cron script as [cron_job.txt](https://github.com/dvatterott/reminder_email/blob/master/cron_job.txt).

I use mailutils and postfix to send the reminder emails from the machine. Check out [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-postfix-as-a-send-only-smtp-server-on-ubuntu-14-04) for how to set up a send only mail server. The trickiest part of this process was repeatedly telling gmail that my emails were not spam.

Now I receive my weekly reminder on an unknown date so I can *act* spontaneous!
