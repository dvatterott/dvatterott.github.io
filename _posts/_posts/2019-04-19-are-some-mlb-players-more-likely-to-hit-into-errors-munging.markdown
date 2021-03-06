---
layout: post
title: "Are some mlb players more likely to hit into errors than others: Data Munging"
date: 2019-04-19 11:02:56 -0500
comments: true
categories: [python, data science, bash]
---

I recently found myself wondering if some baseball players are more likely to hit into errors than others. In theory, the answer should be "no" since fielders produce errors regardless of who is hitting. Nonetheless, it's also possible that some hitters "force" errors by hitting the ball harder or running to first base faster. 

In order to evaluate this possibility, I found play-by-play data on [retrosheet.org](https://www.retrosheet.org/). This data contains row by row data describing each event (e.g., a hit, stolen base etc) in a baseball game. I've posted this analysis on [github](https://github.com/dvatterott/mlb_errors) and will walk through it here. 

The user is expected to input what year's data they want. I write the code's output for the year 2018 as comments. The code starts by downloading and unzipping the data.

{% codeblock lang:bash %}
YEAR=$1
FILE_LOC=https://www.retrosheet.org/events/${YEAR}eve.zip

echo "---------DOWNLOAD----------"
wget -nc $FILE_LOC -O ./raw_data/${YEAR}.zip

echo "---------UNPACK----------"
mkdir raw_data/${YEAR}/
unzip -o raw_data/${YEAR}.zip -d raw_data/${YEAR}/
{% endcodeblock %}

The unzipped data contain play-by-play data in files with the EVN or EVA extensions. Each team's home stadium has its own file. I combine all the play-by play eveSSplants (.EVN and .EVA files) into a single file. I then remove all non batting events (e.g., balk, stolen base etc).

I also combine all the roster files (.ROS) into a single file. 

{% codeblock lang:bash %}
# export playbyplay to single file
mkdir processed_data/${YEAR}/
find raw_data/${YEAR}/ -regex '.*EV[A|N]' | \
	xargs grep play > \
	./processed_data/${YEAR}/playbyplay.out

# get all plate appearances from data (and hitter). remove all non plate appearance rows
cat ./processed_data/${YEAR}/playbyplay.out | \
	awk -F',' '{print $4","$7}' | \
	grep -Ev ',[A-Z]{3}[0-9]{2}' | \
	grep -Ev ',(NP|BK|CS|DI|OA|PB|WP|PO|POCS|SB|FLE)' > \
	./processed_data/${YEAR}/batters.out

# one giant roster file
find raw_data/${YEAR}/ -name '*ROS' | \
	xargs awk -F',' '{print $1" "$2" "$3}' > \
	./processed_data/${YEAR}/players.out
{% endcodeblock %}

In this next few code blocks I print some data just to see what I am working with. For instance, I print out players with the most plate appearances. I was able to confirm these numbers with [baseball-reference](https://baseball-reference.com). This operation requires me to groupby player and count the rows. I join this file with the roster file to get player's full names.

{% codeblock lang:bash %}
echo "---------PLAYERS WITH MOST PLATE APPEARANCES----------"
cat ./processed_data/${YEAR}/batters.out | \
	awk -F, '{a[$1]++;}END{for (i in a)print i, a[i];}' | \
	sort -k2 -nr | \
	head > \
	./processed_data/${YEAR}/most_pa.out
join <(sort -k 1 ./processed_data/${YEAR}/players.out) <(sort -k 1 ./processed_data/${YEAR}/most_pa.out) | \
	uniq | \
	sort -k 4 -nr | \
	head | \
	awk '{print $3", "$2", "$4}'

#---------PLAYERS WITH MOST PLATE APPEARANCES----------
#Francisco, Lindor, 745
#Trea, Turner, 740
#Manny, Machado, 709
#Cesar, Hernandez, 708
#Whit, Merrifield, 707
#Freddie, Freeman, 707
#Giancarlo, Stanton, 706
#Nick, Markakis, 705
#Alex, Bregman, 705
#Marcus, Semien, 703
{% endcodeblock %}

Here's the players with the most hits. Notice that I filter out all non-hits in the grep, then group by player.

{% codeblock lang:bash %}
echo "---------PLAYERS WITH MOST HITS----------"
cat ./processed_data/${YEAR}/batters.out | \
	grep -E ',(S|D|T|HR)' | \
	awk -F, '{a[$1]++;}END{for (i in a)print i, a[i];}' | \
	sort -k2 -nr | \
	head

#---------PLAYERS WITH MOST HITS----------
#merrw001 192
#freef001 191
#martj006 188
#machm001 188
#yelic001 187
#markn001 185
#castn001 185
#lindf001 183
#peraj003 182
#blacc001 182
{% endcodeblock %}

Here's the players with the most at-bats.

{% codeblock lang:bash %}
echo "---------PLAYERS WITH MOST AT BATS----------"
cat ./processed_data/${YEAR}/batters.out | \
	grep -Ev 'SF|SH' | \
	grep -E ',(S|D|T|HR|K|[0-9]|E|DGR|FC)' | \
	awk -F, '{a[$1]++;}END{for (i in a)print i, a[i];}' > \
	./processed_data/${YEAR}/abs.out
cat ./processed_data/${YEAR}/abs.out | sort -k2 -nr | head

#---------PLAYERS WITH MOST AT BATS----------
#turnt001 664
#lindf001 661
#albio001 639
#semim001 632
#peraj003 632
#merrw001 632
#machm001 632
#blacc001 626
#markn001 623
#castn001 620
{% endcodeblock %}

And, finally, here's the players who hit into the most errors.

{% codeblock lang:bash %}
echo "---------PLAYERS WHO HIT INTO THE MOST ERRORS----------"
cat ./processed_data/${YEAR}/batters.out | \
    	grep -Ev 'SF|SH' | \
	grep ',E[0-9]' | \
	awk -F, '{a[$1]++;}END{for (i in a)print i, a[i];}' > \
	./processed_data/${YEAR}/errors.out
cat ./processed_data/${YEAR}/errors.out | sort -k2 -nr | head

#---------PLAYERS WHO HIT INTO THE MOST ERRORS----------
#gurry001 13
#casts001 13
#baezj001 12
#goldp001 11
#desmi001 11
#castn001 10
#bogax001 10
#andum001 10
#turnt001 9
#rojam002 9
{% endcodeblock %}

Because players with more at-bats hit into more errors, I need to take the number of at-bats into account. I also filter out all players with less than 250 at bats. I figure we only want players with lots of opportunities to create errors.

{% codeblock lang:bash %}
echo "---------PLAYERS WITH MOST ERRORS PER AT BAT----------"
join -e"0" -a1 -a2 <(sort -k 1 ./processed_data/${YEAR}/abs.out) -o 0 1.2 2.2 <(sort -k 1 ./processed_data/${YEAR}/errors.out) | \
	uniq | \
	awk -v OFS=', ' '$2 > 250 {print $1, $3, $2, $3/$2}' >  \
	./processed_data/${YEAR}/errors_abs.out
cat ./processed_data/${YEAR}/errors_abs.out | sort -k 4 -nr | head

#---------PLAYERS WITH MOST ERRORS PER AT BAT----------
#pereh001, 8, 316, 0.0253165
#gurry001, 13, 537, 0.0242086
#andre001, 9, 395, 0.0227848
#casts001, 13, 593, 0.0219224
#desmi001, 11, 555, 0.0198198
#baezj001, 12, 606, 0.019802
#garca003, 7, 356, 0.0196629
#bogax001, 10, 512, 0.0195312
#goldp001, 11, 593, 0.0185497
#iglej001, 8, 432, 0.0185185
{% endcodeblock %}

At-bats is great but even better is to remove strike-outs and just look at occurences when a player hit the ball into play. I remove all players with less than 450 balls hit into play which limits us to just 37 players but the players have enough reps to make the estimates more reliable.

{% codeblock lang:bash %}
echo "---------PLAYERS WITH MOST ERRORS PER BALL IN PLAY----------"
cat ./processed_data/${YEAR}/batters.out | \
	grep -Ev 'SF|SH' | \
	grep -E ',(S|D|T|HR|[0-9]|E|DGR|FC)' | \
	awk -F, '{a[$1]++;}END{for (i in a)print i, a[i];}' > \
	./processed_data/${YEAR}/bip.out
join -e"0" -a1 -a2 <(sort -k 1 ./processed_data/${YEAR}/bip.out) -o 0 1.2 2.2 <(sort -k 1 ./processed_data/${YEAR}/errors.out) | \
	uniq | \
	awk -v OFS=', ' '$2 > 450 {print $1, $3, $2, $3/$2}' > \
	./processed_data/${YEAR}/errors_bip.out
cat ./processed_data/${YEAR}/errors_bip.out | sort -k 4 -nr | head

#---------PLAYERS WITH MOST ERRORS PER BALL IN PLAY----------
#casts001, 13, 469, 0.0277186
#gurry001, 13, 474, 0.0274262
#castn001, 10, 469, 0.021322
#andum001, 10, 476, 0.0210084
#andeb006, 9, 461, 0.0195228
#turnt001, 9, 532, 0.0169173
#simma001, 8, 510, 0.0156863
#lemad001, 7, 451, 0.0155211
#sancc001, 7, 462, 0.0151515
#freef001, 7, 486, 0.0144033
{% endcodeblock %}

Now we have some data. In future posts I will explore how we can use statistics to evaluate whether some players are more likely to hit into errors than others.

Check out the [companion post](https://danvatterott.com/blog/2019/06/04/are-some-mlb-players-more-likely-to-hit-into-errors-statistics/) that statistically explores this question.