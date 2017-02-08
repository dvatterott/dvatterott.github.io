---
layout: post
title: "Creating videos of NBA action with Sportsvu data"
date: 2016-06-16 08:11:47 -0400
comments: true
categories: [python, plotting, nba, open source, data analytics]
---


All basketball teams have a camera system called [SportVU](https://en.wikipedia.org/wiki/SportVU) installed in their arenas. These camera systems track players and the ball throughout a basketball game. 

The data produced by sportsvu camera systems used to be freely available on NBA.com, but was recently removed (I have no idea why). Luckily, the data for about 600 games are available on [neilmj's github](https://github.com/neilmj/BasketballData). In this post, I show how to create a video recreation of a given basketball play using the sportsvu data. 

This code is also available as a jupyter notebook on my [github](https://github.com/dvatterott/jupyter_notebooks). 


{% codeblock lang:python %}
#import some libraries
import matplotlib.pyplot as plt, pandas as pd, numpy as np, matplotlib as mpl
from __future__ import print_function

mpl.rcParams['font.family'] = ['Bitstream Vera Sans']
{% endcodeblock %}

The data is provided as a json. Here's how to import the python json library and load the data. I'm a T-Wolves fan, so the game I chose is a wolves game.


{% codeblock lang:python %}
import json #import json library
json_data = open('/home/dan-laptop/github/BasketballData/2016.NBA.Raw.SportVU.Game.Logs/0021500594.json') #import the data from wherever you saved it.
data = json.load(json_data) #load the data
{% endcodeblock %}

Let's take a quick look at the data. It's a dictionary with three keys: gamedate, gameid, and events. Gamedate and gameid are the date of this game and its specific id number, respectively. Events is the structure with data we're interested in. 


{% codeblock lang:python %}
data.keys()
{% endcodeblock %}




    [u'gamedate', u'gameid', u'events']



Lets take a look at the first event. The first event has an associated eventid number. We will use these later. There's also data for each player on the visiting and home team. We will use these later too. Finally, and most importantly, there's the "moments." There are 25 moments for each second of the "event" (the data is sampled at 25hz). 


{% codeblock lang:python %}
data['events'][0].keys()
{% endcodeblock %}




    [u'eventId', u'visitor', u'moments', u'home']



Here's the first moment of the first event. The first number is the quarter. The second number is the time of the event in milliseconds. The third number is the number of seconds left in the quarter (the 1st quarter hasn't started yet, so 12 * 60 = 720). The fourth number is the number of seconds left on the shot clock. I am not sure what fourth number (None) represents. 

The final matrix is 11x5 matrix. The first row describes the ball. The first two columns are the teamID and the playerID of the ball (-1 for both because the ball does not belong to a team and is not a player). The 3rd and 4th columns are xy coordinates of the ball. The final column is the height of the ball (z coordinate). 

The next 10 rows describe the 10 players on the court. The first 5 players belong to the home team and the last 5 players belong to the visiting team. Each player has his teamID, playerID, xy&z coordinates (although I don't think players' z coordinates ever change). 


{% codeblock lang:python %}
data['events'][0]['moments'][0]
{% endcodeblock %}




    [1,
     1452903036782,
     720.0,
     24.0,
     None,
     [[-1, -1, 44.16456, 26.34142, 5.74423],
      [1610612760, 201142, 45.46259, 32.01456, 0.0],
      [1610612760, 201566, 10.39347, 24.77219, 0.0],
      [1610612760, 201586, 25.86087, 25.55881, 0.0],
      [1610612760, 203460, 47.28525, 17.76225, 0.0],
      [1610612760, 203500, 43.68634, 26.63098, 0.0],
      [1610612750, 708, 55.6401, 25.55583, 0.0],
      [1610612750, 2419, 47.95942, 31.66328, 0.0],
      [1610612750, 201937, 67.28725, 25.10267, 0.0],
      [1610612750, 203952, 47.28525, 17.76225, 0.0],
      [1610612750, 1626157, 49.46814, 24.24193, 0.0]]]



Alright, so we have the sportsvu data, but its not clear what each event is. Luckily, the NBA also provides play by play (pbp) data. I write a function for acquiring play by play game data. This function collects (and trims) the play by play data for a given sportsvu data set. 


{% codeblock lang:python %}
def acquire_gameData(data):
    import requests
    header_data = { #I pulled this header from the py goldsberry library
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'en-US,en;q=0.8',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64)'\
        ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.82 '\
        'Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9'\
        ',image/webp,*/*;q=0.8',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive'
    }
    game_url = 'http://stats.nba.com/stats/playbyplayv2?EndPeriod=0&EndRange=0&GameID='+data['gameid']+\
                '&RangeType=0&StartPeriod=0&StartRange=0' #address for querying the data
    response = requests.get(game_url,headers = header_data) #go get the data
    headers = response.json()['resultSets'][0]['headers'] #get headers of data
    gameData = response.json()['resultSets'][0]['rowSet'] #get actual data from json object
    df = pd.DataFrame(gameData, columns=headers) #turn the data into a pandas dataframe
    df = df[[df.columns[1], df.columns[2],df.columns[7],df.columns[9],df.columns[18]]] #there's a ton of data here, so I trim  it doown
    df['TEAM'] = df['PLAYER1_TEAM_ABBREVIATION']
    df = df.drop('PLAYER1_TEAM_ABBREVIATION', 1)
    return df
{% endcodeblock %}

Below I show what the play by play data looks like. There's a column for event number (eventnum). These event numbers match up with the event numbers from the sportsvu data, so we will use this later for seeking out specific plays in the sportsvu data. There's a column for the event type (eventmsgtype). This column has a number describing what occured in the play. I list these number codes in the comments below. 

There's also short text descriptions of the plays in the home description and visitor description columns. Finally, I use the team column to represent the primary team involved in a play. 

I stole the idea of using play by play data from [Raji Shah](http://projects.rajivshah.com/sportvu/PBP_NBA_SportVu.html). 


{% codeblock lang:python %}
df = acquire_gameData(data)
df.head()
#EVENTMSGTYPE
#1 - Make 
#2 - Miss 
#3 - Free Throw 
#4 - Rebound 
#5 - out of bounds / Turnover / Steal 
#6 - Personal Foul 
#7 - Violation 
#8 - Substitution 
#9 - Timeout 
#10 - Jumpball 
#12 - Start Q1? 
#13 - Start Q2?
{% endcodeblock %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EVENTNUM</th>
      <th>EVENTMSGTYPE</th>
      <th>HOMEDESCRIPTION</th>
      <th>VISITORDESCRIPTION</th>
      <th>TEAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>12</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
      <td>Jump Ball Adams vs. Towns: Tip to Ibaka</td>
      <td>None</td>
      <td>OKC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>Westbrook Out of Bounds Lost Ball Turnover (P1...</td>
      <td>None</td>
      <td>OKC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
      <td>None</td>
      <td>MISS Wiggins 16' Jump Shot</td>
      <td>MIN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>Westbrook REBOUND (Off:0 Def:1)</td>
      <td>None</td>
      <td>OKC</td>
    </tr>
  </tbody>
</table>
</div>



When viewing the videos, its nice to know what players are on the court. I like to depict this by labeling each player with their number. Here I create a dictionary that contains each player's id number (these are assigned by nba.com) as the key and their jersey number as the associated value. 


{% codeblock lang:python %}
player_fields = data['events'][0]['home']['players'][0].keys()
home_players = pd.DataFrame(data=[i for i in data['events'][0]['home']['players']], columns=player_fields)
away_players = pd.DataFrame(data=[i for i in data['events'][0]['visitor']['players']], columns=player_fields)
players = pd.merge(home_players, away_players, how='outer')
jerseydict = dict(zip(players.playerid.values, players.jersey.values))
{% endcodeblock %}

Alright, almost there! Below I write some functions for creating the actual video! First, there's a short function for placing an image of the basketball court beneath our depiction of players moving around. This image is from gmf05's github, but I will provide it on [mine](https://github.com/dvatterott/nba_project) too. 

Much of this code is either straight from [gmf05's github](https://github.com/gmf05/nba/blob/master/scripts/notebooks/svmovie.ipynb) or slightly modified. 


{% codeblock lang:python %}
# Animation function / loop
def draw_court(axis):
    import matplotlib.image as mpimg
    img = mpimg.imread('./nba_court_T.png') #read image. I got this image from gmf05's github.
    plt.imshow(img,extent=axis, zorder=0) #show the image. 

def animate(n): #matplotlib's animation function loops through a function n times that draws a different frame on each iteration
    for i,ii in enumerate(player_xy[n]): #loop through all the players
        player_circ[i].center = (ii[1], ii[2]) #change each players xy position
        player_text[i].set_text(str(jerseydict[ii[0]])) #draw the text for each player. 
        player_text[i].set_x(ii[1]) #set the text x position
        player_text[i].set_y(ii[2]) #set text y position
    ball_circ.center = (ball_xy[n,0],ball_xy[n,1]) #change ball xy position
    ball_circ.radius = 1.1 #i could change the size of the ball according to its height, but chose to keep this constant
    return tuple(player_text) + tuple(player_circ) + (ball_circ,)

def init(): #this is what matplotlib's animation will create before drawing the first frame. 
    for i in range(10): #set up players
        player_text[i].set_text('')
        ax.add_patch(player_circ[i])
    ax.add_patch(ball_circ) #create ball
    ax.axis('off') #turn off axis
    dx = 5
    plt.xlim([0-dx,100+dx]) #set axis
    plt.ylim([0-dx,50+dx])  
    return tuple(player_text) + tuple(player_circ) + (ball_circ,)
{% endcodeblock %}

The event that I want to depict is event 41. In this event, Karl Anthony Towns misses a shot, grabs his own rebounds, and puts it back in.


{% codeblock lang:python %}
df[37:38]
{% endcodeblock %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EVENTNUM</th>
      <th>EVENTMSGTYPE</th>
      <th>HOMEDESCRIPTION</th>
      <th>VISITORDESCRIPTION</th>
      <th>TEAM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>41</td>
      <td>1</td>
      <td>None</td>
      <td>Towns 1' Layup (2 PTS)</td>
      <td>MIN</td>
    </tr>
  </tbody>
</table>
</div>



We need to find where event 41 is in the sportsvu data structure, so I created a function for finding the location of a particular event. I then create a matrix with position data for the ball and a matrix with position data for each player for event 41. 


{% codeblock lang:python %}
#the order of events does not match up, so we have to use the eventIds. This loop finds the correct event for a given id#.
search_id = 41
def find_moment(search_id):
    for i,events in enumerate(data['events']):
        if events['eventId'] == str(search_id):
            finder = i
            break
    return finder

event_num = find_moment(search_id) 
ball_xy = np.array([x[5][0][2:5] for x in data['events'][event_num]['moments']]) #create matrix of ball data
player_xy = np.array([np.array(x[5][1:])[:,1:4] for x in data['events'][event_num]['moments']]) #create matrix of player data
{% endcodeblock %}

Okay. We're actually there! Now we get to create the video. We have to create figure and axes objects for the animation to draw on. Then I place a picture of the basketball court on this plot. Finally, I create the circle and text objects that will move around throughout the video (depicting the ball and players). The location of these objects are then updated in the animation loop.


{% codeblock lang:python %}
import matplotlib.animation as animation

fig = plt.figure(figsize=(15,7.5)) #create figure object
ax = plt.gca() #create axis object

draw_court([0,100,0,50]) #draw the court
player_text = range(10) #create player text vector
player_circ = range(10) #create player circle vector
ball_circ = plt.Circle((0,0), 1.1, color=[1, 0.4, 0]) #create circle object for bal
for i in range(10): #create circle object and text object for each player
    col=['w','k'] if i<5 else ['k','w'] #color scheme
    player_circ[i] = plt.Circle((0,0), 2.2, facecolor=col[0],edgecolor='k') #player circle
    player_text[i] = ax.text(0,0,'',color=col[1],ha='center',va='center') #player jersey # (text)

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,np.size(ball_xy,0)), init_func=init, blit=True, interval=5, repeat=False,\
                             save_count=0) #function for making video
ani.save('Event_%d.mp4' % (search_id),dpi=100,fps=25) #function for saving video
plt.close('all') #close the plot
{% endcodeblock %}

{% video {{ root_url }}/images/Event_41.mp4 %}

I've been told this video does not work for all users. I've also posted it on [youtube](https://youtu.be/ZPvQOorvVtI).