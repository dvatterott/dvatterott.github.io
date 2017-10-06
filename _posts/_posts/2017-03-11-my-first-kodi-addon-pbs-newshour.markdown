---
layout: post
title: "My first Kodi Addon - PBS NewsHour (a tutorial)"
date: 2017-03-11 13:21:03 -0500
comments: true
categories: [python, open source, Kodi, HTPC]
---   

I've been using [Kodi/XBMC](https://kodi.tv/) since 2010. It provides a flexible and (relatively) intuitive interface for interacting with content through your TV (much like an apple TV). One of the best parts of Kodi is the addons - these are apps that you can build or download. For instance, I use the NBA League Pass addon for watching Wolves games. I've been looking for a reason to build my own Kodi addon for years.

Enter [PBS NewsHour](http://www.pbs.org/newshour/). If you're not watching PBS NewsHour, I'm not sure what you're doing with your life because it's the shit. It rocks. PBS NewsHour disseminates all their content on youtube and their website. For the past couple years, I've been watching their broadcasts every morning through the [Youtube addon](http://kodi.wiki/view/Add-on:YouTube). This works fine, but it's clunky. I decided to stream line watching the NewsHour by building a Kodi addon for it.

I used [this tutorial](http://forum.kodi.tv/showthread.php?tid=254207) to build a Kodi addon that accesses the PBS NewsHour content through the youtube addon. This addon can be found on [my github](https://github.com/dvatterott/Kodi_addons/tree/master/plugin.video.pbsnewshouryoutube). The addon works pretty well, but it includes links to all NewsHour's content, and I only want the full episodes. I am guessing I could have modified this addon to get what I wanted, but I really wanted to build my own addon from scratch.

The addon I built is available on [my github](https://github.com/dvatterott/Kodi_addons/tree/master/plugin.video.pbsnewshour). To build my addon, I used [this tutorial](http://kodi.wiki/view/HOW-TO:Video_addon), and some code from [this github](https://github.com/learningit/Kodi-plugins-source) repository. Below I describe how the addon works. I only describe the file default.py because this file does the majority of the work, and I found the linked tutorials did a good job explaining the other files.

I start by importing libraries that I will use. Most these libraries are used for scraping content off the web. I then create some basic variables to describe the addon's name (addonID), its name in kodi (base_url), the number used to refer to it (addon_handle - I am not sure how this number is used), and current arguments sent to my addon (args).

{% codeblock lang:python %}
import zlib
import json
import sys
import urlparse
import xbmc
import xbmcgui
import xbmcplugin

import urllib2
import re

addonID = 'plugin.video.pbsnewshour'

base_url = sys.argv[0]
addon_handle = int(sys.argv[1])
args = urlparse.parse_qs(sys.argv[2][1:])
{% endcodeblock %}

The next function, getRequest, gathers html from a website (specified by the variable url). The dictionary httpHeaders tells the website a little about myself, and how I want the html. I use urllib2 to get a compressed version of the html, which is decompressed using zlib.

{% codeblock lang:python %}
# -----------  Create some functions for fetching videos ---------------
# https://github.com/learningit/Kodi-plugins-source/blob/master/script.module.t1mlib/lib/t1mlib.py
UTF8 = 'utf-8'
USERAGENT = """Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36"""
httpHeaders = {'User-Agent': USERAGENT,
               'Accept': "application/json, text/javascript, text/html,*/*",
               'Accept-Encoding': 'gzip,deflate,sdch',
               'Accept-Language': 'en-US,en;q=0.8'
               }


def getRequest(url, udata=None, headers=httpHeaders):
    req = urllib2.Request(url.encode(UTF8), udata, headers)
    try:
        response = urllib2.urlopen(req)
        page = response.read()
        if response.info().getheader('Content-Encoding') == 'gzip':
            page = zlib.decompress(page, zlib.MAX_WBITS + 16)
        response.close()
    except Exception:
        page = ""
        xbmc.log(msg='REQUEST ERROR', level=xbmc.LOGDEBUG)
    return(page)
{% endcodeblock %}

The hardest part of building this addon was finding video links. I was able to find a [github repo](https://github.com/learningit/Kodi-plugins-source/) with code for identifying links to PBS's videos, but PBS initially posts their videos on youtube. I watch PBS NewsHour the morning after it airs, so I needed a way to watch these youtube links. I started this post by saying I wanted to avoid using Kodi's youtube addon, but I punted and decided to use the youtube addon to play these links. Below is a function for finding the youtube id of a video.

{% codeblock lang:python %}
def deal_with_youtube(html):
    vid_num = re.compile('<span class="youtubeid">(.+?)</span>',
                         re.DOTALL).search(html)
    url = vid_num.group(1)
    return url
{% endcodeblock %}

This next function actually fetches the videos (the hard part of building this addon). This function fetches the html of the website that has PBS's video. It then searches the html for "coveplayerid," which is PBS's name for the video. I use this name to create a url that will play the video. I get the html associated with this new url, and search it for a json file that contains the video. I grab this json file, and viola I have the video's url! In the final part of the code, I request a higher version of the video than PBS would give me by default.

If I fail to find "coveplayerid," then I know this is a video with a youtube link, so I grab the youtube id. Some pages have a coveplayerid class, but no actual coveplayerid. I also detect these cases and find the youtube id when it occurs.

{% codeblock lang:python %}
# https://github.com/learningit/Kodi-plugins-source/blob/master/plugin.video.thinktv/resources/lib/scraper.py
# modified from link above
def getAddonVideo(url, udata=None, headers=httpHeaders):
    html = getRequest(url)

    vid_num = re.compile('<span class="coveplayerid">(.+?)</span>',
                         re.DOTALL).search(html)
    if vid_num:
        vid_num = vid_num.group(1)
        if 'youtube' in vid_num:
            return deal_with_youtube(html)
        pg = getRequest('http://player.pbs.org/viralplayer/%s/' % (vid_num))
        query = """PBS.videoData =.+?recommended_encoding.+?'url'.+?'(.+?)'"""
        urls = re.compile(query, re.DOTALL).search(pg)

        url = urls.groups()
        pg = getRequest('%s?format=json' % url)
        url = json.loads(pg)['url']
    else:  # weekend links are initially posted as youtube vids
        deal_with_youtube(html)

    url = url.replace('800k', '2500k')
    if 'hd-1080p' in url:
        url = url.split('-hls-', 1)[0]
        url = url+'-hls-6500k.m3u8'
    return url
{% endcodeblock %}

This next function identifies full episodes that have aired in the past week. It's the meat of the addon. The function gets the html of [PBS NewsHour's page](http://www.pbs.org/newshour/videos/), and finds all links in a side-bar where PBS lists their past week's episodes. I loop through the links and create a menu item for each one. These menu items are python objects that Kodi can display to users. The items include a label/title (the name of the episode), an image, and a url that Kodi can use to find the video url.

The most important part of this listing is the url I create. This url gives Kodi all the information I just described, associates the link with an addon, and tells Kodi that the link is playable. In the final part of the function, I pass the list of links to Kodi.

{% codeblock lang:python %}
# -------------- Create list of videos --------------------
# http://kodi.wiki/view/HOW-TO:Video_addon
def list_videos(url='http://www.pbs.org/newshour/videos/'):
    html = getRequest(url)

    query = """<div class='sw-pic maxwidth'>.+?href='(.+?)'.+?src="(.+?)".+?title="(.+?)" """
    videos = re.compile(query, re.DOTALL).findall(html)

    listing = []
    for vids in videos:
        list_item = xbmcgui.ListItem(label=vids[2],
                                     thumbnailImage=vids[1])
        list_item.setInfo('video', {'title': vids[2]})
        list_item.setProperty('IsPlayable', 'true')

        url = ("%s?action=%s&title=%s&url=%s&thumbnail=%s"
               % (base_url, 'play', vids[2], vids[0], vids[1]))

        listing.append((url, list_item, False))

    # Add list to Kodi.
    xbmcplugin.addDirectoryItems(addon_handle, listing, len(listing))
    xbmcplugin.endOfDirectory(handle=addon_handle, succeeded=True)
{% endcodeblock %}

Okay, thats the hard part. The rest of the code implements the functions I just described. The function below is executed when a user chooses to play a video. It gets the url of the video, and gives this to the xbmc function that will play the video. The only hiccup here is I check whether the link is for the standard PBS video type or not. If it is, then I give the link directly to Kodi. If it's not, then this is a youtube link and I launch the youtube plugin with my youtube video id.

{% codeblock lang:python %}
def play_video(path):
    path = getAddonVideo(path)
    if '00k' in path:
        play_item = xbmcgui.ListItem(path=path)
        xbmcplugin.setResolvedUrl(addon_handle, True, listitem=play_item)
    else:  # deal with youtube links
        path = 'plugin://plugin.video.youtube/?action=play_video&videoid=' + path
        play_item = xbmcgui.ListItem(path=path)
        xbmcplugin.setResolvedUrl(addon_handle, True, listitem=play_item)
{% endcodeblock %}

This final function is launched whenever a user calls the addon or executes an action in the addon (thats why I call the function in the final line of code here). params is an empty dictionary if the addon is being opened. params being empty causes the addon to call list_videos, creating the list of episodes that PBS has aired in the past week. If the user selects one of the episodes, then router is called again, but this time the argument is the url of the selected item. This url is passed to the play_video function, which plays the video for the user!

{% codeblock lang:python %}
def router():
    params = dict(args)

    if params:
        if params['action'][0] == 'play':
            play_video(params['url'][0])
        else:
            raise ValueError('Invalid paramstring: {0}!'.format(params))
    else:
        list_videos()


router()
{% endcodeblock %}

That's my addon! I hope this tutorial helps people create future Kodi addons. Definitely reach out if you have questions. Also, make sure to check out the NewsHour soon and often. It's the bomb.
