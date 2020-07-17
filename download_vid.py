from pytube import YouTube

vid = YouTube('https://www.youtube.com/watch?v=NmM_5M6kU0k')
print(vid.streams.filter(resolution='1080p'))
vid.streams.get_highest_resolution().download()