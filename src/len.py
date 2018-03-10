import subprocess
import re

process = subprocess.Popen(['ffmpeg',  '-i', 'loop.wav'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
stdout, stderr = process.communicate()
matches = re.search(r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout, re.DOTALL).groupdict()

print matches['hours']
print matches['minutes']
print matches['seconds']
