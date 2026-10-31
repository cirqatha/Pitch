# Pitch

hand tracking using only your laptop's speakers and mic. no camera, no extra hardware, nothing.

WATCH THIS VIDEO FIRST (its 2 min, it explains it way better than this readme does)

https://github.com/user-attachments/assets/4dd9da26-d69c-4381-b658-fe73e5ab960f

so basically the speakers play quiet tones pitched right at the edge of what you can hear (hence the name lol), the mic picks up the echo off your hand, and from how long that echo takes to come back from each speaker it figures out where your hand is. the laptop is basically a bat now. it shows everything as a top down map with your hand as a dot

could be used for touchless stuff, like skipping songs while cooking with messy hands or flipping slides. also no camera so its not watching you, and it works in the dark

## running it

needs python 3.10 - 3.13 (3.14 might break numpy, didnt test)

windows: just double click run.bat. first time it installs the packages, after that it just starts

anything else:

pip install -r requirements.txt
python -m pitch

linux also needs libportaudio2. ive mostly tested on windows tho

## how to use
unplug headphones, volume around 50-80%
keep your hands away while it calibrates (~5 sec). it sweeps the speakers to see what your laptop can actually play and hear, so it adapts to whatever laptop its on
when it asks, hold your palm ~20 cm in front of the middle of the screen. (space skips this)
move your hand around!!

works best 12-50 cm from the screen. it tracks movement, so if you hold your hand totally still it fades out after a couple seconds. also keep your other hand off the touchpad, it WILL track that too

red trail = moving toward the laptop, blue = moving away

keys:

R - recalibrate
N - redo the palm thing
D - draw in the air, C clears it
E / I / V - ellipses / sonogram / evidence map on/off
M - mic settings (windows)
+ / - - louder / quieter
S - screenshot
F - fullscreen
Esc - quit
IMPORTANT: mic level

lots of laptops (gaming ones especially) have the mic turned way up, which makes the echoes really weak. if it says the probe level was lowered cause the mic is too sensitive -> press M, set the mic volume to ~20%, press R. this made a HUGE difference on my laptop

if your speakers cant go very high you might hear a faint hiss. run.bat --quiet gets rid of it but tracking gets a bit blurrier

## how it works

left and right speakers play different sets of tones so the mic can tell them apart. dividing what the mic hears by what was played gives an echo profile for each speaker (how much sound comes back from each distance)

anything that doesnt move (desk, walls, the laptop itself) gets learned and subtracted. a moving hand shifts the echo's phase (doppler), which is how it tells your hand apart from your body just breathing and swaying around. each echo puts the hand somewhere on an ellipse around a speaker and the mic, and where the left and right ellipses cross = your hand

single frames are super noisy, so evidence builds up over time along paths a hand could actually move. and the dot has to follow physics: no teleporting, max 1.5 m/s, and the nearest moving thing wins (your hand is in front of your arm and body so it sticks to the hand)

## limits
fast movements make it lag a bit
cant see closer than ~12 cm, gets weaker past ~45 cm
hand on the touchpad/keyboard gets tracked like any other hand
## if something goes wrong
- "Python not found" - install python 3.12 from python.org and tick "Add python.exe to PATH"
- "Couldn't open the laptop's audio" - close stuff using the mic (zoom, discord etc). then try python -m pitch --list-devices and pick yours with --in N --out M
- "The mic is only giving silence" - Settings > Privacy & security > Microphone, allow desktop apps. also check its not muted
- "The mic can't hear the speakers" / "speakers are too quiet" / "couldn't find a usable band" - unmute, volume up, unplug headphones, turn off audio enhancements / dolby / nahimic, set the mic to 48 kHz in sound settings, then R
- "The speakers or the mic are overloaded" - volume down to 30-50%, turn off mic boost, R
- "Left and right sound the same" - turn off Mono audio (Settings > Accessibility > Audio). otherwise it only tracks distance
- jumpy tracking / "something moved during calibration" - R with your hands away, sit still, then N for the palm step
