# Pitch
Hand tracking using only your laptop's speakers and mic. No camera, no extra hardware.

WATCH THIS VIDEO BELOW

https://github.com/user-attachments/assets/4dd9da26-d69c-4381-b658-fe73e5ab960f

The speakers play quiet tones pitched right at the edge of what you can hear (hence the name), the mic picks up the echo off your hand, and from how long that echo takes to come back from each speaker it works out where your hand is. Basically it turns the laptop into a bat. It shows everything as a top-down map with your hand as a dot.

Could be used for touchless stuff like skipping songs while cooking with messy hands or flipping slides, and since there's no camera it isn't watching you and works in the dark.

## Running it

Needs Python 3.10 - 3.13.

On Windows just double click run.bat. It installs what it needs the first time and starts. Anywhere else:

## How to use it
1. Unplug headphones, volume around 50-80%.
2. Keep your hands away while it calibrates (~5 s). It sweeps the speakers to see what your laptop can actually play and hear, so it adapts to whatever laptop it's on.
3. Hold your palm about 20 cm in front of the middle of the screen when it asks. Space skips this.
4. Move your hand around.

Works best 12-50 cm from the screen. It tracks movement, so a hand held totally still fades out after a couple of seconds. Keep your other hand off the touchpad, it'll track that too.

Red trail = moving toward the laptop, blue = moving away.

Keys: R recalibrate, N redo the palm step, D draw in the air, C clear, E / I / V toggle ellipses / sonogram / evidence map, M mic settings, probe volume, S screenshot, F fullscreen, Esc quit.

lots of laptops have the mic turned way up, which makes the echoes really weak. If it says the probe level was lowered because the mic is too sensitive, press M, set the mic volume to ~20% and press R. Made a huge difference on my laptop.


## How it works

Left and right speakers play different sets of tones, so the mic can tell them apart. Dividing what the mic hears by what was played gives an echo profile for each speaker, i.e. how much sound comes back from each distance.

Everything that doesn't move (desk, walls, the laptop) gets learned and subtracted. A moving hand shifts the echo's phase (Doppler), which separates it from your body just breathing and swaying. Each echo puts the hand somewhere on an ellipse around a speaker and the mic, and where the left and right ellipses cross is the hand.

Single frames are noisy, so evidence builds up over time along paths a hand could actually move, and the dot has to follow physics: no teleporting, max 1.5 m/s, and the nearest moving thing wins (your hand is in front of your arm and body).

## If something goes wrong

- "Python not found": install Python 3.12 from python.org, tick "Add python.exe to PATH".
- "Couldn't open the laptop's audio": close apps using the mic (Zoom, Discord), then try python -m pitch --list-devices and --in N --out M.
- "The mic is only giving silence": Settings > Privacy & security > Microphone, allow desktop apps, and check it isn't muted.
- "The mic can't hear the speakers" / "speakers are too quiet" / "couldn't find a usable band": unmute, turn the volume up, unplug headphones, turn off audio enhancements / Dolby / Nahimic, set the mic to 48 kHz in Sound settings, then press R.
- "The speakers or the mic are overloaded": volume down to 30-50%, turn off mic boost, press R.
- "Left and right sound the same": turn off Mono audio (Settings > Accessibility > Audio). Otherwise it only tracks distance.
- Jumpy tracking or "something moved during calibration": press R with your hands away and sit still, then N to redo the palm step.
