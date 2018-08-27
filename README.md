# AlgoRhythmics

## Requirements:
* python 3 (at least 3.4)
* TensorFlow (requires `numpy`, `dev`, `pip`, `wheel`)
* Keras 
* Music21 -> interface for MIDI in python, including parsing, conversion, and automatic feature detection
* MuseScore -> allows display of scores from Music21
* SuperCollider (to pass and convert MIDI and OSC messages)
* Ableton Live (to use the MIDI notes generated live)

## Setup

Incomplete and unordered list of steps to set up the environment.

### MuseScore

One needs to manually set the path to MuseScore for Music21.

First, create a file that permanently stores the user settings.

`from music21 import environment`  
`us = environment.UserSettings()`  
`us.create()`  
(`us.getSettingsPath()` holds the location of the created user settings file.)


Then, set the path, replacing `my_path` with your local path to MuseScore (Unix shell `which musescore`).
`us["musescoreDirectPNGPath"] = "my_path"`

### musAIc

This is the GUI for live performances of the network. Sends out OSC messages that need to be converted to MIDI noteOn messages for Ableton to process, synthesis and record the performance.

TODO: write detailed setup instructions for SuperCollider and Ableton
