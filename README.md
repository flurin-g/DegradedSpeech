# Setup
On linux you have to install `libsndfile` for `pysoundfile` to work:  
`sudo apt-get install libsndfile1`

# Impulse Responses
- The impulse responses were taken from EchoThies: [EchoThief](http://www.echothief.com/downloads/)
- To create the impulse response meta data csv, enter:  
` python main.py task=create_ir_meta`