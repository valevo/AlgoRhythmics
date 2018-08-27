// Converts OSC messages to MIDI output for Ableton

s.boot;




MIDIClient.init;
// MIDIClient.disposeClient;
MIDIClient.externalSources
MIDIIn.connectAll;
MIDIClient.sources;


// MIDIClient.destinations;
// MIDIIn.connect(device:0);
// MIDIIn.disconnectAll;


~mOut = MIDIOut.newByName("LoopBe Internal MIDI", "LoopBe Internal MIDI").latency_(Server.default.latency);

~mOut.allNotesOff(1);

NetAddr.localAddr;
b = NetAddr.new("192.168.56.103", 7101)

(
OSCdef.new(
	\noteOn,
	{
		arg msg;
		//msg.postln;
		~mOut.noteOn(msg[1]-1, note: msg[2])
	};,
	'/noteOn');

OSCdef.new(
	\noteOff,
	{
		arg msg;
		~mOut.noteOff(msg[1]-1, note: msg[2])
	};,
	'/noteOff');

OSCdef.new(
	\allOff,
	{
		arg msg;
		16.do({arg i; ~mOut.allNotesOff(i);});
	};,
	'/allOff');

OSCdef.new(
	\ping,
	{
		arg msg;
		b.sendMsg("/pingReply", 1);
	};,
	'/ping'
);

OSCdef.new(
	\midiCC,
	{
		arg msg;
		msg.postln;
		~mOut.control(chan:1, ctlNum:msg[1], val:msg[2])
	};,
	'/midiCC'
)
)

(
MIDIdef.noteOn(\midiNoteOn, {
	arg vel, nn, chan, src;
	~mOut.noteOn(1, note: nn);
	b.sendMsg("/noteOn", nn);
}, srcID: 1);   // need to add source ID (1)

MIDIdef.noteOff(\noteOff,
	{
		arg vel, nn;
		~mOut.noteOff(1, note: nn);
		b.sendMsg("/noteOff", nn);
	}, srcID: 1);
)

(
MIDIdef.cc(\ccMsg,
	{
		arg val, nn, chan, src;
		//nn.postln;
		b.sendMsg("/cc", val, nn);
}, srcID: 1);
)
)

b.sendMsg("/cc", 1, 127);
~mOut.control(chan:1, ctlNum:1, val:0)
~mOut.control(chan:1, ctlNum:2, val:0)