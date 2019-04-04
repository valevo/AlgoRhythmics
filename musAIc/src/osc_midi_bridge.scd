// Converts OSC messages to MIDI output for Ableton

s.boot;

OSCFunc.trace

(
f = { |msg, time, addr|
    if(msg[0] != '/status.reply') {
        "time: % sender: %\nmessage: %\n".postf(time, addr, msg);
    }
};
thisProcess.addOSCRecvFunc(f);
);

// stop posting.
thisProcess.removeOSCRecvFunc(f);


(
MIDIClient.init;
// MIDIIn.connectAll;

// ~mOut = MIDIOut.newByName("LoopBe Internal MIDI", "LoopBe Internal MIDI").latency_(Server.default.latency);

// ~mOut = MIDIOut.newByName("MiniBrute", "MiniBrute").latency_(Server.default.latency);
~mOut = MIDIOut.newByName("Komplete Audio 6 MIDI", "Komplete Audio 6 MIDI").latency_(Server.default.latency);

~mOut.allNotesOff(1);

NetAddr.localAddr;
// ip address and port of musAIc...
// get from running ifconfig on guest machine, port is arbitrary
b = NetAddr.new("192.168.56.102", 7121)
// b = NetAddr.new("10.0.2.15", 7121)
)

(
OSCdef.new(
	\noteOn,
	{
		arg msg;
		msg.postln;
		~mOut.noteOn(msg[1]-1, note: msg[2], veloc: msg[3])
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
		~mOut.allNotesOff(msg[1]-1);
	};,
	'/allOff');

OSCdef.new(
	\clockStart,
	{
		arg msg;
		~mOut.start;
	};,
	'/clockStart');

OSCdef.new(
	\clockStop,
	{
		arg msg;
		~mOut.stop;
1
	};,
	'/clockStop');

OSCdef.new(
	\clock,
	{
		arg msg;
		~mOut.midiClock;
		// "clock".postln;
	};,
	'/clock');

OSCdef.new(
	\panic,
	{
		arg msg;
		16.do({arg i; ~mOut.allNotesOff(i);});
	};,
	'/panic');

OSCdef.new(
	\ping,
	{
		arg msg;
		// msg.postln;
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
);

MIDIdef.noteOn(\midiNoteOn, {
	arg vel, nn, chan, src;
	// ~mOut.noteOn(1, note: nn);
	b.sendMsg("/noteOn", nn);
	nn.postln;
}, srcID: 2);   // need to add source ID (1)

MIDIdef.noteOff(\noteOff,
	{
		arg vel, nn;
		~mOut.noteOff(1, note: nn);
		b.sendMsg("/noteOff", nn);
	}, srcID: 2);

MIDIdef.cc(\ccMsg,
	{
		arg val, nn, chan, src;
		[nn, val].postln;
		b.sendMsg("/cc", val, nn);
}, srcID: 2);
)


"a;lsdkjf".postln


~mOut.start;

~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;
~mOut.midiClock;


~mOut.stop


b = NetAddr.new("192.168.56.102", 7121)
b = NetAddr.new("10.114.69.242", 8000)

b = NetAddr.new("10.114.69.242", 8000)

b = NetAddr.new("localhost", 57120)

b.sendMsg("/cc", 1, 9);
b.sendMsg("/bpm", 100);
b.sendMsg("/hello", "there!")

b.sendMsg("/metronome", 1, 1, 0, 0)

t = 60
n = NetAddr("100.75.0.230", 57120);
(
OSCdef.new(
	\bpm,
	{
		|msg, time, addr|
		msg.postln;
		// ~mOut.control(chan:1, ctlNum:msg[1], val:msg[2])
	};,
	'/tempo',
	n
);
)



~mOut.control(chan:1, ctlNum:1, val:0)
~mOut.control(chan:1, ctlNum:2, val:0)



(
MIDIdef.noteOn(\midiNoteOn, {
	arg vel, nn, chan, src;
	// ~mOut.noteOn(1, note: nn);
	// b.sendMsg("/noteOn", nn);
	nn.postln;
}, srcID: 2);
)