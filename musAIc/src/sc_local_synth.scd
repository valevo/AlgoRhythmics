s.boot;

// test tone...
{ SinOsc.ar(440, 0, 0.2) }.play

// get local address and port
NetAddr.localAddr;
b = NetAddr.new("127.0.0.1", 7121)

// piano synth
(
SynthDef("mdapiano", { |out=0, freq=440, gate=1|
	var son = MdaPiano.ar(freq, gate, release:0.9, stereo:0.3, sustain:0);
	DetectSilence.ar(son, 0.01, doneAction:2);
	Out.ar(out, son*0.1);
}).add;
)

x = Synth("mdapiano", [\freq, 490]);

x.set(\freq, 460);


// handle OSC messages...
(
var notes;
notes =  Array.newClear(128);

OSCdef.new(
	\noteOn,
	{
		arg msg;
		notes[msg[2]] = Synth("mdapiano", [\freq, msg[2].midicps])
	};,
	'/noteOn');

OSCdef.new(
	\noteOff,
	{
		arg msg;
		notes[msg[2]].release;
	};,
	'/noteOff');

OSCdef.new(
	\allOff,
	{
		arg msg;
		// msg.postln;
		notes.do({arg i; i.free;});
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

)












(
TempoClock.default.tempo = 1.2;
Ppar([
    Pbind(
        \instrument, \mdapiano,
        \degree,
            Pseq([
                0, 7, -5, 7,  0, 5, -7, 5,  -2, 5, -7, 5,  -2,  3, -9,  3,
                0, 7, -5, 7,  0, 5, -7, 5,  -2, 5, -7, 5,  -2, -3, -4, -5 ], inf),
        \dur, 0.5,
        \octave, 3,
        \root, 3,
        \vel, Prand([Pseq([100, 30, 50, 10]), Pseq([100, 30, 10, 10, 5, 10, 20, 30])], inf),
        \legato, 0.95
    ),
    Pbind(
        \instrument, \mdapiano,
        \degree,
            Pseq([ \, 0, -1, 0, 2, 0, \, \, \, 0, -2, \, \, -2, -4, \, \ ], inf),
        \dur, 0.5,
        \octave, 6,
        \root, 3,
        \vel, Pwhite(50, 100, inf)
    )
], inf).play
)

