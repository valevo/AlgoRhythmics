s.boot;

NetAddr.langPort;
b = NetAddr.new("127.0.0.1", 7101);

~notes = Array.newClear(128)
~synth = nil

(
OSCdef.new(
	\setupSynth,
	{
		~synth.free;
		~synth = Synth.new(
			\simple, [\gate, 0]
		)
	},
	'/setup_synth');

OSCdef.new(
	\playNote,
	{
		arg msg;
		~synth.set(\nn, msg[1], \gate, 1);
	},
	'/play_note');

OSCdef.new(
	\endNote,
	{
		~synth.set(\gate, 0);
	},
	'/end_note');

OSCdef.new(
	\setTrem,
	{
		arg msg;
		~synth.set(\freq1, msg[1]);
	},
	'/set_trem');


)


(
SynthDef.new(\simple, {
	arg nn=60, amp=1, gate=0, freq1=5, depth=0.2;
	var sig, env, trem, warmth;
	sig = LFSaw.ar(nn.midicps)!2;
	env = EnvGen.kr(Env.adsr, gate, doneAction:2);
	trem = SinOsc.kr(freq1, 0, depth*0.5, 1-depth*0.5);
	sig = sig*env*amp*trem;
	Out.ar(0, sig);
}).add;
)

(
SynthDef.new(\synth, {
	arg nn=60, nharm=20, amp=1, gate=0;
	var sig, env;
	env = EnvGen.kr(Env.asr(0.05, 0.5, 1), gate);
	sig = Blip.ar(
		nn.midicps *
		LFNoise1.kr(2!16),
		nharm
	);
	sig = sig * LFNoise1.kr(0.5!16).exprange(0.1, 0.5);
	sig = Splay.ar(sig);
	sig = sig*env*amp;
	Out.ar(0, sig);
}).add;
)

(
SynthDef.new(\bigsynth, {
	arg nn=60, gate=0;
	var temp, sum, env;
	sum = 0;
	env = EnvGen.kr(Env.asr(0.01, 3, 1), gate);
	10.do{
		temp = VarSaw.ar(
			nn.midicps * {Rand(0.99, 1.02)}!2,
			{Rand(0.0, 1.0)}!2,
			{ExpRand(0.005, 0.05)}!2
		);
		sum = sum+temp;
	};
	sum = sum*0.02*env;
	Out.ar(0, sum);
}).add;
)


(
// kill synths
~synth.free;
s.freeAll
)


x = Synth.new(\simple, [\gate, 1, \freq1, 2, \depth, 0]);
x.set(\gate, 0);
x.free


// GUI ---------------------
(
var w, numbox;

w = Window('My Window', Rect(100, 300, 200, 200));

numbox = NumberBox(w, Rect(50, 10, 50, 20));
numbox.value = 2;
numbox.action = {arg num; b.sendMsg('/spread', num.value);};

w.front;

)






