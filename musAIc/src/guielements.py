import tkinter as tk
from tkinter import font

#              BG       Text
COLOURS = [('#ff0011', 'white'),
           ('yellow',  'black'),
           ('#22ff22', 'white'),
           ('#1100ff', 'white'),
           ('#eeeeee', 'black'),
           ('#222222', 'white')]

PAUSED = 2
PLAY_WAIT = -1
PAUSE_WAIT = -2
PLAYING = 1

class VScrollFrame(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.canvas = tk.Canvas(root, borderwidth=0)
        self.frame = tk.Frame(self.canvas)
        self.vsb = tk.Scrollbar(root, orient='vertical',
                                command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.update()
        self.canvas.create_window((40, 40), window=self.frame, anchor='nw',
                                 tags='self.frame',
                                  width=self.canvas.winfo_width())
        #print(self.canvas.winfo_width())
        #self.frame.pack(fill='x')
        self.frame.bind('<Configure>', self.onFrameConfigure)

    def onFrameConfigure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        self.canvas.configure(height=self.frame.winfo_reqheight(), 
                              width=self.frame.winfo_reqwidth())

class SelectionGrid(tk.Frame):
    def __init__(self, master, variable, rows, columns, labels, func,
                 colour='yellow', **options):
        tk.Frame.__init__(self, master, **options)
        self.variable = variable
        self.variable.set(0)
        self.variable.trace('w', self.update)
        self.rows = rows
        self.columns = columns
        self.labels = labels
        self.func = func
        self.colour = colour
        self.buttons = {}

        self.selection = None

        self.fontLabel = font.Font(family=tk.font.nametofont('TkDefaultFont').cget('family'),
                     size=8)

        for i in range(rows):
            for j in range(columns):
                idx = i * columns + j
                label = tk.Label(self, text=self.labels[idx], bd=1,
                                 relief='solid', font=self.fontLabel)
                label.grid(row=i, column=j, sticky='nesw', padx=1, pady=1)
                label.bind('<Button-1>', self.clicked)
                self.buttons[int(self.labels[idx])] = label


    def clicked(self, event):
        widget = event.widget

        if self.selection == widget:
            self.set(0)
        else:
            self.set(widget.cget('text'))

    def set(self, val, call=True):
        self.variable.set(val)

        if call:
            self.func(self.variable)

    def update(self, *args):
        # called when the variable is .set()
        val = int(self.variable.get())
        bg_col = self.cget('bg')

        for b in self.buttons.values():
            b['bg'] = bg_col
        self.selection = None

        if val > 0:
            try:
                b = self.buttons[val]
                b['bg'] = self.colour
                self.selection = b
            except:
                # none valid setting
                pass


#class BarCanvas(tk.Canvas):
#    def __init__(self, root, *args, **kwargs):
#        tk.Canvas.__init__(self, root, *args, **kwargs)
#        self.hsb = tk.Scrollbar(root, orient='horizontal', command=self.xview)
#        self.hsb.pack(side='bottom', fill='x')
#        self.configure(xscrollcommand=self.hsb.set)
#
#        self.bind('<Configure>', self.onCanvasConfigure)
#
#    def onCanvasConfigure(self, event):
#        self.configure(scrollregion=self.bbox('all'))
#        self.configure(width=self.winfo_width())


class InstrumentPanel(tk.Frame):
    def __init__(self, master, instrument, name='Instrument', **options):
        tk.Frame.__init__(self, master, **options)
        self.configure(padx=2, pady=2)
        self.instrument = instrument
        self.ins_manager = instrument.ins_manager
        self.name = tk.StringVar()
        self.name.set(name + ' ({})'.format(instrument.ins_id))
        self.chan = tk.IntVar(self)
        self.chan.set(instrument.chan)
        self.colour = COLOURS[self.instrument.ins_id % len(COLOURS)]

        self.controlFrame = tk.Frame(self)
        self.controlFrame.grid(row=0, column=0, sticky='ns')

        self.colourStrip = tk.Frame(self.controlFrame, width=6,
                                    bg=self.colour[0])
        self.nameLabel = tk.Label(self.controlFrame, textvariable=self.name,
                                  bg=self.colour[0], fg=self.colour[1], anchor='w')
        self.nameLabel.bind('<Button-1>', lambda event: self.editEntry(event,
                                                                       self.nameUpdate))

        self.removeButton = tk.Label(self.controlFrame, text='x',
                                     bg=self.colour[0], fg=self.colour[1], anchor='w')
        self.removeButton.bind('<Button-1>', self.remove)

        self.continuousVar = tk.IntVar(self.controlFrame)
        self.continuousVar.set(1)
        self.continuousButton = tk.Checkbutton(self.controlFrame,
                                               var=self.continuousVar,
                                               command=self.continuous)

        self.confidence = tk.IntVar(self.controlFrame)
        self.confidenceOption = tk.OptionMenu(self.controlFrame,
                                              self.confidence, 0, 1, 2, 3, 4,
                                              command=self.confidenceUpdate)
        self.transposeVar = tk.IntVar(self.controlFrame)
        self.transposeVar.set(0)
        self.transpose = tk.OptionMenu(self.controlFrame, self.transposeVar,
                                       -2, -1, 0, 1, 2,
                                       command=self.transUpdate)
        self.chanLabel = tk.Label(self.controlFrame, textvariable=self.chan,
                                  bg=self.colour[0], fg=self.colour[1],
                                  anchor='e')
        self.chanLabel.bind('<Button-1>', lambda event: self.editEntry(event,
                                                                  self.chanUpdate))

        self.repeatVar = tk.StringVar(self.controlFrame)
        self.repeatSelect = SelectionGrid(self.controlFrame, self.repeatVar, 1,
                                          4, [1, 2, 4, 8], self.loopUpdate)

        self.recVar = tk.StringVar(self.controlFrame)
        self.recSelect = SelectionGrid(self.controlFrame, self.recVar, 1, 4, 
                                       [1, 2, 4, 8], self.recUpdate,
                                       colour='#cc1010')

        self.pauseButton = tk.Button(self.controlFrame, text='Pause')
        self.pauseButton['command'] = lambda: self.toggle_playback(self.pauseButton)
        self.muteButton = tk.Button(self.controlFrame, text='Mute', command=self.mute)

        self.update()
        self.canvasHeight = 80
        self.barCanvas = tk.Canvas(self, width=self.winfo_width(),
                                   height=self.canvasHeight, bg='#303030')
        self.barCanvas.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        self.barCursor = self.barCanvas.create_line(1, 0, 1, self.canvasHeight,
                                                    fill='orange', width=2)

        # pack all the elements...
        self.colourStrip.grid(row=0, column=0, rowspan=4, sticky='ns')
        self.removeButton.grid(row=0, column=1, sticky='ew')
        self.nameLabel.grid(row=0, column=2, columnspan=1, sticky='ew')
        self.confidenceOption.grid(row=1, column=1)
        self.transpose.grid(row=1, column=2)
        self.continuousButton.grid(row=1, column=3)
        self.chanLabel.grid(row=0, column=3, sticky='ew')
        self.repeatSelect.grid(row=2, column=1, )
        self.recSelect.grid(row=3, column=1)
        self.pauseButton.grid(row=2, column=2, rowspan=2)
        self.muteButton.grid(row=2, column=3, rowspan=2)

        # initialise bar display...
        self.update_display(0)
        self.bind('<Configure>', self.onConfigure)

    def onConfigure(self, event):
        self.barCanvas.configure(width=self.winfo_width())

    def update_display(self, beat):
        # pretty inefficient for now
        self.barCanvas.delete('all')
        self.beat_width = 25
        noteRange = (36, 85)   # +- one octave from middle C
        scale = self.canvasHeight / (noteRange[0] - noteRange[1])
        cb = self.instrument.bar_num - 1
        bars = self.instrument.bars[max(0, cb-2): min(len(self.instrument.bars), cb+7)]
        bar_nums = range(max(0, cb-2), min(len(self.instrument.bars), cb+7))

        if self.instrument.status == PAUSED or self.instrument.status == PLAY_WAIT:
            beat = 0.0
            cb += 1

        # draw bars
        for i, bar in zip(list(bar_nums), bars):
            offset = (2 + i - cb - beat/4) * self.beat_width * 4

            if self.instrument.loopLevel > 0:
                loop_end = self.instrument.loopEnd
                loop_start = loop_end - self.instrument.loopLevel

                if i >= loop_start and i < loop_end:
                    self.barCanvas.create_rectangle(offset, 0,
                                                    offset+(4*self.beat_width),
                                                    self.canvasHeight,
                                                    fill='#525100')

            if i in self.instrument.record_bars:
                self.barCanvas.create_rectangle(offset, 0,
                                                offset+(4*self.beat_width),
                                                self.canvasHeight,
                                                fill='#800000')


            self.barCanvas.create_line(offset, 0, offset, 100, fill='#aaaaaa')
            self.barCanvas.create_text(offset+5, 5, text=i+1, fill='#aaaaaa')

            noteOn_times = sorted(list(bar.keys())) + [4]
            for j, t in enumerate(noteOn_times[:-1]):
                x = offset + t * self.beat_width
                l = (noteOn_times[j+1] - t) * self.beat_width
                y = scale * (bar[t] - noteRange[1])
                self.barCanvas.create_line(x, y, x+l, y, fill='#cccccc',
                                           width=2)

        # draw cursor
        cursor = 8*self.beat_width
        self.barCanvas.create_line(cursor, 0, cursor,
                                   self.canvasHeight, fill='orange', width=2)
        self.barCanvas.create_polygon(cursor, 5, cursor+5, 0, cursor-5, 0,
                                      fill='orange')
        self.barCanvas.create_polygon(cursor, self.canvasHeight-4, cursor+5,
                                      self.canvasHeight+1, cursor-5,
                                      self.canvasHeight+1, fill='orange')

        # make sure controls are up-to-date...
        if self.instrument.status == PAUSED:
            self.pauseButton['text'] = 'Play'
            self.pauseButton['fg'] = 'black'
        elif self.instrument.status == PAUSE_WAIT:
            self.pauseButton['text'] = 'Pausing'
            self.pauseButton['fg'] = 'orange'
        elif self.instrument.status == PLAYING:
            self.pauseButton['text'] = 'Pause'
            self.pauseButton['fg'] = 'black'
        elif self.instrument.status == PLAY_WAIT:
            self.pauseButton['text'] = 'Playing'
            self.pauseButton['fg'] = 'orange'

        if self.instrument.active:
            self.configure(highlightbackground='black', highlightthickness=1)
        else:
            self.configure(highlightbackground='grey', highlightthickness=0)



    def editEntry(self, event, func):
        widget = event.widget
        entry_widget = tk.Entry(widget)
        entry_widget.delete(0, 'end')
        entry_widget.insert(0, widget['text'])
        entry_widget.place(x=0, y=0, anchor='nw', relwidth=1, relheight=1)
        entry_widget.bind('<Return>', func)
        entry_widget.bind('<FocusOut>', func)
        entry_widget.focus_set()

    def nameUpdate(self, event):
        entry = event.widget
        self.name.set(entry.get())
        entry.destroy()

    def chanUpdate(self, event):
        entry = event.widget
        try:
            new_chan = max(1, min(16, int(entry.get())))
        except ValueError:
            entry.destroy()
            return

        self.chan.set(new_chan)
        self.instrument.chan = new_chan
        entry.destroy()

    def transUpdate(self, event):
        self.instrument.transpose = self.transposeVar.get()

    def confidenceUpdate(self, event):
        self.instrument.confidence = self.confidence.get()

    def loopUpdate(self, variable):
        self.instrument.loopLevel = int(variable.get())
        self.instrument.loopEnd = self.instrument.bar_num

    def recUpdate(self, variable):
        num = int(variable.get())
        self.ins_manager.set_recording_instrument(self.instrument, num)

    def remove(self, event):
        self.instrument.delete()

    def toggle_playback(self, button):
        self.instrument.toggle_paused()

    def mute(self):
        self.instrument.toggle_mute()

    def continuous(self):
        self.instrument.toggle_continuous()

