import os
import math
import time
import tkinter as tk
from tkinter import font

#              BG       Text
INS_COLOURS = [('#ff0011', 'white'),
           ('yellow',  'black'),
           ('#22ff22', 'black'),
           ('#1100ff', 'white'),
           ('#eeeeee', 'black'),
           ('#222222', 'white')]

COLOR_SCHEME = {
    'panel_bg':         'black',
    'panel_select':     '#808080',
    'light_grey' :      '#aaaaaa',
    'dark_grey' :       '#101010',
    'note_panel_bg':    '#303030',
    'text_light':       '#bbbbbb',
    'text_dark':        '#101010',
}

PAUSED = 2
PLAY_WAIT = -1
PAUSE_WAIT = -2
PLAYING = 1

class VScrollFrame(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        self.grid(row=1, sticky='nesw')
        #self.pack(side='bottom', expand=True, fill='both')
        self.canvas = tk.Canvas(self, background=COLOR_SCHEME['panel_bg'], borderwidth=0)
        self.frame = tk.Frame(self.canvas, background=self.canvas.cget('bg'))
        self.frame.bind('<Configure>', self.onFrameConfigure)
        self.vsb = tk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.vsb.configure(bg=COLOR_SCHEME['panel_bg'], troughcolor=COLOR_SCHEME['dark_grey'])

        self.canvas.configure(width=self.winfo_reqwidth(),
                              height=self.winfo_reqheight(),
                              yscrollcommand=self.vsb.set)

        self.vsb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.update()
        self.canvas.create_window((0, 0), window=self.frame, anchor='sw',
                                 tags='self.frame', width=self.canvas.winfo_width())
        self.canvas.yview_scroll(-1, 'units')

        # mousewheel events...
        self.bind_all('<MouseWheel>', self.onMouseWheel)
        self.bind_all('<Button-4>', self.onMouseWheel)
        self.bind_all('<Button-5>', self.onMouseWheel)

    def onFrameConfigure(self, event):
        bbox = self.canvas.bbox('all')
        #bbox[1] = min(bbox[1], -self.winfo_height())
        self.canvas.configure(scrollregion=bbox)
        self.frame.configure(height=self.frame.winfo_reqheight(), width=self.frame.winfo_reqwidth())

    def onMouseWheel(self, event):
        if event.num == 4 or event.delta == -120:
            self.canvas.yview_scroll(-1, 'units')
        elif event.num == 5 or event.delta == 120:
            self.canvas.yview_scroll(1, 'units')

class Knob(tk.Frame):
    def __init__(self, master, radius, variable, name='', min_=0, max_=1, default=0,
                 **options):

        tk.Frame.__init__(self, master, **options)

        self.radius = radius
        self.variable = variable
        self.min_ = min_
        self.max_ = max_

        self.name = tk.Label(self, text=name, fg=COLOR_SCHEME['text_light'], bg=self.cget('bg'))
        self.name.grid()

        self.canvas = tk.Canvas(self, width=2*radius+2, height=2*radius+2, bd=0,
                                highlightthickness=0, bg=COLOR_SCHEME['panel_bg'] )
        self.canvas.grid(row=1, column=0)

        if default > max_:
            self.val = max_
        elif default < min_:
            self.val = min_
        else:
            self.val = default

        self.label = tk.Label(self, text='{:5.02f}'.format(self.val), fg=COLOR_SCHEME['text_light'], bg=self.cget('bg'))
        self.label.grid(row=2, column=0)

        self.arc = self.canvas.create_arc(1, 1, 2*radius+1, 2*radius+1, style=tk.ARC,
                               outline=COLOR_SCHEME['text_light'], width=1,
                               extent=300, start=-60, tags='knob')

        # sets the sensitivity
        self.range_ = 200/(max_ - min_)

        self.line = self.canvas.create_line(radius+1, radius+1, 2*radius+1,
                                            radius+1, fill=COLOR_SCHEME['text_light'], width=2,
                                            tags='knob')

        self.dragStart = 0
        self.valStart = self.range_*self.val
        self.lastChangeTime = 0

        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.update_line()

    def click(self, event):
        self.dragStart = event.y
        self.valStart = self.range_* self.val

    def drag(self, event):
        offset = event.y - self.dragStart
        self.val = min(self.max_, max(self.min_, (self.valStart - offset)/self.range_))
        self.update_line()

    def release(self, event):
        self.variable.set(self.val)

    def update_percent(self, p):
        old_p = (self.val - self.min_) / (self.max_ - self.min_)
        if abs(p - old_p) < 0.02:
            return

        self.val = self.min_ + p*(self.max_ - self.min_)
        self.update_line()
        self.variable.set(self.val)

    def update_line(self):
        if self.val < self.min_:
            self.val = self.min_
        elif self.val > self.max_:
            self.val = self.max_

        #self.variable.set(self.val)

        p = (self.val - self.min_)/(self.max_ - self.min_)
        a = 4.18879 - p*5.23599

        x = math.cos(a) * self.radius + self.radius
        y = -math.sin(a) * self.radius + self.radius

        self.canvas.coords(self.line, self.radius+1, self.radius+1, x+1, y+1)
        self.label.config(text='{:5.02f}'.format(self.val))


class ModeSelect(tk.Frame):
    def __init__(self, master, variable, name, labels, default=0, colour='orange', **options):
        tk.Frame.__init__(self, master, **options)
        self.variable = variable
        self.variable.trace('w', self.update)
        self.labels = labels
        self.selection = default
        self.colour = colour

        self.fontLabel = font.Font(family=tk.font.nametofont('TkDefaultFont').cget('family'),
                     size=8)

        self.columnconfigure(1, weight=1)
        name = tk.Label(self, text=name, fg=COLOR_SCHEME['text_light'], bg=self.cget('bg'))
        name.grid(row=0, column=0)
        self.label = tk.Label(self, text=self.labels[self.selection], bd=2, relief='solid',
                              fg=COLOR_SCHEME['text_dark'], bg=self.colour)

        self.label.grid(row=0, column=1, sticky='ew')
        self.label.bind('<Button-1>', self.clicked)

        self.variable.set(self.selection)
        self.bind('<Configure>', self.onConfigure)

    def onConfigure(self, event):
        self.configure(width=self.winfo_reqwidth())

    def clicked(self, event):
        self.selection = (self.selection+1) % len(self.labels)
        self.variable.set(self.selection)

    def update(self, *args):
        self.label.config(text=self.labels[self.variable.get()])


#class CheckSelection(tk.Frame):
#    def __init__(self, master, variable, labels, name, colour='yellow', **options):
#        tk.Frame.__init__(self, master, **options)
#        self.variable = variable
#        self.labels = labels

class SimpleButton(tk.Frame):
    def __init__(self, master, variable=None, label='', func=None, colour='orange', fg=COLOR_SCHEME['text_light'], **options):
        tk.Frame.__init__(self, master, **options)
        self.variable = variable
        if variable:
            self.variable.trace('w', self.update)
        self.colour = colour
        self.offColour = master.cget('bg')
        self.func = func

        self.columnconfigure(0, weight=1)
        #self.button = tk.Frame(self, bd=2, relief='solid', bg=self.offColour)
        #self.button.grid(row=0, column=0, sticky='nesw', padx=1, pady=1)

        self.label = tk.Label(self, text=label, fg=fg, bd=2, relief='solid', bg=self.offColour)
        self.label.grid(row=0, column=0, sticky='ew', padx=1, pady=1)
        self.label.bind('<Button-1>', self.clicked)

        self.bind('<Configure>', self.onConfigure)
        self.update()

    def onConfigure(self, event):
        self.label.config(width=self.cget('width'))

    def clicked(self, event):
        if self.variable:
            self.variable.set(not self.variable.get())

        if self.func:
            self.func()

    def update(self, *args):
        if self.variable:
            if self.variable.get():
                self.label.config(fg=COLOR_SCHEME['text_dark'])
                self.label.config(bg=self.colour)

            else:
                self.label.config(fg=COLOR_SCHEME['text_light'])
                self.label.config(bg=self.offColour)

    def configure(self, **kwargs):
        if 'command' in kwargs:
            self.func = kwargs['command']

        if 'text' in kwargs:
            self.label.config(label=kwargs['text'])

        if 'bg' in kwargs:
            self.config(bg=kwargs['bg'])

        if 'fg' in kwargs:
            self.label.config(fg=kwargs['fg'])



class SelectionGrid(tk.Frame):
    def __init__(self, master, variable, rows, columns, labels, func,
                 name, colour='orange', **options):
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

        self.name = tk.Label(self, text=name, fg=COLOR_SCHEME['text_light'], bg=self.cget('bg'))
        self.name.grid(column=0)

        for i in range(rows):
            for j in range(columns):
                idx = i * columns + j
                label = tk.Label(self, text=self.labels[idx], bd=1,
                                 relief='solid', fg=COLOR_SCHEME['text_light'],
                                 bg=self.cget('bg'), font=self.fontLabel)
                label.grid(row=i, column=j+1, sticky='nesw', padx=1, pady=1)
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

        if call and self.func:
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

class PlayerControls(tk.Frame):
    def __init__(self, master, app, engine, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        self.app = app
        self.engine = engine

        size = 25
        self.col_grey = COLOR_SCHEME['light_grey']
        self.dark_grey = COLOR_SCHEME['dark_grey']

        self.play_button = tk.Canvas(self, width=size, height=size,
                                     bg=self.col_grey, highlightthickness=1,
                                     highlightbackground=self.dark_grey)
        self.rec_button = tk.Canvas(self, width=size, height=size,
                                     bg=self.col_grey, highlightthickness=1,
                                     highlightbackground=self.dark_grey)
        self.stop_button = tk.Canvas(self, width=size, height=size,
                                     bg=self.col_grey, highlightthickness=1,
                                     highlightbackground=self.dark_grey)
        self.add_button = tk.Canvas(self, width=size, height=size,
                                     bg=self.col_grey, highlightthickness=1,
                                     highlightbackground=self.dark_grey)
        self.reader_button = tk.Canvas(self, width=size, height=size,
                                     bg=self.col_grey, highlightthickness=1,
                                     highlightbackground=self.dark_grey)

        self.play_button.grid(row=0, column=0, padx=1, pady=1)
        self.rec_button.grid(row=0, column=1, padx=1, pady=1)
        self.stop_button.grid(row=0, column=2, padx=1, pady=1)
        self.add_button.grid(row=0, column=3, padx=1, pady=1)
        self.reader_button.grid(row=0, column=4, padx=1, pady=1)

        # draw icons...
        to = 6

        # PLAY
        self.play_icon = self.play_button.create_polygon(to, to, to, size-to+1,
                                        size-to+1, size//2+1,
                                        fill=self.dark_grey, outline='')

        # REC
        self.rec_icon = self.rec_button.create_oval(to, to, size-to+1, size-to+1,
                                        fill=self.dark_grey, outline='')

        # STOP
        self.stop_icon = self.stop_button.create_rectangle(to, to, size-to+1,
                                        size-to+1, fill=self.dark_grey, outline='')
        # ADD
        self.add_button.create_rectangle(size//2-2, to, size//2+4, size-to+1,
                                         fill=self.dark_grey, outline='')
        self.add_button.create_rectangle(to, size//2-2, size-to+1, size//2+4,
                                         fill=self.dark_grey, outline='')

        # READER
        a = to
        b = size//2-2
        c = size//2+4
        d = size-to+1
        self.reader_button.create_polygon(b, a, c, a, c, b, d, b, d, c, c, c, c, d, b, d, b, c, a, c, a, b, b, b, b, a,
                                         fill='', outline=self.dark_grey)


        self.play_button.bind('<Button-1>', self.play)
        self.rec_button.bind('<Button-1>', self.record)
        self.stop_button.bind('<Button-1>', self.stop)
        self.add_button.bind('<Button-1>', self.add)
        self.reader_button.bind('<Button-1>', self.add_reader)

    def update_buttons(self):
        # PLAY button...
        if self.app.clockVar['playing']:
            self.play_button.configure(bg='orange')
        else:
            self.play_button.configure(bg=self.col_grey)

        # REC button...
        if self.app.clockVar['recording']:
            self.rec_button.itemconfig(self.rec_icon, fill='red')
            self.rec_button.configure(bg='orange')
        else:
            self.rec_button.itemconfig(self.rec_icon, fill=self.dark_grey)
            self.rec_button.configure(bg=self.col_grey)

        # STOP button...
        if self.engine.clockVar['stopping']:
            self.stop_button.configure(bg='orange')
        else:
            self.stop_button.configure(bg=self.col_grey)

    def play(self, event):
        self.app.clock.toggle_playback()
        #self.update_buttons()

    def record(self, event):
        pass
        #self.engine.record = not self.engine.record
        #self.update_buttons()

    def stop(self, event):
        self.app.stop()
        #if self.engine.play_request.is_set():
        #    self.stop_button.configure(bg='yellow')
        #else:
        #    self.stop_button.configure(bg=self.col_grey)


    def add(self, event):
        self.app.ins_manager.addInstrument()

    def add_reader(self, event):
        self.app.ins_manager.addInstrument(player='reader')

class InstrumentPanelBase(tk.Frame):
    def __init__(self, master, instrument, name='Instrument', **options):
        tk.Frame.__init__(self, master, **options)
        self.configure(padx=2, pady=2)
        self.instrument = instrument
        self.ins_manager = instrument.ins_manager
        self.name = tk.StringVar()
        self.name.set(name + ' ({})'.format(instrument.ins_id))
        self.chan = tk.IntVar(self)
        self.chan.set(instrument.chan)
        self.colour = INS_COLOURS[self.instrument.ins_id % len(INS_COLOURS)]
        self.bar = -1
        self.global_bar = -1

        self.controlFrame = tk.Frame(self, bg=COLOR_SCHEME['panel_bg'])
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

        self.statusLabel = tk.Label(self.controlFrame, text='',
                                     bg=self.colour[0], fg=self.colour[1], anchor='w')

        self.chanLabel = tk.Label(self.controlFrame, textvariable=self.chan,
                                  bg=self.colour[0], fg=self.colour[1],
                                  anchor='e')
        self.chanLabel.bind('<Button-1>', lambda event: self.editEntry(event,
                                                                  self.chanUpdate))


        # ------ Display track
        self.update()
        self.canvasHeight = 100
        self.beat_width = 25
        self.bar_width = 4 * self.beat_width
        self.null_bars = 1
        self.barCanvas = tk.Canvas(self, width=self.winfo_width(),
                                   height=self.canvasHeight, bg=COLOR_SCHEME['note_panel_bg'])
        self.barCanvas.grid(row=0, column=1, sticky='ew', padx=2, pady=2)

        self.loopRegion = self.barCanvas.create_rectangle(-100, 0, -100,
                                    self.canvasHeight, fill='#605500')
        self.recRegion = self.barCanvas.create_rectangle(-100, 0, -100,
                                    self.canvasHeight, fill='#800000')

        self.cursor = self.barCanvas.create_line(1, 0, 1, self.canvasHeight,
                                                 fill='orange', width=3)

        # ------ Pack all the elements...
        self.colourStrip.grid(row=0, column=0, rowspan=6, sticky='ns')
        self.removeButton.grid(row=0, column=1, sticky='ew')
        self.statusLabel.grid(row=0, column=2, sticky='ew')
        self.nameLabel.grid(row=0, column=3, columnspan=1, sticky='ew')
        self.chanLabel.grid(row=0, column=4, sticky='ew')

        # initialise bar display...
        self.update_canvas()
        self.move_canvas(0)

        self.bind('<Configure>', self.onConfigure)

    def onConfigure(self, event):
        self.barCanvas.configure(width=self.winfo_width())

    def update_buttons(self):
        if self.instrument.status == PAUSED:
            self.pauseButton.config(text='paused')
            self.pauseButton.config(foreground=COLOR_SCHEME['text_light'])
        elif self.instrument.status == PAUSE_WAIT:
            self.pauseButton.config(text='pausing')
            self.pauseButton.config(foreground='orange')
        elif self.instrument.status == PLAY_WAIT:
            self.pauseButton.config(text='playing')
            self.pauseButton.config(foreground='orange')
        elif self.instrument.status == PLAYING:
            self.pauseButton.config(text='playing')
            self.pauseButton.config(foreground=COLOR_SCHEME['text_light'])


    def update_highlighted_bars(self):
        if self.instrument.loopLevel > 0:
            loopEnd = self.instrument.loopEnd
            loopStart = max(0, loopEnd - self.instrument.loopLevel+1)
            x = loopStart * self.bar_width
            width = loopEnd - loopStart + 1
            self.barCanvas.coords(self.loopRegion, x, 0,
                                  x+width*self.bar_width, self.canvasHeight)

        else:
            self.barCanvas.coords(self.loopRegion, -100, 0, -100, self.canvasHeight)

        if self.instrument.armed and self.instrument.record_bars:
            recStart = self.instrument.record_bars[0]
            recEnd = self.instrument.record_bars[-1]

            x = recStart * self.bar_width
            width = recEnd - recStart
            self.barCanvas.coords(self.recRegion, x, 0,
                                 x+width*self.bar_width, self.canvasHeight)

            for i in range(recStart, recEnd+20):
                x = i * self.bar_width
                self.barCanvas.create_line(x, 0, x, 100, fill='#aaaaaa',
                                           tags='redraw')
                self.barCanvas.create_text(x+5, 5, text=i,
                                           fill='#aaaaaa', tags='redraw')

        else:
            self.barCanvas.coords(self.recRegion, -100, 0, -100, self.canvasHeight)


    def clear_canvas(self, start):
        ''' Clears the canvas over a range of bar numbers '''
        bbox = self.barCanvas.bbox('all')
        print(start)

        x1 = start * self.bar_width

        elements = self.barCanvas.find_overlapping(x1, -5, bbox[2] + 5, self.canvasHeight + 5)

        for item in elements:
            if item == self.cursor or \
               item == self.loopRegion or \
               item == self.recRegion:
                continue
            self.barCanvas.delete(item)

    def update_canvas(self):
        ''' Updates the canvas with the notes to draw '''
        self.null_bars = 1

        #self.barCanvas.delete('redraw')
        self.beat_width = 25
        self.bar_width = 4 * self.beat_width
        noteRange = (36, 97)   # +- two octaves from middle C
        scale = self.canvasHeight / (noteRange[0] - noteRange[1])

        bar = self.instrument.bar_num

        stream = self.instrument.stream

        # draw bars
        for i in range(bar - self.null_bars - 1, len(stream)+20):
            x = i * self.bar_width
            self.barCanvas.create_line(x, 0, x, 100, fill='#aaaaaa',
                                       tags='redraw')
            self.barCanvas.create_text(x+5, 5, text=i,
                                       fill='#aaaaaa', tags='redraw')

        # draw notes
        for n in stream.notes:
            if n.drawn:
                continue

            x = n.getOffset()*self.beat_width
            y = scale * (n.midi - noteRange[1])
            note_width = n.getDuration() * self.beat_width

            if n.isChord():
                for c in n.chord:
                    interval = c*scale
                    self.barCanvas.create_line(x, y+interval, x+note_width-2,
                                        y+interval, fill='#aaaaaa', width=2,
                                        tags='note')
            else:
                self.barCanvas.create_line(x, y, x+note_width-2, y,
                                          fill='#aaaaaa', width=2, tags='note')

            n.drawn = True

        # draw cursor
        if not self.cursor:
            x = self.instrument.bar_num * self.bar_width
            self.cursor = self.barCanvas.create_line(x, 0, x, 100, fill='orange',
                                                 width=3)

    def move_canvas(self, beat):
        if beat < 0.05:
            self.bar = self.instrument.bar_num

        # update cursor position
        if self.instrument.status == PAUSED or self.instrument.status == PLAY_WAIT:
            beat = 0

        x = self.bar * self.bar_width + beat*self.beat_width
        self.barCanvas.coords(self.cursor, x, 0, x, 100)

        # scroll canvas
        b_frac = beat / 4
        scroll_x = (self.bar + b_frac - 2) * self.bar_width
        self.barCanvas.config(scrollregion=(scroll_x, 0, scroll_x + 4*self.bar_width, 100))

        self.barCanvas.xview('moveto', 0)

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

    def updateLead(self):
        if self.instrument.lead:
            #self.controlFrame.config(bg=COLOR_SCHEME['panel_select'])
            self.statusLabel.config(text='(LEAD)')
        else:
            #self.controlFrame.config(bg=COLOR_SCHEME['panel_bg'])
            self.statusLabel.config(text='')

    def loopUpdate(self, variable):
        self.instrument.toggle_loop(int(variable.get()))

    def remove(self, event):
        self.instrument.delete()

    def toggle_playback(self):
        self.instrument.toggle_paused()

    def continuous(self):
        self.instrument.toggle_continuous()

    def getMetaParams(self):
        return None

    def updateMetaParams(self, params):
        return


class InstrumentPanel(InstrumentPanelBase):
    def __init__(self, master, instrument, name='Instrument', **options):
        InstrumentPanelBase.__init__(self, master, instrument, name, **options)

        # ------ Player Parameter Knobs 
        self.playerParamFrame = tk.Frame(self.controlFrame, bg=self.controlFrame.cget('bg'))
        # Controls for:
        # - span
        # - center
        # - chord density
        # - chord depth
        # - average interval (jump)
        # - rhythmic density

        self.spanVar = tk.DoubleVar()
        self.centVar = tk.DoubleVar()
        self.cDenVar = tk.DoubleVar()
        self.cDepVar = tk.DoubleVar()
        self.jumpVar = tk.DoubleVar()
        self.rDenVar = tk.DoubleVar()
        self.sPosVar = tk.DoubleVar()

        self.spanVar.trace('w', self.updateSpan)
        self.centVar.trace('w', self.updateCent)
        self.cDenVar.trace('w', self.updateCDen)
        self.cDepVar.trace('w', self.updateCDep)
        self.jumpVar.trace('w', self.updateJump)
        self.rDenVar.trace('w', self.updateRDen)
        self.sPosVar.trace('w', self.updateSPos)

        self.spanKnob = Knob(self.playerParamFrame, 10, self.spanVar, bg=COLOR_SCHEME['panel_bg'],
                             name='span', min_=1, max_=60, default=20)
        self.centKnob = Knob(self.playerParamFrame, 10, self.centVar, bg=COLOR_SCHEME['panel_bg'],
                             name='cent', min_=30, max_=90, default=60)
        self.cDenKnob = Knob(self.playerParamFrame, 10, self.cDenVar, bg=COLOR_SCHEME['panel_bg'],
                             name='cDen', min_=0, max_=1, default=0)
        self.cDepKnob = Knob(self.playerParamFrame, 10, self.cDepVar, bg=COLOR_SCHEME['panel_bg'],
                             name='cDep', min_=1, max_=6, default=1)
        self.jumpKnob = Knob(self.playerParamFrame, 10, self.jumpVar, bg=COLOR_SCHEME['panel_bg'],
                             name='jump', min_=0.1, max_=6.0, default=3.0)
        self.rDenKnob = Knob(self.playerParamFrame, 10, self.rDenVar, bg=COLOR_SCHEME['panel_bg'],
                             name='rDen', min_=0, max_=4, default=1)
        self.sPosKnob = Knob(self.playerParamFrame, 10, self.sPosVar, bg=COLOR_SCHEME['panel_bg'],
                             name='sPos', min_=0, max_=1, default=0)

        self.spanKnob.grid(row=0, column=0, sticky='ew')
        self.centKnob.grid(row=0, column=1, sticky='ew')
        self.cDenKnob.grid(row=0, column=2, sticky='ew')
        self.cDepKnob.grid(row=0, column=3, sticky='ew')
        self.jumpKnob.grid(row=0, column=4, sticky='ew')
        self.rDenKnob.grid(row=0, column=5, sticky='ew')
        self.sPosKnob.grid(row=0, column=6, sticky='ew')

        # ------ Mode selection...
        self.lead_mode = tk.IntVar()
        self.lead_mode_select = ModeSelect(self.controlFrame, self.lead_mode,
                                           'lead:', ['none', 'both', 'melody'],
                                           default=1,
                                           bg=self.controlFrame.cget('bg'))

        self.sample_mode = tk.IntVar()
        self.sample_mode_select = ModeSelect(self.controlFrame, self.sample_mode,
                                           'sample:', ['top', 'dist', 'best'],
                                           default=1,
                                           bg=self.controlFrame.cget('bg'))

        self.context_mode = tk.IntVar()
        self.context_mode_select = ModeSelect(self.controlFrame, self.context_mode,
                                           'context:', ['none', 'top', 'real', 'inject'],
                                           default=3,
                                           bg=self.controlFrame.cget('bg'))

        # ------ Bar injection configuration...
        self.injectionFrame = tk.Frame(self.controlFrame, bg=self.controlFrame.cget('bg'))
        self.rhythmInjLabel = tk.Label(self.injectionFrame, bg=self.controlFrame.cget('bg'),
                                       fg=COLOR_SCHEME['text_light'], text='beats:')
        self.injectionVars = {
            'qb': (tk.BooleanVar(), 'quar'),
            'eb': (tk.BooleanVar(), 'egth'),
            'fb': (tk.BooleanVar(), 'fast'),
            'lb': (tk.BooleanVar(), 'long'),
            'tb': (tk.BooleanVar(), 'trip')
        }

        self.scale = tk.IntVar()
        self.scale_select = ModeSelect(self.injectionFrame, self.scale,
                                       'scale:', ['maj', 'min', 'pen', '5th'],
                                       bg=self.controlFrame.cget('bg'))
        self.injectionButtons = dict([(k, SimpleButton(self.injectionFrame, variable=v[0], label=v[1]))
                                      for k, v in self.injectionVars.items()])

        self.injectionVars['qb'][0].set(True)

        # ------ Redundant controls for now...
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

        # ------ Grid selection panels

        self.repeatVar = tk.StringVar(self.controlFrame)
        self.repeatSelect = SelectionGrid(self.controlFrame, self.repeatVar, 1,
                                          4, [1, 2, 4, 8], self.loopUpdate,
                                          'loop:', colour='orange', bg=self.controlFrame.cget('bg'))
        self.repeatVar.trace('w', self.repeatVarUpdate)

        self.chordVar = tk.StringVar(self.controlFrame)
        self.chordSelect = SelectionGrid(self.controlFrame, self.chordVar, 1, 4,
                                         [1, 2, 3, 4], None,
                                         'chrd:', colour='orange', bg=self.controlFrame.cget('bg'))
        #self.recVar.trace('w', self.recVarUpdate)

        self.rhythmVar = tk.StringVar(self.controlFrame)
        self.rhythmSelect = SelectionGrid(self.controlFrame, self.rhythmVar, 1,
                                          4, [1, 2, 3, 4], self.rhythmUpdate,
                                          'rhythm:', colour='orange', bg=self.controlFrame.cget('bg'))
        self.rhythmVar.trace('w', self.rhythmVarUpdate)

        # ------ Controls
        self.hold = tk.BooleanVar()
        self.holdButton = SimpleButton(self.controlFrame, variable=self.hold, label='hold')

        self.pauseButton = tk.Button(self.controlFrame, bg=self.controlFrame.cget('bg'), fg=COLOR_SCHEME['text_light'], text='pause',
                                    activebackground='orange', activeforeground='black')
        self.pauseButton['command'] = self.instrument.toggle_paused

        self.lengthVar = tk.DoubleVar(self.controlFrame)
        self.lengthVar.trace('w', self.lengthUpdate)
        self.lengthSlider = tk.Scale(self.controlFrame, from_=0, to_=1, bg=self.controlFrame.cget('bg'),
                                     orient=tk.HORIZONTAL, resolution=0.1, activebackground='orange',
                                     showvalue=False, variable=self.lengthVar,
                                     troughcolor=COLOR_SCHEME['panel_bg'])

        # ------ Pack all the elements...
        self.playerParamFrame.grid(row=1, column=1, columnspan=4, sticky='ew')
        #self.confidenceOption.grid(row=1, column=1)
        #self.transpose.grid(row=1, column=2)
        #self.continuousButton.grid(row=1, column=3)
        self.repeatSelect.grid(row=2, column=1, columnspan=2)
        self.chordSelect.grid(row=3, column=1, columnspan=2)
        self.rhythmSelect.grid(row=2, column=4)
        self.holdButton.grid(row=2, column=3, sticky='ew')
        self.pauseButton.grid(row=3, column=3, sticky='ew')
        self.lengthSlider.grid(row=3, column=4, sticky='ew')

        self.lead_mode_select.grid(row=4, column=1, columnspan=2, sticky='ew')
        self.sample_mode_select.grid(row=4, column=3, columnspan=1, sticky='ew')
        self.context_mode_select.grid(row=4, column=4, columnspan=2, sticky='ew')

        self.injectionFrame.grid(row=5, column=1, columnspan=5)
        self.rhythmInjLabel.grid(row=0, column=1)
        self.scale_select.grid(row=0, column=0)
        for i, button in enumerate(self.injectionButtons.values()):
            button.grid(row=0, column=i+2, sticky='ew')


    def repeatVarUpdate(self, *args):
        val = self.repeatVar.get()
        x = {'0':1, '1':2, '2':3, '4':4, '8':5}[val]
        self.ins_manager.send_touchOSC_message('/{}/loopBars'.format(self.instrument.ins_id+1), (1, x, 1))

    def rhythmVarUpdate(self, *args):
        val = self.rhythmVar.get()
        x = int(val)+1
        self.ins_manager.send_touchOSC_message('/{}/loopRhythm'.format(self.instrument.ins_id+1), (1, x, 1))

    def updateSpan(self, *args):
        self.instrument.update_params({'span': self.spanVar.get()})

    def updateCent(self, *args):
        self.instrument.update_params({'tCent': self.centVar.get()})

    def updateCDen(self, *args):
        self.instrument.update_params({'cDens': self.cDenVar.get()})

    def updateCDep(self, *args):
        self.instrument.update_params({'cDepth': self.cDepVar.get()})

    def updateJump(self, *args):
        self.instrument.update_params({'jump': self.jumpVar.get()})

    def updateRDen(self, *args):
        self.instrument.update_params({'rDens': self.rDenVar.get()})

    def updateSPos(self, *args):
        self.instrument.update_params({'pos': self.sPosVar.get()})

    def transUpdate(self, event):
        self.instrument.transpose = self.transposeVar.get()

    def confidenceUpdate(self, event):
        self.instrument.confidence = self.confidence.get()

    def recUpdate(self, variable):
        num = int(variable.get())
        self.ins_manager.set_recording_instrument(self.instrument, num)

    def rhythmUpdate(self, variable):
        num = int(variable.get())
        self.instrument.toggle_rhythm_loop(num)

    def lengthUpdate(self, *args):
        self.instrument.offMode = self.lengthVar.get()

    def remove(self, event):
        self.instrument.delete()

    def toggle_playback(self):
        self.instrument.toggle_paused()

    def continuous(self):
        self.instrument.toggle_continuous()

    def changeParameter(self, param, percent):
        KEYS = {'span': self.spanKnob,
                'tCent': self.centKnob,
                'cDens': self.cDenKnob,
                'cDepth': self.cDepKnob,
                'jump': self.jumpKnob,
                'rDens': self.rDenKnob,
                'pos': self.sPosKnob}

        KEYS[param].update_percent(percent)

    def getMetaParams(self):
        ''' returns a dictionary of meta params '''
        return {'span':   self.spanKnob.val,
                'tCent':  self.centKnob.val,
                'cDens':  self.cDenKnob.val,
                'cDepth': self.cDepKnob.val,
                'jump':   self.jumpKnob.val,
                'rDens':  self.rDenKnob.val,
                'pos':    self.sPosKnob.val}

    def updateMetaParams(self, params):
        if 'span' in params:
            self.spanKnob.val = params['span']
            self.spanKnob.update_line()

        if 'tCent' in params:
            self.centKnob.val = params['tCent']
            self.centKnob.update_line()

        if 'cDens' in params:
            self.cDenKnob.val = params['cDens']
            self.cDenKnob.update_line()

        if 'cDepth' in params:
            self.cDepKnob.val = params['cDepth']
            self.cDepKnob.update_line()

        if 'jump' in params:
            self.jumpKnob.val = params['jump']
            self.jumpKnob.update_line()

        if 'rDens' in params:
            self.rDenKnob.val = params['rDens']
            self.rDenKnob.update_line()

        if 'pos' in params:
            self.sPosKnob.val = params['pos']
            self.sPosKnob.update_line()


class DataReaderPanel(InstrumentPanelBase):
    def __init__(self, master, instrument, name='Instrument', **options):
        InstrumentPanelBase.__init__(self, master, instrument, name, **options)

        self.label = tk.Label(self.controlFrame, bg=self.controlFrame.cget('bg'),
                              fg=COLOR_SCHEME['text_light'], text='DATA READER')

        self.pauseButton = tk.Button(self.controlFrame, bg=self.controlFrame.cget('bg'), fg=COLOR_SCHEME['text_light'], text='pause',
                                    activebackground='orange', activeforeground='black')
        self.pauseButton['command'] = self.instrument.toggle_paused

        self.repeatVar = tk.StringVar(self.controlFrame)
        self.repeatSelect = SelectionGrid(self.controlFrame, self.repeatVar, 1,
                                          4, [1, 2, 4, 8], self.loopUpdate,
                                          'loop:', colour='orange', bg=self.controlFrame.cget('bg'))
        self.repeatVar.trace('w', self.repeatVarUpdate)
        
        self.fileVar = tk.StringVar(self.controlFrame)
        options = os.listdir('./userContexts/')
        if len(options) == 0:
            options.append('None')
        print(options)
        self.fileVar.set('')
        self.loadBox = tk.OptionMenu(self.controlFrame, self.fileVar, *options)
        self.fileVar.trace('w', self.instrument.load_file)

        self.lengthVar = tk.DoubleVar(self.controlFrame)
        self.lengthVar.trace('w', self.lengthUpdate)
        self.lengthSlider = tk.Scale(self.controlFrame, from_=0, to_=1, bg=self.controlFrame.cget('bg'),
                                     orient=tk.HORIZONTAL, resolution=0.1, activebackground='orange',
                                     showvalue=False, variable=self.lengthVar,
                                     troughcolor=COLOR_SCHEME['panel_bg'])


        # ------ Pack all the elements...
        self.label.grid(row=2, column=2)
        self.repeatSelect.grid(row=2, column=1, columnspan=2)
        self.pauseButton.grid(row=2, column=3, sticky='ew')
        self.lengthSlider.grid(row=2, column=4, columnspan=1, sticky='ew')
        self.loadBox.grid(row=3, column=2, columnspan=3, sticky='ew')



    def repeatVarUpdate(self, *args):
        val = self.repeatVar.get()
        x = {'0':1, '1':2, '2':3, '4':4, '8':5}[val]
        self.ins_manager.send_touchOSC_message('/{}/loopBars'.format(self.instrument.ins_id+1), (1, x, 1))

    def lengthUpdate(self, *args):
        self.instrument.offMode = self.lengthVar.get()






