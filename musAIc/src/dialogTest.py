import simpleDialog
import tkinter as tk

class MyDialog(simpleDialog.Dialog):

    def body(self, master):

        tk.Label(master, text="First:").grid(row=0)
        tk.Label(master, text="Second:").grid(row=1)

        self.e1 = tk.Entry(master)
        self.e2 = tk.Entry(master)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        return self.e1 # initial focus

    def apply(self):
        first = int(self.e1.get())
        second = int(self.e2.get())
        print(first, second) # or something
        self.result = (first, second)
        return (first, second)

root = tk.Tk()
tk.Button(root, text="Hello!").pack()
root.update()

d = MyDialog(root)
print(d)
print(d.result)

#root.wait_window(d.top)
