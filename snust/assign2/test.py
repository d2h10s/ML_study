from tkinter import *

def selectFunc():
    global listbox, lbl
    print(lisbox.curselection())

root = Tk()
root.geometry('600x400')
root.resizable(width=False, height=False)
listbox = Listbox(root)
listbox.pack()
for i in range(30):
    listbox.insert(i,i)
btn = Button(root, text='선택', command=selectFunc)
btn.pack(pady=10)
lbl = Lable(root, text='선택한 항목:')
lbl.pack()
root.mainloop()