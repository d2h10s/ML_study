def refreshFunc():
    global lbx2, lstSelect
    lbx2.delete(0, lbx2.size())
    for i in lstSelect:
        lbx2.insert(END, i)

def startFunc():
    global lbx2, lstSelect
    if lbx2.size()!= 6:
        return
    lstLotto = sample(range(1,46),6)
    lstLotto.sort()
    answerText = 'answer: '
    for i in lstLotto:
        answerText += str(i)+', '
    answerLabel.config(text=answerText)
    ansCnt = 0
    for val in lstSelect:
        if val in lstLotto:
            ansCnt += 1
    msg = str(ansCnt)
    msg += '개 당첨되었습니다.\n'
    if ansCnt == 6:
        msg += '1등 상품 - 재민 처형권'
    if ansCnt == 5:
        msg += '2등 상품 - 재민 화형권'
    if ansCnt == 4:
        msg += '3등 상품 - 재민 고문권'
    if ansCnt == 3:
        msg += '4등 상품 - 재민 칼빵권'
    if ansCnt == 2:
        msg += '5등 상품 - 재민 죽빵권'
    if ansCnt == 1:
        msg += '6등 상품 - 10억!'
    else:
        msg += '꽝'
    messagebox.showinfo('추첨결과', msg)

   
def selectFunc():
    global lbx1, lstSelect
    idx = lbx1.curselection()[0]
    val = lbx1.get(idx)
    if val not in lstSelect:
        if len(lstSelect)==6:
            return
        lstSelect.append(val)
        lstSelect.sort()
        refreshFunc()


def deleteFunc():
    global lbx2, lstSelect
    if lbx2.size()==0:
        return
    idx = lbx2.curselection()[0]
    val = lbx2.get(idx)
    lstSelect.remove(val)
    refreshFunc()

lstSelect = []
from tkinter import *
from tkinter import font
from tkinter import messagebox
from random import *

root = Tk()
root.geometry('600x400+600+400')
root.resizable(width = False, height = False)
bgColor = 'azure'
root.config(bg = bgColor)
consolas15 = font.Font(family = 'consolas', size = 15, weight = 'bold')
consolas20 = font.Font(family = 'consolas', size = 20, weight = 'bold')
consolas30 = font.Font(family = 'consolas', size = 30, weight = 'bold')
titleLabel = Label(root)
titleLabel.config(text = 'lotto program', font = consolas30)
titleLabel.config(bg = bgColor, fg = 'gray26')
titleLabel.pack(pady = 30)

outFrame = Frame(root, bg = bgColor)
outFrame.pack()
frame1 = Frame(outFrame, bg = bgColor)
frame1.pack(side = 'left', padx = 10)
frame2 = Frame(outFrame, bg = bgColor)
frame2.pack(side = 'left', padx = 10)
frame3 = Frame(outFrame, bg = bgColor)
frame3.pack(side = 'left', padx = 10)

scroll = Scrollbar(frame1)
scroll.pack(side = 'right', fill = 'y')
lbx1 = Listbox(frame1, yscrollcommand = scroll.set)
for i in range(1, 46):
    lbx1.insert(END, i)
lbx1.pack(side = 'left')
scroll.config(command = lbx1.yview)

selectBtn = Button(frame2, text = '↪', font = consolas15, bg = 'snow', fg = 'blue')
selectBtn.config(command = selectFunc)
selectBtn.pack(pady = 5)
deleteBtn = Button(frame2, text = '↩', font = consolas15, bg = 'snow', fg = 'blue')
deleteBtn.config(command = deleteFunc)
deleteBtn.pack(pady = 5)

lbx2 = Listbox(frame3)
lbx2.pack()
startBtn = Button(root, text = 'start', font = consolas20, bg = 'snow', fg = 'blue')
startBtn.config(command = startFunc)
startBtn.pack(side='top')
answerLabel = Label(root, text='answer:')
answerLabel.pack()
root.mainloop()