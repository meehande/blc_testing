# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:38:49 2016

@author: Deirdre Meehan
"""

from Tkinter import *
import demo_factorise
import blc
import numpy as np
class Application(Frame):
    def say_hi(self):
        print "hi there, everyone!"  
    
    def factorise(self):
        #call test that does factorisation with each dataset...
        if self.datasetoption.get() == "real":
            print "real factorize..."    
            return
        if self.datasetoption.get() == "theoretical":
            print "theoretical.."
            ferr_temp, perr_temp, self.Rsampled, self.Rmissing, self.Rtilde, self.Ut, self.Vt, self.P = demo_factorise.demofactorise(0)
            self.ferr.set(ferr_temp)
            self.perr.set(perr_temp)         
            return
        else:
            print "no input"
            return

    def createWidgets(self):
        frame0 = Frame(self)
        frame0.pack(pady = 20)
        frame1 = Frame(self)
        frame1.pack(side = BOTTOM, pady=20)
                
#-------CHOOSE DATASET OPTIONMENU---------------------------------------------#
  
        self.datasetoption = StringVar(frame0)
        self.datasetoption .set("Choose Dataset")
        self.ChooseDataset = OptionMenu(frame0, self.datasetoption, "theoretical", "real")        
        #self.ChooseDataset.pack(side="left")
        self.ChooseDataset.grid(row=0, column=0)
        
#-------FACTORISE BUTTON------------------------------------------------------#        
        self.FACTORISE = Button(frame0)
        self.FACTORISE["text"] = "Factorise"
        self.FACTORISE.pack(side = "left", pady = 20)
        self.FACTORISE["command"] = self.factorise
        self.FACTORISE.grid(row=0, column=1)
        
#-------RESULTS LABEL------------------------------------------------------# 
        pText = StringVar()
        pText.set("Prediction Error")
        self.PredictionTextLabel = Label(frame0, textvariable=pText)
        self.PredictionTextLabel.pack(side="left", pady = 10)
        self.PredictionTextLabel.grid(row=1, column=0)
        
        self.PredictErrLabel = Label(frame0, textvariable = self.perr)
        self.PredictErrLabel.pack(side = "right", pady=10)
        self.PredictErrLabel.grid(row=1, column=1)
        
        fText = StringVar()
        fText.set("Factorisation Error")
        self.PredictionTextLabel = Label(frame0, textvariable=fText)
        self.PredictionTextLabel.pack(side="left", pady = 10)
        self.PredictionTextLabel.grid(row=2, column=0)
        
        self.FactErrLabel = Label(frame0, textvariable = self.ferr)
        self.FactErrLabel.pack()
        self.FactErrLabel.grid(row=2, column=1)


#-------CHOOSE USER FOR RECOMMENDATION--------------------------------------------#
        self.ChooseUser = Entry(frame1)#**Add "command" attribute to add function... - put validation on input here
        self.ChooseUser.pack(side="left")
        self.chosenuser.set(self.ChooseUser.get())
        #print self.ChooseUser.get()
#-------GEN RECOMMENDATION BUTTON------------------------------------------------------#        
        self.RECOMMEND = Button(frame1)
        self.RECOMMEND["text"] = "Generate recommendation"
        self.RECOMMEND["command"] = self.recommendation
        self.RECOMMEND.pack({"side": "left"})
        #self.FACTORISE["command"] = some function to do the things
#http://python-textbok.readthedocs.org/en/latest/Introduction_to_GUI_Programming.html

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.ferr = IntVar()
        self.perr = IntVar()
        self.chosenuser = StringVar()
        self.ferr.set(0)
        self.perr.set(0)
        self.chosenuser.set("yass")
        self.pack()
        self.createWidgets()
    def validate(self, new_text):
        if not new_text: # the field is being cleared
            self.entered_number = 0
            return True
        try:
            self.entered_number = int(new_text)
            return True
        except ValueError:
            return False
            
    def recommendation(self):
        print "herey"
        self.chosenuser.set(self.ChooseUser.get())
        user = self.chosenuser.get()
        print "user\n", user
        user = int(user)
        group = np.argmax(self.P[:,user])#group the chosen user is in
        self.rec_vector = blc.recommend(self.Rsampled[user,:], self.Ut, self.Vt, group)
        print "recommendation vector\n",rec_vector
        xV = np.dot(self.rec_vector, self.Vt)#this is the predicted UV from that user - gives the recommendation
        Ru = np.expand_dims(self.Rsampled[user, :],0)
        perr = blc.rms(Ru, self.rec_vector.T, self.Vt)
        print "error in rec\n", perr        
        rated = Ru!=0
        if not(rated.all()):
            xV[rated] = np.nan
            recommend_item = np.nanargmax(xV)
            print "recommendation\n", recommend_item
        
        return
           

root = Tk()
app = Application(master=root)
app.master.geometry("400x400")#set initial window size

app.mainloop() 
root.destroy()

#**add density option...




"""     
        self.Dataset = Menubutton(frame0, text = "Choose Dataset")
        self.Dataset.grid()
        self.Dataset.menu = Menu(self.Dataset, tearoff=0)
        self.Dataset["menu"] = self.Dataset.menu
        theoretical = IntVar()
        real = IntVar()
        self.Dataset.menu.add_checkbutton(label="theoretical", variable=theoretical)
        self.Dataset.menu.add_checkbutton(label="real", variable=real)        
        self.Dataset.pack(side="left", padx = 40, pady=20)
        
        
        
        
        self.QUIT = Button(frame0)
        self.QUIT["text"] = "QUIT"
        self.QUIT["command"] =  self.quit

        self.QUIT.pack({"side": "left"})

        self.hi_there = Button(frame0)
        self.hi_there["text"] = "Hello",
        self.hi_there["command"] = self.say_hi

        self.hi_there.pack({"side": "left"})        
        
        
        
"""  