import tkinter as tk

def setSubmitState(submitButton, state):
    submitButton["state"] = tk.NORMAL if state else tk.DISABLED
    #Can't modify an outside variable from inside a lambda function

def createWindow(inputs, train, run):
    window = tk.Tk()
    outputVar = tk.StringVar()
    accVar = tk.StringVar()
    buttons = []
    opts = []

    tk.Button(window, text="Train", command=lambda: (train(list(map(lambda var: var.get(), buttons)), accVar), setSubmitState(submitButton, True))).grid(row=4, pady=(25, 5), columnspan=len(inputs))
    #Button to train the model

    submitButton = tk.Button(window, state=tk.DISABLED, text="Submit", command=lambda: run(list(map(lambda var: var.get(), buttons)), list(map(lambda var: var.get(), opts)), outputVar))
    submitButton.grid(row=5, columnspan=len(inputs))
    #Button to generate and display an output

    for i, input in enumerate(inputs):
        tk.Label(text=input).grid(row=0, column=i)
        #Label for each input

        checkboxVar = tk.BooleanVar(value=True)
        entryVar = tk.StringVar()
        button = tk.Checkbutton(window, variable=checkboxVar, command=lambda: setSubmitState(submitButton, False))
        input = tk.Entry(window, textvariable=entryVar)
        #Input box and checkbox for each input

        buttons.append(checkboxVar)
        opts.append(entryVar)
        button.grid(row=1, column=i)
        input.grid(row=2, column=i)
        #Place widgets in grid

        window.columnconfigure(i, weight=1)
        #Stretch all columns as much as possible with equal width on each

    tk.Label(textvariable=accVar).grid(row=6, pady=15, columnspan=len(inputs))
    tk.Label(textvariable=outputVar).grid(row=7, columnspan=len(inputs))
    #Accuracy and output text in window

    window.mainloop()