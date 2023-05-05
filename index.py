import util
import ui

import numpy as np
import torch
import json

def trainModel(inputs, outputs, uniqueOutputs, inputSelection, accVar):
    trainTestInputs = []

    for i, inputSet in enumerate(inputs):
        trainTestInputs.append([])

        for j in range(len(inputSet)):
            if inputSelection[j]:
                trainTestInputs[i].append(inputSet[j])
        #Filter training dataset to only include inputs selected by user

    inputs_train, inputs_test, outputs_train, outputs_test = util.splitTrainingData(np.array(trainTestInputs, dtype=float), np.array(outputs, dtype=float), config["trainingDataSplit"])
    model[0] = util.Model(sum(inputSelection), len(uniqueOutputs), config["hiddenLayerCount"], config["hiddenLayerNodeCount"])
    #Split training data and create model

    util.trainModel(model[0], inputs_train, outputs_train, config["batchSize"], config["learningRate"], config["trainEpochs"])
    #Train model for defined amount of epochs

    acc = util.evalModel(model[0], torch.tensor(inputs_test, dtype=torch.float32), torch.tensor(outputs_test, dtype=torch.float32))
    #Calculate accuracy

    accVar.set(f"Accuracy using training data: {acc * 100:.2f}%")
    #Display accuracy

def runModel(inputs, uniqueOutputs, inputSelection, inputValues, outputVar):
    userInputs = []

    for i in range(len(inputs[0])):
        if inputSelection[i]:
            try:
                userInputs.append(float(inputValues[i]))
                #Try and convert user input to float
            except:
                return print(f"Error converting string to float for input \"{config['inputHeadings'][i]}\"")
                #Print error if user input is invalid

    pred = model[0](torch.tensor(userInputs, dtype=torch.float32))
    yhat = torch.max(pred.data, 0)
    #Get prediction

    print(uniqueOutputs[yhat.indices.item()])

    outputVar.set(f"Output: {uniqueOutputs[yhat.indices.item()]}")
    #Display output

def main():
    inputs, outputs, uniqueOutputs, headings = util.readTrainingData(config["trainingSet"], config["trainingSetHasHeadings"])
    #Read training data from dataset file

    if config["trainingSetHasHeadings"]:
        config["inputHeadings"] = headings
        #Use headings from file

    if len(config["inputHeadings"]) > len(inputs[0]):
        config["inputHeadings"] = config["inputHeadings"][:len(inputs[0])]
        #Remove input headings if there are more headings than inputs

    if len(inputs[0]) > len(config["inputHeadings"]):
        for i in range(len(config["inputHeadings"]), len(inputs[0])):
            config["inputHeadings"].append("[Unknown]")
            #Fill undefined headings if there are more inputs than headings

    ui.createWindow(config["inputHeadings"], lambda selection, accVar: trainModel(inputs, outputs, uniqueOutputs, selection, accVar), lambda selection, values, outputVar: runModel(inputs, uniqueOutputs, selection, values, outputVar))
    #Create tkinter window

if __name__ == "__main__":
    global model
    global config

    model = [None]
    #Model variable needs to be created in trainModel and used in runModel but these functions are both used as callbacks in ui.py
    #So use model as a global variable in index.py
    #Needs to be an array so it can be modified by reference instead of changing the value of a copy
    #Otherwise if model was modified in trainModel, when used in runModel it would hold the original value
    #Easier to use global variable than passing model array to createWindow?

    config = json.load(open("./config.json"))
    #Read config from file
    #Global variable for convenience

    main()

#Machine learning model based on https://github.com/lschmiddey/PyTorch-Multiclass-Classification/blob/master/Softmax_Regression_Deep_Learning_Iris_Dataset.ipynb