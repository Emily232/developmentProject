from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch

import csv

class Model(nn.Module):
    def __init__(self, input_count, output_count, hidden_layer_count, hidden_layer_node_count):
        super().__init__()

        self.hidden = [nn.Linear(hidden_layer_node_count if i else input_count, hidden_layer_node_count) for i in range(hidden_layer_count)]
        self.output = nn.Linear(hidden_layer_node_count, output_count)
        #Create hidden and output layers

    def forward(self, input):
        for layer in self.hidden:
            input = torch.sigmoid(layer(input))
            #Neuron activation function for each layer

        return self.output(input)

class Data(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.length = self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
    
    def __len__(self):
        return self.length

def trainModel(model, inputs, outputs, batchSize, learningRate, epochCount):
    dataset = Data(inputs, outputs)
    loader = DataLoader(dataset=dataset, batch_size=batchSize)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learningRate)
    #Define optimiser for model

    model.train()
    #Set model to training model

    for epoch in range(epochCount):
        for batchInputs, batchOutputs in loader:
            optimiser.zero_grad()
            #Reset gradients

            batchOutputs = batchOutputs.long()
            predictions = model(batchInputs)
            loss = criterion(predictions, batchOutputs)
            #Get predictions and calculate loss

            loss.backward()
            optimiser.step()

    model.eval()
    #Set model to eval mode

def evalModel(model, test_inputs, test_outputs):
    pred = model(test_inputs)
    yhat = torch.max(pred, 1)
    correct = 0
    #Get predictions

    for i, val in enumerate(yhat.indices):
        if test_outputs[i].item() == val.item():
            correct += 1
            #Count total correct predictions

    return correct / len(test_inputs)
    #Return accuracy

def readTrainingData(path, headings):
    inputs = []
    outputs = []
    unique_outputs = []
    fileHeadings = []

    with open(path) as file:
        reader = csv.reader(file, delimiter=",")
        #Read dataset file

        for i, row in enumerate(reader):
            if i or not headings:
                if row[-1] not in unique_outputs:
                    unique_outputs.append(row[-1])
                    #Get list of unique possible outputs

                inputs.append(row[:-1])
                outputs.append(unique_outputs.index(row[-1]))
                #Read inputs and outputs into lists
            else:
                fileHeadings = row
                #Get file headings from first row if defined in file

    return inputs, outputs, unique_outputs, fileHeadings

def splitTrainingData(inputs, outputs, trainSize):
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, train_size=trainSize, shuffle=True)
    #Split training data

    inputs_train = inputs_train.reshape(-1, inputs_train.shape[1]).astype("float32")
    inputs_test = inputs_test.reshape(-1, inputs_test.shape[1]).astype("float32")
    #Set types

    return torch.from_numpy(inputs_test), inputs_train, torch.from_numpy(outputs_test), outputs_train
    #Return as numpy arrays