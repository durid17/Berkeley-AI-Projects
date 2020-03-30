# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"

        # mira weights store all each result weights for each C in Cgrid
        # each weight is counted same way as perceptron training but instead of subtracting or adding vector to weights we add and 
        # subtract weighted vector, with tau which is counted with the formula from problem statement
        mira_weights = util.Counter()
        for C in Cgrid:
            new_weights = self.weights.copy()
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):

                    scores = util.Counter()
                    for label in self.legalLabels:
                        scores[label] = new_weights[label] * trainingData[i]
                    
                    trueLabel = trainingLabels[i]
                    possibleLabel = scores.argMax()
                    
                    if(trueLabel != possibleLabel):
                        x = (new_weights[possibleLabel] - new_weights[trueLabel]) * trainingData[i] + 1.0 
                        y =  (trainingData[i] * trainingData[i] * 2.0)
                                                
                        tau = min( C , x/y )
                        
                        for key in trainingData[i].keys(): 
                            change = trainingData[i][key] * tau             
                            new_weights[possibleLabel][key] -= change
                            new_weights[trueLabel][key] += change
            mira_weights[C] = new_weights
        
        # after all wights are counted we need to choose weights which gives best answer on validationData
        # and set it as self.weights
        bestScore = 0
        result_weight = None
        for C  in mira_weights:
            self.weights = mira_weights[C]
            guesses = self.classify(validationData)
            score = 0
            for i in range(len(guesses)):
                if guesses[i] == validationLabels[i]: score += 1
            if score > bestScore:
                bestScore = score
                result_weight = self.weights
            
        self.weights = result_weight
  



    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


