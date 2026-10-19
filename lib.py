from sys import stdout
from csv import DictReader, DictWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import svm

class PeekyReader:
    def __init__(self, reader):
        self.peeked = None
        self.reader = reader

    def peek(self):
        if self.peeked is None:
            self.peeked = next(self.reader)
        return self.peeked

    def __iter__(self):
        return self

    def __next__(self):
        if self.peeked is not None:
            ret = self.peeked
            self.peeked = None
            return ret
        try:
            return next(self.reader)
        except StopIteration:
            self.peeked = None
            raise StopIteration


class Person:
    def __init__(self, reader):
        self.__rows = []
        self.__idx = reader.peek()['id']
        self.train = True
        try:
            while reader.peek()['id'] == self.__idx:
                self.__rows.append(next(reader))
        except StopIteration:
            pass

    def make_test(self):
        self.train = False
        
    @property
    def is_training(self):
        # returns True if this person is a part of training data;
        # False if it's in test data
        return self.train

    @property
    def score(self):
        return self.__rows[0]['score_text']

    @property
    def race(self):
        return self.__rows[0]['race']

    @property 
    def is_afam(self):
        if (self.race == "African-American"):
            return 1

        return 0
    
    @property
    def valid(self):
        return self.__rows[0]['is_recid'] != "-1"

    @property
    def compas_felony(self):
        return 'F' in self.__rows[0]['c_charge_degree']

    @property
    def score_valid(self):
        return self.score in ["Low", "Medium", "High"]

    @property
    def recidivist(self):
        return self.__rows[0]['is_recid'] == "1"

    @property
    def rows(self):
        return self.__rows

    @property
    def low(self):
        return self.__rows[0]['score_text'] == "Low"

    @property
    def medium(self):
        return self.__rows[0]['score_text'] == "Medium"

    @property
    def high(self):
        return self.__rows[0]['score_text'] == "High"

    @property
    def age(self):
        return int(self.__rows[0]['age'])
    
    @property
    def priors(self):
        return int(self.__rows[0]['priors_count'])
    
    @property 
    def is_felony(self):
        if self.compas_felony:
            return 1
        
        return 0

    @property
    def juvenile_felony_count(self):
        return int(self.__rows[0]['juv_fel_count'])

    @property
    def juvenile_misdemeanor_count(self):
        return int(self.__rows[0]['juv_misd_count'])

    @property
    def juvenile_other_count(self):
        return int(self.__rows[0]['juv_other_count'])
    
    @property
    def decile_score(self):
        return self.__rows[0]['decile_score']
    
## helper functions
    
# computes the number of elements in the list @data, filtered
# according to the specified function @fn
def count(fn, data):
    return len(list(filter(fn, list(data))))

def train_test_split(population, frac_train):
    import random

    random.seed(1)
    
    for p in population:
        if (random.random() > frac_train):
            p.make_test()


def plot_costs(costs, total_costs_afam, total_costs_cauc):
    #p1 = plt.plot(costs,total_costs)
    p2 = plt.plot(costs,total_costs_afam)
    p3 = plt.plot(costs,total_costs_cauc)
    plt.ylim(0,1)
    plt.xlabel("False Positive Costs / False Negative Costs")
    plt.ylabel("Average Cost")
    plt.legend((p2[0], p3[0]), ('African-Americans', 'Caucasians'))
    plt.show()

def print_statistics(predictions, ground_truth):
    acc = accuracy(predictions, ground_truth)
    fpr = FPR(predictions, ground_truth)
    ppv = PPV(predictions, ground_truth)
    forval = FOR(predictions, ground_truth)

    print("Accuracy: ", acc, "FPR: ", fpr, "PPV: ", ppv, "FOR: ", forval)
    
def print_statistics_metascores(meta_scores, ground_truth):
    print("Predict M/H as Positives (ProPublica)")
    print_statistics(make_prediction_metascore(meta_scores, "Medium"), ground_truth)
    print("Predict only H as Positives")
    print_statistics(make_prediction_metascore(meta_scores, "High"), ground_truth)
    
# HOMEWORK 2 Implementation
# the next set of functions are for you to implement

def get_meta_scores(pop):
    """
    Get the meta scores for a population

    Parameters:
    pop (list): list of Persons
    """
    meta_scores = [p.score for p in pop]

    # TODO: implement
        
    return meta_scores

def get_ground_truth(pop):
    """
    Get the ground truth for a population

    Parameters:
    pop (list): list of Persons
    """
    recid = [1 if p.recidivist else 0 for p in pop]

    # TODO: implement
        
    return recid

def accuracy(predictions, ground_truth):
    """
    Calculate the accuracy of a set of predictions 

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    """

    # TODO: implement accuracy measure
    # computes accuracy of predictions given ground truth
    correct = sum(1 for pred, ground_truth_val in zip(predictions, ground_truth) if pred == ground_truth_val)
    return correct / len(predictions) if predictions else 0

def FPR(predictions, ground_truth):
    """
    Calculate the false positive rate of a set of predictions 

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    """

    # TODO: implement false-positive-rate measure
    false_positives = sum(1 for pred, ground_truth_val in zip(predictions, ground_truth) if pred == 1 and ground_truth_val == 0)
    total_negatives = sum(1 for ground_truth_val in ground_truth if ground_truth_val == 0)
    return false_positives / total_negatives if total_negatives else 0


def PPV(predictions, ground_truth):
    """
    Calculate the positive predictive value (precision) of a set of predictions 

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    """

    # TODO: implement precision (positive predicted value) measure
    true_positives = sum(1 for pred, ground_truth_val in zip(predictions, ground_truth) if pred == 1 and ground_truth_val == 1)
    total_positives = sum(1 for pred in predictions if pred == 1)
    return true_positives / total_positives if total_positives else 0


def FOR(predictions, ground_truth):
    """
    Calculate the false omission rate of a set of predictions 

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    """

    # TODO: implement false omission rate measure
    false_omissions = sum(1 for pred, ground_truth_val in zip(predictions, ground_truth) if pred == 0 and ground_truth_val == 1)
    total_omissions = sum(1 for pred in predictions if pred == 0)
    return false_omissions / total_omissions if total_omissions else 0

def make_prediction_metascore(meta_scores, meta_threshold):
    """
    Make predictions based on the meta scores and threshold

    Parameters:
    meta_scores (list): list of strings representing the meta scores (i.e. "High")
    meta_threshold (string): threshold representing the lower bound on a positive prediction
    """
    order = {"Low": 0, "Medium": 1, "High": 2}
    preds = [1 if order[score] >= order[meta_threshold] else 0 for score in meta_scores]

    # TODO: implement
                
    return preds

def compute_cost(predictions, ground_truth, costfp):
    """
    Calculate the cost of a set of predictions  

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    costfp (int): cost of a false positive prediction 
    """
    cost = 0
    for pred, ground_truth_val in zip(predictions, ground_truth):
        if pred == 1 and ground_truth_val == 0:
            cost += costfp
        elif pred == 0 and ground_truth_val == 1:
            cost += 1
    return cost/len(predictions)

    # TODO: implement

def get_ave_costs_preds(predictions, ground_truth, costs):
    """
    Calculate the cost of a set of predictions for various false positive costs 

    Parameters:
    predictions (list): list of predictions 
    ground_truth (list): list of ground truth values 
    costs (list): list of ints representing the various false positive costs 
    """

    ave_costs = [compute_cost(predictions, ground_truth, costfp) for costfp in costs]

    # TODO: implement
        
    return ave_costs
    
def get_ave_costs(meta_scores, ground_truth, costs):
    """
    Computes predictions and gets the average costs for a set of meta scores 

    Parameters:
    meta_scores (list): meta scores for a population 
    ground_truth (list): list of ground truth values 
    costs (list): list of ints representing the various false positive costs 
    """
    preds = make_prediction_metascore(meta_scores, "High")
    ave_costs = get_ave_costs_preds(preds, ground_truth, costs)

    # TODO: implement
    
    return ave_costs

def construct_feature_matrix(population):
    """
    Generates the feature matrix for a population using the six features documented in the assignment 

    Parameters:
    population (list): list of Persons 
    """

    x = np.zeros((len(population), 6), dtype=int)
    for i, p in enumerate(population):
        x[i] = [p.age, p.is_felony, p.priors,
        p.juvenile_felony_count,
        p.juvenile_misdemeanor_count,
        p.juvenile_other_count]

    # TODO: populate the feature matrix x and return it

    return x

def get_predictions(population, model):
    """
    Makes predictions on the given population using the given model 

    Parameters:
    population (list): list of Persons 
    model: machine learning model to use for predictions 
    """

    pred = model.predict(construct_feature_matrix(population))

    # TODO: implement
    
    return pred

def learn(population, model, weights=None):    
    """
    Learning a model using a given population 

    Parameters:
    population (list): list of Persons
    model: model to train from the given training data  
    """

    y = np.zeros(len(population),dtype=int)

    # TODO: construct the array y of outputs
    # y = 1 for recidivists, and y = 0 for non-recidivists
    for i, person in enumerate(population):
        y[i] = 1 if person.recidivist else 0
    
    x = construct_feature_matrix(population)

    model = model.fit(x,y, sample_weight = weights)

    return model

def get_weights(population, w):
    """
    Calculates the weights which indicate how important certain points are to the classifier 

    Parameters:
    population (list): list of Persons 
    w (int): weight to assign to African American members of the population 
    """

    weights = np.ones(len(population), dtype=int)
    

    # TODO: implement
    return np.array([w if p.is_afam else 1 for p in population], dtype=float)
