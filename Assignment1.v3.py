"""Implementation of an exhaustive search algorithm for the solution of a combinatorial problem"""
# by Pietro Zafferani, Genomics

import random
from codetiming import Timer
import numpy
import scipy.optimize as optimization

'''The class Supplier creates instances representing objects of type Supplier, each instance is defined by an integer 
label, a weight randomly created and assigned, and a 'blacklist' that contains all the labels representing the other
suppliers that are not compatible with it, they are drawn randomly too.'''


class Supplier(object):

    def __init__(self, label: int, weight: float, blacklist: list, interactions: list):  # the initiator method assigns all the attributes
        # the attributes are set private since they cannot be invoked directly
        self.__label = label
        self.__weight = weight
        self.__blacklist = blacklist
        self.__interactions = interactions

    def get_label(self):  # allows to return the label
        return self.__label

    def get_weight(self):  # allows to return the weight
        return self.__weight

    def get_blacklist(self):  # allows to return the list of incompatibility
        return list(self.__blacklist)

    def get_interactions(self):   # allows to return the list of compatibilities
        return list(self.__interactions)

    def __str__(self) -> print:  # defines the modality to print each instance
        return 'Supplier_' + str(self.__label) + '\n' \
               + 'offers grams: ' + str(self.__weight) + '\n' \
               + 'competes with suppliers: ' + str(self.__blacklist) + '\n' \
               + 'interacts with: ' + str(self.__interactions)


'''This function draws randomly a floating number in [0,1) by exploiting the .random() method.
  It is called to assign a weight to each Supplier's instance.'''


def random_weight() -> float:
    return round(random.random(), 3)


'''This function creates randomly a list of n integers non-repeated and excluding a specific value passed as parameter.
It is exploited to create the 'black_list' assigned to each Supplier's instance.'''


def create_blacklist(n: int, i: int) -> tuple:
    blacklist = []
    count = 0
    space = [m for m in range(1, n + 1)]
    space.remove(i)
    while count < n // 2:  # condition imposed by the problem specification
        number = random.choice(space)
        space.remove(number)
        blacklist.append(number)
        count += 1

    return (blacklist, space)


'''This function iterates over the range of an integer passed as parameter that represents the number of 
different suppliers chosen, it returns a list of n instances of the Supplier class.'''


def create_suppliers(n: int) -> dict:
    suppliers = {}
    for i in range(n):
        lists = create_blacklist(n, i + 1)
        suppliers[i + 1] = (Supplier(i + 1, random_weight(), lists[0], lists[1]))

    return suppliers  # suppliers = ( Supplier 1 , ... , Supplier n )


'''Auxiliary function to print each Supplier instance, takes as parameter the supplier list described above. '''


def print_suppliers(f: dict) -> print:
    for supplier in f:
        print()
        print(f[supplier])


'''This function carries out the exhaustive search and it's the main part of the algorithm. As only parameter 
it takes the list containing n instances of the Supplier class. As output it produces a tuple with the best score 
possible and the combination of suppliers that yielded it.
The best result can be obtained with certainty only by searching through the whole search space since that is the 
only way to prove that the algorithm is correct.'''


def ExhaustiveSearch(Dict: dict) -> tuple:
    best_weight = 0  # set by default to 0
    best_set = []  # set by default empty

    for supplier in Dict.values():  # each Supplier instance is stored as a value in the dictionary
        affines = supplier.get_interactions()
        interference = supplier.get_blacklist()
        current_weight = supplier.get_weight()
        current_set = [supplier.get_label()]

        for affine in affines:
            compatible = True

            if affine not in interference:
                for i in current_set:
                    if i in Dict[affine].get_blacklist():
                        compatible = False

                if compatible:
                    current_weight += Dict[affine].get_weight()
                    interference += Dict[affine].get_blacklist()
                    current_set.append(affine)

            if current_weight >= best_weight:
                best_weight = current_weight
                best_set = current_set

    return (round(best_weight, 3), best_set)


"""This section of the file is composed by functions designed to test the running time of the exhaustive search
 algorithm and then to produce a set of data that can be used to analysis its efficiency."""

'''This function computes the running time of the exhaustive search given a list of suppliers and it reports the 
  average time after having repeated the execution of the algorithm 100 times, set by default'''


def TestAlgorithm(suppliers, repetitions=1000) -> float:
    for trials in range(repetitions):
        test = create_suppliers(suppliers)

        with Timer(name='ExhaustiveSearch', logger=None):
            ExhaustiveSearch(test)

    return Timer.timers.mean('ExhaustiveSearch')  # returns the average among all the execution times


'''this function allows to execute the previous function by passing as parameters a list of integers, every element
of this list specifies the number of supplier to be passed to .TestAlgorithm() . It returns a list containing the
respective running times.'''


def DataSets(Set: list) -> list:
    l = []
    for number in Set:
        l.append(TestAlgorithm(number))

    return l


'''this function returns the mathematical representation of the complexity of the exhaustive search algorithm.
It is exploited by the optimization.curve_fit() function to produce a good approximation of the algorithm's behaviour.
The total complexity of the algorithm is polynomial times exponential, so it is computationally very demanding.'''


def expo(x, a):
    return a * x ** 2  # complexity is polynomial


'''The only purpose of this function is to produce a set of pairs of data that can be easily implemented in the Latex
system.'''


def printPairs(Set, time_data) -> print:
    for (n, t) in zip(Set, time_data):  # Set = array of numbers of suppliers; time_data = respective running times
        print((n, round(t, 5)), end=' ')


if __name__ == '__main__':
    Set = [8, 12, 16, 18, 20, 22, 24, 26]  # each element is the numbers of suppliers considered
    times = DataSets(Set)  # running times produced by testing the Set list
    printPairs(Set, times)
    print()
    print(optimization.curve_fit(expo, Set, times))  # approximation of the algorithm's behaviour
    # print(TestAlgorithm(2))
