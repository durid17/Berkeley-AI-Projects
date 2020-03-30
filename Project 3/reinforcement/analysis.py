# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

# make noise equal to zero , so pacman will risk going on the cliff because there is no risk in getting trouble
def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

# answer discount is very small so pacman prefers 1 point early than 10 points afterwards
# noise is zero so pacman will risk the cliff
# living reward can be zero does not really matter
def question3a():
    answerDiscount = 0.00001
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# answer discount is 0.333 so pacman prefers 1 point early than 10 points afterwards
# noise is 0.2 so pacman will not risk goijnd over cliff because of -10 s
# living reward can be zero does not really matter
def question3b():
    answerDiscount = 0.333
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# answer discount is 1 so pacman will prefer 10 points afterwards to one point before
# noise is zero so pacman will risk the cliff
# living reward is small negative number so pacman will try to go to 10 as soon as possible
def question3c():
    answerDiscount = 1
    answerNoise = 0
    answerLivingReward = -0.005
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# answer discount is 1 so pacman will prefer 10 points afterwards to one point before
# noise is 0.2 so pacman will not risk goijnd over cliff because of -10 s
# living reward is small negative number so pacman will try to go to 10 as soon as possible
def question3d():
    answerDiscount = 1
    answerNoise = 0.2
    answerLivingReward = -0.005
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# answer discount is 1 so for pacman there is no different in points now to points later
# noise is zero so pacma will go where he wanst
# living reward is zero so pacman will not prefer taking point now to later
def question3e():
    answerDiscount = 1
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

#  50 iterations is very small for such a big world , so it is not possible
def question6():
    # answerEpsilon = None
    # answerLearningRate = None
    return 'NOT POSSIBLE'
    # return answerEpsilon, answerLearningRate
    # If not possible, 

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
