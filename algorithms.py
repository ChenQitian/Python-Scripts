# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:15:43 2016

@author: finn
"""

import numpy as np
from math import sqrt, log
from models import Parallel


def argmax_rand(x):
    """ return the index of the maximum element in the array, ignoring nans. 
    If there are multiple max valued elements return 1 at random"""
    max_val = np.nanmax(x)
    indicies = np.where(x == max_val)[0]
    return np.random.choice(indicies) 



class GeneralCausal(object):
    label = "Algorithm 2"
    
    def __init__(self,truncate = "clip"):
        self.truncate = truncate
        #self.label = "Algorithm 2-"+truncate

    def run(self,T,model):
        eta = model.eta
        m = model.m
        n = len(eta)
        self.B = sqrt(m*T/log(2.0*T*n))
        
        actions = range(n)
        u = np.zeros(n)
        for t in xrange(T):
            a = np.random.choice(actions,p=eta)
            x,y = model.sample(a) #x is an array containing values for each variable
            #y = y - model.get_costs()[a]
            pa = model.P(x)
            r = model.R(pa,eta)
            if self.truncate == "zero":
                z = (r<=self.B)*r*y
            elif self.truncate == "clip":
                z = np.minimum(r,self.B)*y
            else:
                z = r*y
                
            u += z
        self.u = u/float(T)
        r = self.u - model.get_costs()
        self.best_action = np.argmax(r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]   
    
        

class ParallelCausal(object):
    label = "Algorithm 1"
    
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        h = T/2
        for t in range(h):
            x,y = model.sample(model.K-1) # do nothing
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
            
        self.infrequent = self.estimate_infrequent(h)
        h=T-h
        n = int(float(h)/len(self.infrequent))
        self.trials[self.infrequent] = n # note could be improved by adding to rather than reseting observation results - does not change worst case. 
        self.success[self.infrequent] = model.sample_multiple(self.infrequent,n)
        self.u = np.true_divide(self.success,self.trials)
        self.r = self.u - model.get_costs()
        self.best_action = argmax_rand(self.r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
   
            
    def estimate_infrequent(self,h):
        qij_hat = np.true_divide(self.trials,h)
        s_indx = np.argsort(qij_hat) #indexes of elements from s in sorted(s)
        m_hat = Parallel.calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent
    
    
class ParallelCausal_addSR(object):
    label = "proposed algorithm"
    
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        h = T/2
        for t in range(h):
            x,y = model.sample(model.K-1) # do nothing
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
            
        self.infrequent = self.estimate_infrequent(h) # self.infrequent =set A
                
        # the second stage: use successive rejection
        h=T-h  # duo yi ge shi yi ge       
        actions=self.infrequent
        K1=len(actions)
        logkbar=0.5+np.sum(np.ones(K1-1)/[range(2,K1+1)])
        nj=0
        self.trials[actions]=0
        self.success[actions]=0
        temp_cost=model.get_costs()
        for j in range(K1-1):
            nj_1=nj
            nj=(h-K1)/(logkbar*(K1+1-j))
            self.trials[actions] = nj
            self.success[actions] += model.sample_multiple(actions,nj-nj_1)
            tempu=np.true_divide(self.success[actions],self.trials[actions])
            tempr=tempu-temp_cost[actions]  
            actions=actions[np.argsort(tempr)[1:]] # delete arg min tempr
            
        self.u = np.true_divide(self.success,self.trials)
        self.r = self.u - model.get_costs()
        array1=self.infrequent
        array1=array1[array1!=actions]
        self.r[array1]=-np.inf
        self.best_action = argmax_rand(self.r)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
   
            
    def estimate_infrequent(self,h):
        qij_hat = np.true_divide(self.trials,h)
        s_indx = np.argsort(qij_hat) #indexes of elements from s in sorted(s)
        m_hat = Parallel.calculate_m(qij_hat[s_indx])
        infrequent = s_indx[0:m_hat]
        return infrequent

class GeneralGraph_addSR(object):
    label="proposed algorithm2"
    def __init__(self,truncate = "clip"):
        self.truncate = truncate
        #self.label = "Algorithm 2-"+truncate

    def run(self,T,model):
        eta = model.eta
#        m = model.m
        n = len(eta)        
        actions = range(n)
        u = np.zeros(n)        

        logkbar=0.5+np.sum(np.ones(n-1)/[range(2,n+1)])
        
        narray=np.floor(np.true_divide((T-n)/float(logkbar),range(n,0,-1)))
        narray=np.hstack([0,narray])
        model.actions_index=actions
        countT=0
        for k in xrange(n-1):
            tempT=(n-k)*(narray[k+1]-narray[k])
            countT += tempT
            temp_eta,temp_m=model.find_eta()
            for t in xrange(tempT):
                a=np.random.choice(actions,p=temp_eta)
                x,y=model.sample(a)
                pa = model.P(x)
                r=model.R(pa,temp_eta)
                if self.truncate=="zero":
                    self.tempB=sqrt(temp_m*tempT/log(2.0*tempT*(n-k)))
                    z=(r<self.tempB)*r*y
                elif self.truncate =="clip":
                    z=np.minimum(r,self.tempB)*y
                else:
                    z=r*y
                u+=z
            u1=u/float(countT)
            u1=np.array(u1)
            tempworst=np.argmin(u1[model.actions_index])
            model.actions_index=model.actions_index.pop(tempworst)
        self.best_actions=model.actions_index
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]          
class GeneralGraph_addSR_simple(object):
    label="proposed algorithm2"
    def __init__(self,truncate = "clip"):
        self.truncate = truncate
        #self.label = "Algorithm 2-"+truncate

    def run(self,T,model):
        eta = model.eta
        n = len(eta)  
        model.actions_index=list(np.where(eta!=0)[0])
        ini_n=len(model.actions_index)
        m=model.m
        u = np.zeros(n)        

        logkbar=0.5+np.sum(np.ones(ini_n-1)/[range(2,ini_n+1)])
        
        narray=np.floor(np.true_divide((T-ini_n)/float(logkbar),range(ini_n,0,-1)))
        narray=np.hstack([0,narray])

        countT=0
        temp_eta=eta
        temp_m=m
        for k in xrange(ini_n-1):

            tempT=np.int((n-k)*(narray[k+1]-narray[k]))
            countT += tempT
            for t in xrange(tempT):
                
                a=np.random.choice(range(n),p=temp_eta)
                x,y=model.sample(a)
                pa = model.P(x)
                r=model.R(pa,temp_eta)
                if self.truncate=="zero":
                    self.tempB=sqrt(temp_m*tempT/log(2.0*tempT*(n-k)))
                    z=(r<self.tempB)*r*y
                elif self.truncate =="clip":
                    z=np.minimum(r,self.tempB)*y
                else:
                    z=r*y
                u+=z
            u1=u/float(countT)
            u1=np.array(u1)
            tempworst=np.argmin(u1[model.actions_index])
            
            model.actions_index.pop(tempworst)
            if k < n-2:
#                temp_m=model.analytic_eta_changeActions()
                temp_eta[tempworst]=0
                temp_eta=temp_eta/temp_eta.sum()
        self.best_actions=model.actions_index
        return max(model.expected_rewards) - model.expected_rewards[self.best_actions]          
    
    
class ThompsonSampling(object):
    """ Sample actions via the Thomson sampling approach and return the empirically best arm 
        when the number of rounds is exhausted """
    label = "Thompson Sampling"
    
    def run(self,T,model):
        self.trials = np.full(model.K,2,dtype=int)
        self.success = np.full(model.K,1,dtype=int)
        
        for t in xrange(T):
            fails = self.trials - self.success
            theta = np.random.beta(self.success,fails)
            arm = argmax_rand(theta)
            self.trials[arm] +=1
            self.success[arm]+= model.sample_multiple(arm,1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
class UCB(object):
    """ 
    Implements Generic UCB algorithm.
    """
    def run(self,T,model):
        if T <= model.K: # result is not defined if the horizon is shorter than the number of actions
            self.best_action = None
            return np.nan
        
        actions = range(0,model.K)
        self.trials = np.ones(model.K)
        self.success = model.sample_multiple(actions,1)
        
        for t in range(model.K,T):
            arm = argmax_rand(self.upper_bound(t))
            self.trials[arm] += 1
            self.success[arm] +=model.sample_multiple(arm,1)
        
        mu = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(mu)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
        

class AlphaUCB(UCB):
    """ Implementation based on ... """
    label = "UCB"
    
    def __init__(self,alpha):
        self.alpha = alpha
    
    def upper_bound(self,t):
        mu = np.true_divide(self.success,self.trials)
        interval = np.sqrt(self.alpha*np.log(t)/(2.0*self.trials))
        return mu+interval
        
class SuccessiveRejects(object):
    """ Implementation based on the paper 'Best Arm Identification in Multi-Armed Bandits',Audibert,Bubeck & Munos"""
    label = "Successive Reject"
    
    def run(self,T,model):
        
        if T <= model.K:
            self.best_action = None
            return np.nan
        else:
            self.trials = np.zeros(model.K)
            self.success = np.zeros(model.K)
            self.actions = range(0,model.K)
            self.allocations = self.allocate(T,model.K)
            self.rejected = np.zeros((model.K),dtype=bool)
            for k in range(0,model.K-1):
                nk = self.allocations[k]
                self.success[self.actions] += model.sample_multiple(self.actions,nk)
                self.trials[self.actions] += nk
                self.reject()
            
            assert len(self.actions) == 1, "number of arms remaining is: {0}, not 1.".format(len(self.actions))
            assert sum(self.trials) <= T,"number of pulls = {0}, exceeds T = {1}".format(sum(self.trials),T)
            self.best_action = self.actions[0]
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
    
    def allocate(self,T,K):
        logK = .5 + np.true_divide(1,range(2,K+1)).sum()
        n = np.zeros((K),dtype=int)
        n[1:] =  np.ceil((1.0/logK)*np.true_divide((T - K),range(K,1,-1)))
        allocations = np.diff(n)
        return allocations
                       
    def reject(self):       
        worst_arm = self.worst()
        self.rejected[worst_arm] = True
        self.actions = np.where(~self.rejected)[0] 
        
    def worst(self):
        mu = np.true_divide(self.success,self.trials)
        mu[self.rejected] = 2 # we don't want to reject the worst again
        min_val = np.min(mu)
        indicies = np.where(mu == min_val)[0] # these are the arms reported as worst
        return np.random.choice(indicies) # select one at random


# Some models useful for sanity checks
# -------------------------------------------------------------------------------------------------

class ObservationalEstimate(object):
    """ Just observes for all actions, and then selects the arm with the best emprical mean. 
        Assumes P(Y|do(X)) = P(Y|X) as for ParallelCausal. Some actions may be entirely unexplored. """
    label = "Observational"
    
    def run(self,T,model):
        self.trials = np.zeros(model.K)
        self.success = np.zeros(model.K)
        for t in xrange(T):
            x,y = model.sample(model.K-1)
            xij = np.hstack((1-x,x,1)) # first N actions represent x_i = 0,2nd N x_i=1, last do()
            self.trials += xij
            self.success += y*xij
        self.u = np.true_divide(self.success,self.trials)
        self.best_action = argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
class UniformSampling(object):
    label = "Uniform"
    
    def run(self,T,model):
        trials_per_action = T/model.K
        success = model.sample_multiple(range(model.K),trials_per_action)
        self.u = np.true_divide(success,trials_per_action)
        self.best_action = argmax_rand(self.u)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action]
        
class RandomArm(object):
    label = "Random arm"
    
    def run(self,T,model):
        self.best_action = np.random.randint(0,model.K)
        return max(model.expected_rewards) - model.expected_rewards[self.best_action] 

