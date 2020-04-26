###   MACHINE LEARNING ALGORITHM PYTHON CODE  ####

--------------------------------------------------------------------------------------------

############################ Genetic Algorithm ######################################

import numpy
import random

#find the sum of 5 weights w1, w2, w3, w4 , w5 such that x1w1 + x2w2 + x3w3 + x4w4 + x5w5 = N, N = some number (Integer)

MUTATION_PROBABILITY = 0.04 # initial value

X = numpy.array([1, 2, 3, 4])

DECAY_RATE = 0.003 # reduce mutation probability , so that changes of mutation are less at endings

N = 30

NB_MOST_FIT = 4

class Member : 

    def __init__(self, genes, mutation_probability) :
        
        self.genes = genes
        self.nb_genes = len(self.genes)

        if mutation_probability > random.random() : 

            #select any 3 random genes for mutation :
            random_genes = numpy.random.randint(low = 0, high = self.nb_genes, size = (4))

            for i in random_genes :

                self.genes[i] = self.genes[i] + random.randint(a = 1, b = 50)


    def fitness_score(self, N) :
        LHS = numpy.matmul(X.T , self.genes)
        if(LHS==N):
        	print(X.T,self.genes,LHS,numpy.fabs(LHS - N))
        return numpy.fabs(LHS - N)
    
    def __str__(self):

        return "Member "+str(self.genes) + " LEAST ERROR  : "+str(self.fitness_score(N))
    

class Population : 

    #a population is a set of members, We limit population size to 5, so there are 5 members
    #select ration is 3 : 2 while crossover
    def __init__(self, nb_popsize = 10, nb_genesize = 4, select = 3, callback = None) :

        self.nb_popsize = nb_popsize
        self.nb_genesize = nb_genesize
        self.callback = callback

        self.population = self.create()
    
    def create(self):

        #creates a pouplation initially with random values for genes : 
        members = []
        for i in range(self.nb_popsize) :
            genes = numpy.random.randint(low = 1, high = 30, size = (self.nb_genesize))
            member = Member(genes, mutation_probability = MUTATION_PROBABILITY)
            members.append(member)
        
        return members

    
    def grade(self) :

        #obtain fitness scores and returns most fit members from population : 
        #and one unfit member

        fit_members = []
        temp = []
        for i in range(self.nb_popsize) :
            fitness, member = self.population[i].fitness_score(N), i
            temp.append((fitness, i))

        temp.sort()

        #get last 4 members who are most fit
        for p_index in range(NB_MOST_FIT) :

            i = temp[p_index][1]

            fit_members.append(self.population[i])

        #and add one unfit member : 
        rand_unfit = random.randint(a = NB_MOST_FIT, b = len(self.population) - 1)
        fit_members.append(self.population[temp[rand_unfit][1]])
        
        return fit_members
    
    def crossover_policy(self, dad, mom) :

        #Simple crossover scheme, use your own techniques for optimization: 
        child_genes = []
        for i in range(len(dad.genes)) :
            gene = None
            if i % 2 == 0 :
                gene = dad.genes[i]
            else : 
                gene = mom.genes[i]

            child_genes.append(gene)
        
        return Member(numpy.array(child_genes), mutation_probability = MUTATION_PROBABILITY)

    

    def crossover(self, old_fit_members) :

        #there is a requirement of some children, allow parents to reproduce
        requirement = self.nb_popsize - len(old_fit_members)

        #generate random pairs and mutate them :
        children = []
        while requirement > 0 :
            dad = random.randint(a = 0, b = len(old_fit_members) - 1)
            mom = random.randint(a = 0, b = len(old_fit_members) - 1)

            #print(mom, dad)

            if dad != mom : 
                child = self.crossover_policy(old_fit_members[dad], old_fit_members[mom])
                requirement = requirement - 1
                children.append(child)
        
        new_generation = old_fit_members + children
        return new_generation
    
    def next_generation(self) :

        global MUTATION_PROBABILITY

        fits = self.grade()
        population = self.crossover(fits)
        self.population = population
        

#test

population = Population()

NB_EPH = 1000
for i in range(NB_EPH) :
    population.next_generation()


#print optimal solutions : 

print('Population after '+ str(NB_EPH) +' iterations : ')

scores = []

for p in population.population : 
    scores.append(p.fitness_score(N))
    print(p)

print('Error factor : (+ or - error will give the solution) ', numpy.amin(scores))


------------------------------------------------------------------------------------