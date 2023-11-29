import sys
import numpy as np

fileName = sys.argv[1]
gaussCount = int(sys.argv[2])
iterCount = int(sys.argv[3])

def calcProb(val, distro): # formula from the slides
    base = np.sqrt(2 * np.pi)
    base *= np.sqrt(distro.var)
    base **= -1
    exponent = -1/2 * ((val - distro.avg) ** 2) / distro.var
    return base * np.exp(exponent)

class Datum:
    def __init__(self, value, distros):
        self.value = value # what it is
        self.probs = [0] * len(distros) # probability list
        for p in range(len(self.probs)):
            self.probs[p] = calcProb(self.value, distros[p]) * distros[p].prior# probability of appearing in each gaussian
        scale_factor = 1.0 / sum(self.probs)
        for p in range(len(self.probs)):
            self.probs[p] *= scale_factor
            # self.probs[p] = round(self.probs[p], 4)
class Gaussian:
    def __init__(self, data, pc):
        self.pointCount = pc
        self.avg = np.mean(data)  # ez mean
        self.var = np.var(data)  # ez variance
        self.prior = len(data)/self.pointCount


class Points: # The master class for managing everything
    def __init__(self, gc, gi, ap):
        self.distros = []  # will contain Gaussian objects
        self.everything = ap  # contains all points, will be useful for iterating(?)
        self.data = []  # will contain Datum objects
        for g in range(gc):
            self.distros.append(Gaussian(gi[g], len(ap)))  # adding a Gaussian for each in gaussCount
        for a in self.everything:
            self.data.append(Datum(a, self.distros)) # calculating initial probabilities
    def updateGaussians(self):
        for G in range(len(self.distros)):
            priorP = 0
            for P in self.data:
                priorP += P.probs[G]

            mu1 = 0
            mu2 = 0
            sigma1 = 0
            sigma2 = 0

            for P in self.data:
                mu1 += P.probs[G] * P.value
                mu2 += P.probs[G]
            self.distros[G].avg = mu1/mu2

            for P in self.data:
                sigma1 += P.probs[G] * (P.value - self.distros[G].avg)**2
                sigma2 += P.probs[G]
            self.distros[G].var = sigma1/sigma2


            priorP /= len(self.everything)
            self.distros[G].prior = priorP

        for P in range(len(self.data)):
            self.data[P] = Datum(self.data[P].value, self.distros)

gaussInput = {}  # dictionary
allPoints = []  # list
for g in range(gaussCount):
    gaussInput[g] = []
with open(fileName, 'r') as file:
    ctr = 0
    for line in file:
        allPoints.append(float(line))
        gaussInput[ctr % gaussCount].append(float(line))
        ctr += 1

pts = Points(gaussCount, gaussInput, allPoints)

print("After iteration 0:")
for d in range(len(pts.distros)):
    m = pts.distros[d].avg
    v = pts.distros[d].var
    p = pts.distros[d].prior
    print(f"Gaussian {d+1}: mean = {m:.4f}, variance = {v:.4f}, prior = {p:.4f}")
print()

for i in range(iterCount):
    print(f"After iteration {i+1}")
    pts.updateGaussians()
    for d in range(len(pts.distros)):
        m = pts.distros[d].avg
        v = pts.distros[d].var
        p = pts.distros[d].prior
        print(f"Gaussian {d+1}: mean = {m:.4f}, variance = {v:.4f}, prior = {p:.4f}")
    print()