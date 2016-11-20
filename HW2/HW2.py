from itertools import permutations
from itertools import combinations
from itertools import product

class ConditionalProb(object):
	def __init__(self, inputFile):
		self.prob = dict()
		f = open(inputFile)
		self.item = f.readline().strip().split('\t')
		self.item.pop()
		self.N = len(self.item)
		self.independences = list()
		for line in f.readlines():
			tmp = line.strip().split('\t')
			self.prob[tuple(map(eval,tmp[0:self.N]))] = float(tmp[self.N])
		f.close()

	#get prob or marginal prob from the table
	def get_prob(self, **kw):
		if len(kw) > self.N:
			print("invalid input")
		elif len(kw) == self.N:
			return self.prob[tuple([kw[i] for i in self.item])]
		else:
			margin = list(set(self.item) - set(kw.keys()))
			prob = 0
			for marginElement in product([0, 1], repeat = len(margin)):
				tmp = {margin[i]:marginElement[i] for i in range(len(margin))}
				tmp.update(kw)
				prob += self.get_prob(**tmp)
			return prob

	#calculate conditional_prob P(x | condition)
	def cal_conditional_prob(self, x, condition):
		tmp = dict(x)
		tmp.update(condition)
		return self.get_prob(**tmp) / self.get_prob(**condition)

	#check whether P(node1 | node2, condition) = P(node1 | condition), i.e. node1 
	#and node2 are conditionaly independent, if independent return False.
	def check_Dependence(self, node1, node2, condition):
		node1 = list(node1)
		node2 = list(node2)
		condition = list(condition)
		for node2Element in product([0, 1], repeat = len(node2)):
			tmp2 = {node2[i]:node2Element[i] for i in range(len(node2))}
			for node1Element in product([0, 1], repeat = len(node1)):
				tmp1 = {node1[i]:node1Element[i] for i in range(len(node1))}
				for conditionElement in product([0, 1], repeat = len(condition)):
					tmpCondition = {condition[i]:conditionElement[i] for i in range(len(condition))}
					tmp = dict(tmp2)
					tmp.update(tmpCondition)
					p1 = self.cal_conditional_prob(tmp1, tmp)
					p2 = self.cal_conditional_prob(tmp1, tmpCondition)
					if (p1 - p2) > 1e-6:
						return True
		return False

	#check independences like (X / Y | condition), where X, Y are single nodes.
	def get_All_Independences(self):
		for (x, y) in combinations(self.item, 2):
			restNodes = set(self.item) - set((x, y))
			for numOfCondtions in range(len(restNodes) + 1):
				conditions = combinations(restNodes, numOfCondtions)
				for condition in conditions:
					if not self.check_Dependence(set(x), set(y), set(condition)):
						self.independences.append((x, y, condition))

class Imap(object):
	def __init__(self, orderNodes, cProb):
		self.cProb = cProb
		self.vertex = orderNodes
		self.edge = list()

	def build_mininal_Imap(self):
		for i in range(1, len(self.vertex)):
			potentials = set(self.vertex[0:i])
			foundFlag = False
			parentSet = None
			for numParent in range(len(potentials) + 1):
				if foundFlag: 
					break
				parents = combinations(potentials, numParent)
				for parent in parents:
					if not self.cProb.check_Dependence(set(self.vertex[i]), potentials - set(parent), set(parent)):
						foundFlag = True
						parentSet = parent
			if parentSet is not None:
				for parent in parentSet:
					self.edge.append((parent, self.vertex[i]))

if __name__ == '__main__':
	cProb = ConditionalProb("4-table2.txt")
	cProb.get_All_Independences()
	print('All independences implied in P:')
	for x in cProb.independences:
		print('(' + x[0] + ';' + x[1] + '|' + str(x[2]) + ')')
	numOfEdge = 10000
	order = None
	edge = None
	for orderNodes in permutations(cProb.item, 4):
		imap = Imap(orderNodes, cProb)
		imap.build_mininal_Imap()
		if len(imap.edge) < numOfEdge:
			numOfEdge = len(imap.edge)
			order = orderNodes
			edge = imap.edge
	print('The ordering nodes that give the fewest edges ' + str(order) + str(len(edge)) + 'edges')
	print('Edges in the I-map')
	for e in edge:
		print(e[0] + ' -> ' + e[1])