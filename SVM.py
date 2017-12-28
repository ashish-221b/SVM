import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')
class SVM:
	def __init__(self, Xdata, Ydata, C):
		self.Xdata = Xdata
		self.Ydata = Ydata  #row vector of result
		self.trainlen = len(Ydata)  # no of trainin examples
		self.dim = Xdata.shape[1]	# dimensionality of training set or (features)
		self.alphas = np.zeros(self.trainlen) # lagrange multipliers (row vector)
		self.b = np.int64(0) # threshold
		self.w = np.zeros(self.dim) # normal to hyperplane 
		self.C=C
		self.E = np.zeros(self.trainlen)
		self.steps = 0
	def info(self):
		# print("number of training examples")
		print self.trainlen
		print("dimensionality of dataset")
		print self.dim
	def process(self):
		for i in range(0,self.trainlen):
			if self.Ydata[i]==0:
				self.Ydata[i] = -1
	def result_print(self):
		print("W")
		print self.w
		print("b")
		print self.b
	def update_w(self,diff1,diff2,i1,i2):
		# self.w = np.dot(np.multiply(self.alphas,self.Ydata),self.Xdata)
		# print np.dot(self.Ydata,self.Xdata)
		self.w = self.w + self.Ydata[i1]*diff1*self.Xdata[i1] + self.Ydata[i2]*diff2*self.Xdata[i2]
	def Err(self,i):
		return np.dot(self.w,self.Xdata[i]) - self.b - self.Ydata[i]
	def update_error(self):
		for i in range(0,self.trainlen):
			self.E[i]=self.Err(i)
	def update_b(self,diff1,diff2,i1,i2):
		b1 = self.E[i1] + self.Ydata[i1]*diff1*np.dot(self.Xdata[i1],self.Xdata[i1])+self.Ydata[i2]*diff2*np.dot(self.Xdata[i2],self.Xdata[i2])+self.b
		b2 = self.E[i2] + self.Ydata[i1]*diff1*np.dot(self.Xdata[i1],self.Xdata[i1])+self.Ydata[i2]*diff2*np.dot(self.Xdata[i2],self.Xdata[i2])+self.b
		if (self.alphas[i1]<self.C) and (self.alphas[i1]>0):
			self.b=b1
		elif (self.alphas[i2]<self.C) and (self.alphas[i2]>0):
			self.b=b2
		else:
			self.b=(b1+b2)/2
	def Eval_Objective_at(self,i1,i2,s,a):
		x1 = self.Xdata[i1]
		x2 = self.Xdata[i2]
		alpha1 = self.alphas[i1]
		alpha2 = self.alphas[i2]
		f1 = self.Ydata[i1]*(self.E[i1]+self.b)-alpha1*np.dot(x1,x1)-s*alpha2*np.dot(x1,x2)
		f2 = self.Ydata[i2]*(self.E[i2]+self.b)-s*alpha1*np.dot(x1,x2)-alpha2*np.dot(x2,x2)
		a1 = self.alpha1+s*(self.alpha2-a)
		phi_a = a1*f1+a*f2+.5*(a1*a1*np.dot(x1,x1))+.5*(a*a*np.dot(x2,x2))+s*a*a1*np.dot(x1,x2)
		return phi_a
	def optimise_step(self,i1,i2):
		eps = .001
		if (i1 == i2):
			return False
		alp1 = self.alphas[i1]
		alp2 = self.alphas[i2]
		y1 = self.Ydata[i1]
		y2 = self.Ydata[i2]
		E1 = self.E[i1]
		E2 = self.E[i2]
		s = y1*y2
		if(y1==y2):
			L2 = max(0,alp2+alp1-self.C)
			H2 = min(self.C,alp1+alp2)
		else:
			L2 = max(0,alp2-alp1)
			H2 = min(self.C,self.C+alp2-alp1)
		if (H2==L2):
			# print("fail1")
			# print(L2)
			# print(H2)
			return False
		eta = np.dot(self.Xdata[i1],self.Xdata[i1]) + np.dot(self.Xdata[i2],self.Xdata[i2]) - 2*np.dot(self.Xdata[i1],self.Xdata[i2])
		if (eta>0):
			# print("hash")
			a2 = alp2 + y2*(E1-E2)/eta
			if(a2<L2):
				a2=L2
			elif(a2>H2):
				a2=H2
			# print a2
			# print alp2
			# print eta
			# print E1-E2
			# print y2
		else:
			Lobj = self.Eval_Objective_at(i1,i2,s,L2)
			Hobj = self.Eval_Objective_at(i1,i2,s,H2)
			if (Lobj<Hobj-eps):
				a2=L
			elif (Lobj>Hobj+eps):
				a2=H
			else:
				a2=alp2
		if (abs(a2-alp2)<eps*(a2+alp2+eps)):
			# print("fail2")
			return False
		a1 = alp1+s*(alp2-a2)
		self.update_b(a1-alp1,a2-alp2,i1,i2)
		self.update_w(a1-alp1,a2-alp2,i1,i2)
		# print self.w
		self.update_error()
		self.E[i1]=0
		self.E[i2]=0
		self.alphas[i1]=a1
		self.alphas[i2]=a2
		# print("Pass")
		self.steps = self.steps+1
		# print self.steps
		# print i1
		# print i2
		return True
	def second_heur(self,i2):
		val = 0
		i1 = 0
		for i in range(0,self.trainlen):
			if (i==i2):
				continue
			else:
				if (abs(self.E[i]-self.E[i2])>val):
					val=abs(self.E[i]-self.E[i2])
					i1=i
		return i1
	def examineExample(self,i2):
		eps = .001
		y2 = self.Ydata[i2]
		alph2 = self.alphas[i2]
		E2 = self.E[i2]
		r2 = E2*y2
		if ((r2 < -eps) and (alph2 < self.C)) or ((r2 > eps) and (alph2 > 0)):
			# i1 = self.second_heur(i2)
			# if (self.optimise_step(i1,i2)):
			# 	return 1
			for i in range(0,self.trainlen):
				if (self.alphas[i]<self.C) and (self.alphas[i]>0):
					if (self.optimise_step(i,i2)):
						return 1
			for i in range(0,self.trainlen):
				if (self.optimise_step(i,i2)):
					return 1
		return 0
	def train(self):
		self.process()
		self.update_error()
		numchanged = 0
		examineAll = 1
		while (numchanged>0) or (examineAll == 1):
			numchanged=0
			if (examineAll):
				for i in range(0,self.trainlen):
					numchanged = numchanged + self.examineExample(i)
			else :
				for i in range(0,self.trainlen):
					if (self.alphas[i]<self.C) and (self.alphas[i]>0):
						numchanged = numchanged + self.examineExample(i)
			if (examineAll==1):
				examineAll = 0
			elif (numchanged == 0):
				# print("hmmm")
				examineAll = 1
			if self.steps > 2000:
				break
	