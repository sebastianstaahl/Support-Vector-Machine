import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def objective(alpha):
	global PMatrix
	tempo_vector = numpy.dot(alpha,PMatrix) # Ger oss en vektor med längden N
	tempo_scalar = numpy.dot(alpha,tempo_vector) # Ger oss en scalar
	right_scalar = numpy.sum(alpha)
	scalarValue = 1/2*tempo_scalar - right_scalar
	#print("Scalarvalue;		",scalarValue)
	return scalarValue


def zerofun(alpha):
	global targets
	return numpy.dot(targets,alpha)

def b_value():
	global b
	sv = non_zero_dp[0]
	print("sv:		",sv)
	#kernalValue = polynomialKernal(inputs, sv)
	kernalValue = polynomialKernal(non_zero_dp,sv)

	value = numpy.dot(non_zero_a, non_zero_t)
	numpy_product = numpy.dot(value, kernalValue)
	numpy_sum = numpy.sum(numpy_product)
	b = numpy_sum - non_zero_t[0]
	print("s:		",non_zero_t[0])
	print("b:   ", b)

def b_value_3():
	global b
	s = non_zero_dp[0]
	st = non_zero_t[0]
	b=0
	print("s:		",s)
	print("st:		",st)

	for i in range(len(non_zero_dp)):
		b+=non_zero_a[i]*non_zero_t[i]*polynomialKernal(non_zero_dp[i],s)
	b= b-st
	print("b:		",b)

def b_value_2():
	global b
	sv = non_zero_dp[0]
	kernalValue = polynomialKernal(non_zero_dp,sv)
	print("KerkernalValue:		", non_zero_dp)
	sum=0
	for i in range(len(non_Zero_all)):
		#c+=non_Zero_all[i][0]*non_Zero_all[i][2]*polynomialKernal(sv,non_Zero_all[i][1])
		sum+=non_Zero_all[i][0]*non_Zero_all[i][2]*kernalValue[i]
	b=sum-non_zero_t[0]

	print("c:   ", b)




	#sum=0
	#for i in range(len(non_zero_dp)):

	#b = sum-non_zero_t[0]
	#print("b_alternative:		", b)

	#kernalValue_2 = polynomialKernal(sv,non_zero_dp[0])
	#kernalvalue_3 = polynomialKernal(non_zero_dp,sv) # Åt detta håll får man alla kernelvärden



def indicatorFunction_2(x,y):
	global targets
	global inputs
	global non_zero_a
	global non_zero_t
	global non_zero_dp
	global b
	global non_Zero_all

	dp = numpy.array([x,y])
	kernalValue = polynomialKernal(non_zero_dp, dp)
	value = numpy.dot(non_zero_a, non_zero_t)
	ind = numpy.dot(value,kernalValue)
	sum = numpy.sum(ind)
	indication = sum - b
	print(indication)
	return indication

def indicatorFunction(x,y):
	sum = 0
	for i in range (len(non_zero_a)):
		sum+=non_zero_a[i]*non_zero_t[i]*polynomialKernal(non_zero_dp[i],[x,y])
	sum2 = sum-b
	print(sum2)
	return sum2

def extract(alpha):
	global non_zero_a, non_zero_t, non_zero_dp
	global non_Zero_all
	non_Zero_all=[]
	non_zero_a=[]
	non_zero_t=[]
	non_zero_dp=[]

	for i in range(len(alpha)):
		if abs(alpha[i])>10e-5:
			non_zero_a.append(alpha[i])
			non_zero_t.append(targets[i])
			non_zero_dp.append([inputs[i][0], inputs[i][1]])
			non_Zero_all.append([alpha[i], [inputs[i][0], inputs[i][1]] ,targets[i]])
	print(non_Zero_all)



def polynomialKernal(dp1, dp2):
	return numpy.dot(dp1, dp2) + 1


def polynomialKernal(dp1, dp2, power = 3):
	return numpy.power((numpy.dot(dp1, dp2) + 1), power)


def radial_kernel(dp1, dp2, sigma=2):
    diff = numpy.subtract(dp1, dp2)
    return math.exp((-numpy.dot(diff, diff)) / (2 * sigma * sigma))


def calculatePMatrix():
	global PMatrix
	PMatrix = numpy.zeros(shape=(N,N))
	for i in range(N):
		for j in range(N):
			PMatrix[i][j] = targets[i]*targets[j]*polynomialKernal(inputs[i], inputs[j])


    #print("PMATRIX:	", PMatrix)


################################################################################
#Generating data
################################################################################
def generateData():
    #översta rutan på s8
    global classA, classB, inputs, targets, N
    numpy.random.seed(200) #fast seed
    classA = numpy.concatenate (
    (numpy.random.randn(10 , 2) * 0.2 + [2 , 0.5],
    numpy.random.randn(10 , 2) * 0.2 + [-2, 0.5]))
    #classA = numpy.random.randn(30 , 2) * 0.2 + [-0.25, 0.75]

    classB = numpy.random.randn(30,2)*0.2 + [0.0, 0.0]

    inputs = numpy.concatenate((classA, classB))
    #print(inputs)
    targets = numpy.concatenate((numpy.ones(classA.shape[0]),
                                 -numpy.ones(classB.shape[0])))

    N = inputs.shape[0]
    permute = list(range(N))
    numpy.random.shuffle(permute)
    inputs =inputs[permute, :]
    targets = targets[permute]
    ################################################################################
    #print("ClassB:      ",inputs)


def plottingData():
	xgrid=numpy.linspace(-5, 5)
	ygrid=numpy.linspace(-5, 5)
	grid=numpy.array([[indicatorFunction(x, y) for x in xgrid ] for y in ygrid])
	#grid=numpy.array([[indicatorFunction_2(x, y) for x in xgrid ] for y in ygrid])
	plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))



	plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
	plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
	plt.axis('equal') # Force same scale on both axes
	plt.savefig('svmplot.pdf', dpi=1000) # Save a copy in a file
	plt .show() # Show the plot on the screen





def main():
	C=10000
	generateData()
	calculatePMatrix()
	start = numpy.zeros(N) #Initial guess ok!
	B = [(0, C) for b in range(N)]
	XC = {'type':'eq', 'fun':zerofun}
	ret = minimize(objective, start, bounds=B, constraints=XC)
	alpha=ret["x"]
	print(ret)
	#print(ret['success'])
	extract(alpha)
	b_value()
	plottingData()
	#plotDB()
	#plotData()



if __name__ == "__main__":
	main()
