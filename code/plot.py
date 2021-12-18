import matplotlib.pyplot as matlib

def extract(name):
	x = []
	y = []

	file = open(name, "r");
	lines = file.readlines()
	file.close()

	for line in lines:
		loss, runtime = list(map(float, line.split(",")))
		y.append(loss)
		x.append(runtime)

	# x.pop(0)
	# y.pop(0)

	return (x, y)

def finishPlot(name, header, params, labelx, labely):
	matlib.title(header)
	matlib.ylabel(labely)
	matlib.xlabel(labelx)
	matlib.legend(params)
	matlib.savefig(name)
	matlib.close()

def plot():

	size = 50
	for i in range(6):
		x, y = extract("method1size" + str(size) + ".csv")
		matlib.plot(x, y, marker = 'o')
		size = size * 2

	finishPlot("plot1.png", "Constant LR With Varying Batch Size", [50, 100, 200, 400, 800, 1600], "Runtime in secs", "Loss")

	size = 50
	for i in range(6):
		x, y = extract("method2size" + str(size) + ".csv")
		matlib.plot(x, y, marker = 'o')
		size = size * 2

	finishPlot("plot2.png", "Adaptive LR With Varying Batch Size", [50, 100, 200, 400, 800, 1600], "Runtime in secs", "Loss")


	size = 50
	for i in range(6):
		if size == 400:
			size = size * 2
			continue
		x, y = extract("method3size" + str(size) + ".csv")
		matlib.plot(x, y, marker = 'o')
		size = size * 2

	finishPlot("plot3.png", "Backtracking Line Search LR With Varying Batch Size", [50, 100, 200, 800, 1600], "Runtime in secs", "Loss")

	etas = [5, 10]
	for eta in etas:
		x, y = extract("method2eta" + str(eta) + ".csv")
		matlib.plot(x, y, marker = 'o')
	
	finishPlot("plot4.png", "Adaptive LR For Different Eta", etas, "Runtime in secs", "Loss")

	alphas = [0.1, 0.2, 0.3, 0.4]
	for alpha in alphas:
		x, y = extract("method3alpha" + str(alpha) + ".csv")
		matlib.plot(x, y, marker = 'o')
	
	finishPlot("plot5.png", "Backtracking Line Search For Different Alpha", alphas, "Runtime in secs", "Loss")

	betas = [0.1, 0.3, 0.5, 0.7, 0.9]
	for beta in betas:
		x, y = extract("method3beta" + str(beta) + ".csv")
		matlib.plot(x, y, marker = 'o')
	
	finishPlot("plot6.png", "Backtracking Line Search For Different Beta", betas, "Runtime in secs", "Loss")

plot()




