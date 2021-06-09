import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    sumd = np.zeros(y.shape)
    for r1, c1, r2, c2 in predictors:
	    diff = X[:,r1,c1] - X[:,r2,c2]
	    diff[diff > 0] = 1
	    diff[diff < 0] = 0
	    sumd += diff
    mean = np.divide(sumd,len(predictors))
    mean[mean > 0.5] = 1
    mean[mean <= 0.5] = 0
    return fPC(y, mean)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = []
    for i in range(5):
	    bestAccuracy = 0
	    bestLocation = None
	    for r1 in range(24):
		    for c1  in range(24):
			    for r2  in range(24):
				    for c2  in range(24):
					    if (r1,c1) == (r2,c2) or (r1,c1,r2,c2) in predictors:
						    continue
					    currAccuracy = measureAccuracyOfPredictors(predictors +  list(((r1,c1, r2,c2),)), trainingFaces,trainingLabels)
					    if currAccuracy > bestAccuracy:
						    bestAccuracy = currAccuracy
						    bestLocation = (r1,c1, r2,c2)
	    predictors.append(bestLocation)
    r1, c1, r2, c2 = bestLocation
    return predictors

def showFeatures(predictors, testingFaces):
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    for r1, c1, r2, c2 in predictors:
        # Show r1,c1
	    rect = patches.Rectangle((c1,r1),1,1,linewidth=2,edgecolor='r',facecolor='none')
	    ax.add_patch(rect)
	    # Show r2,c2
	    rect = patches.Rectangle((c2,r2),1,1,linewidth=2,edgecolor='b',facecolor='none')
	    ax.add_patch(rect)
    # Display the merged result
    plt.show()

def trainModel(trainingFaces,trainingLabels,testingFaces, testingLabels):
    samples = [400]#[400,800,1200,1600,2000]
    predictors = []
    for sample in samples:
	    print("\nTraining Model with {} image samples:".format(sample))
	    predictors = stepwiseRegression(trainingFaces[:sample],trainingLabels[:sample],testingFaces,testingLabels)
	    print("  Predictors selected:\n    {}".format(predictors))
	    accuracy = measureAccuracyOfPredictors(predictors,trainingFaces,trainingLabels)
	    print("  Training accuracy on {} samples of training data = {}".format(sample, accuracy))
	    accuracy = measureAccuracyOfPredictors(predictors,testingFaces,testingLabels)
	    print("  Testing accuracy on entire test set = {}".format(accuracy))
    print("\n-----------------------------------------")
    print("Training completed. Final results:")
    print("  Final accuracy of model = {}".format(accuracy))
    print("  Final predictors selected:\n    {}".format(predictors))
    showFeatures(predictors,testingFaces)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    trainModel(trainingFaces,trainingLabels,testingFaces,testingLabels)