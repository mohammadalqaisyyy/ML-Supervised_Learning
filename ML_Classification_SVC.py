import numpy
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")

size = int(input("Number of points: "))
input_list=[]
my_out=[]
point=[]

print("Format: X,Y,OUTPUT")
for i in range (size):
    point=input().split(',')
    input_list.append([float(point[0]),float(point[1])])
    my_out.append(int(point[-1]))

my_in=numpy.array(input_list)


my_model=svm.SVC(kernel='linear',C=1.0)
my_model.fit(my_in,my_out)

print("SVC predict[0.5,0.8] : ",my_model.predict([[0.5,0.8]]))
print("SVC predict[0.5,0.8] : ",my_model.predict([[8.5,10]]))

plt.scatter(my_in[:,0],my_in[:,1],c=my_out)
plt.scatter(0.5,0.8,c='r')
plt.scatter(8.5,10,c='r')
plt.show()

print("Done")