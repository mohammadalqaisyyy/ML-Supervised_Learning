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

p = input("Enter point to predict (format X,Y) : ").split(',')
point = [float(p[0]),float(p[1])]
print("SVC predict : ",my_model.predict([point]))

plt.scatter(my_in[:,0],my_in[:,1],c=my_out)
plt.scatter(point[0],point[1],c='r')
plt.show()

print("Done")