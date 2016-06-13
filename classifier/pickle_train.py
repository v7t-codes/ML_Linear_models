#to_classify_the_given_images_from_pickled_file_containing_pickled_training_and_testing_data_(in_this_case_notMNIST_data_is used)

from six.moves import cPickle as pickle 
from sklearn import neighbors, linear_model


knn = neighbors.KNeighborsClassifier()
logit = linear_model.LogisticRegression()

file_address= list() #address of the pickled file containing pasterised images

def train_test(file_address):
	f=open(string)
	loaded_file=pickle.load(f)
        print len(loaded_file)
	#extract the training data and labels and testing data and labels from the saved data set
	train_dataset=loaded_file['train_dataset']
	train_labels =loaded_file['train_labels']  
	test_dataset=loaded_file['test_dataset']
	test_labels=loaded_file['test_labels']
#shuffle the training data to feed it into a required learning model
def shuffle_data(data,labels):
	permute = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permute,:,:]
	shuffled_labels = labels[permute]
	return shuffled_dataset, shuffled_labels

#shuffle the training data and test data
train_set,train_labels = shuffle_data(train_dataset,train_labels)
test_set, train_lables =shuffle_data(test_dataset, test_labels)

#calculate the efficiency of the model by training over various numbers no training samples 
print('KNN_score ---> %f' % knn.fit(train_set, train_labels).score(test_set, test_labels))
print('Logit_score---> %f'
      % logistic.fit(train_dataset, train_labels).score(test_dataset, test_labels))
