#used functions

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def extract_features_and_labels(train_file):
    '''
    This function extracts the features and labels from the training and development dataset 
    
    :param inputfile: filepath to either training dataset or development dataset
    :type inputfile: a string with the filepath 
    
    :returns: a dict of the features and a list of the labels 
    '''
    
    data = []
    targets = []

    for line in open(train_file, encoding='utf-8'):
        content = line.split(',')
        if len(content) > 10:
            token = content[2]
            lemma = content[3]
            pos_univ = content[4]
            pos = content[5]
            prev_pos = content[6]
            next_pos = content[7]
            morph = content[8]
            head = content[9]
            basic_dep = content[10]
            enh_dep = content[11]
            predicate = content[13]

            # create dict
            feature_dict = {'token': token,
                            'lemma': lemma,
                            'pos_univ': pos_univ,
                            'pos': pos,
                            'prev_pos': prev_pos,
                            'next_pos': next_pos, 
                            'morph': morph, 
                            'head': head,
                            'basic_dep': basic_dep,
                            'enh_dep': enh_dep,
                            'predicate': predicate}

            gold_label = list(content[14:])

            data.append(feature_dict)
            targets.append(gold_label)
            
    mlb = MultiLabelBinarizer()
    targets = mlb.fit_transform(targets)
    
        
    return data, targets
   
def extract_features(test_file):
    '''
    This function extracts the features from the test dataset 
    
    :param inputfile: filepath to either training dataset or development dataset
    :type inputfile: a string with the filepath 
    
    :returns: a dict of the features and a list of the labels 
    '''

    data = []

    for line in open(train_file, encoding='utf-8'):
        content = line.split(',')
        if len(content) > 10:

            # define columns
            token = content[2]
            lemma = content[3]
            pos_univ = content[4]
            pos = content[5]
            prev_pos = content[6]
            next_pos = content[7]
            morph = content[8]
            head = content[9]
            basic_dep = content[10]
            enh_dep = content[11]
            predicate = content[13]    

            # create dict
            feature_dict = {'token': token,
                            'lemma': lemma,
                            'pos_univ': pos_univ,
                            'pos': pos,
                            'prev_pos': prev_pos,
                            'next_pos': next_pos, 
                            'morph': morph, 
                            'head': head,
                            'basic_dep': basic_dep,
                            'enh_dep': enh_dep,
                            'predicate': predicate}

            data.append(feature_dict)
        
    return data

def classify_data(model, vec, input_file, outputfile):
    
    features = extract_features(input_file)
    vec_features = vec.transform(features)
    predictions = model.predict(vec_features)

    outfile = open(outputfile, 'w')
    counter = 0
                           
    for line in open(test_file):
        outfile.write(line.rstrip('\n') + ',' + predictions[counter] + '\n')
        counter += 1
                           
    outfile.close()
    
def run_machine_learning_models(train_file, test_file, outputfile, model):
    
    features, gold_labels = extract_features_and_labels(train_file)
    ml_model, vec = create_classifier(features, gold_labels, model)
            
    print("Loading...", model)
    classify_data(ml_model, vec, test_file, outputfile)
    print("Method", model, "is done!")

    return gold_labels

def create_classifier(features, targets, modelname):
    '''
    Function that takes feature-value pairs and gold labels as input and trains a logistic regression classifier
    
        :param features: feature-value pairs
        :param targets: gold labels
        :type features: a list of dictionaries
        :type targets: a list of strings
        :param modelname: ml modelname to execute
        :type modelname: string
        :return model: a trained classifier
        :return vec: a DictVectorizer to which the feature values are fitted. 
    '''
  
    if modelname == "logreg":
        model = LogisticRegression()
  
    vec = DictVectorizer()
    features_vectorized = vec.fit_transform(features)
    model = model.fit(features_vectorized, targets)

    return model, 