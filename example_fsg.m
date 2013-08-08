

%%

% An example for data classification using Fuzzy Stacked Generalization (FSG)

clear all
clc
% You can download a sample Corel Dataset from:
% https://github.com/meteozay/Corel_Dataset.git

% Load Color Layout Features of the samples of Corel Dataset:

load('color_layout.mat')

% Load Edge Histogram Features of the samples of Corel Dataset:

load('edge_histogram.mat')

% Load Region Shape Features of the samples of Corel Dataset:

load('region_shape.mat')

% Define number of classes:

class_num=3;

% Random selection of class indices:

class_ids=randperm(max(labels),class_num);

% Randomly select samples corresponding to the selected classes to construct a
% dataset, and split the dataset into training and test samples:

% Set the data splitting ratio=number of training samples/number of samples
% in the dataset;

ratio=0.5;
tr_data_ids=[];
te_data_ids=[];

for i=1:class_num
    [temp_ids]=find(labels==class_ids(i));
    
    tr_size=round(size(temp_ids,1)*ratio);
    te_size=size(temp_ids,1)-tr_size;
    
    c_tr_idx=randperm(size(temp_ids,1),tr_size);
    c_te_idx=setdiff(1:size(temp_ids,1),c_tr_idx);    

    tr_data_ids=[tr_data_ids; temp_ids(c_tr_idx)];
    te_data_ids=[te_data_ids; temp_ids(c_te_idx)];
end

labels_tr=labels(tr_data_ids);
labels_te=labels(te_data_ids);

% Define the features that will be used in the dataset:

feature_names={'color_layout','edge_histogram','region_shape'};

features_tr.color_layout=color_layout(tr_data_ids,:);
features_te.color_layout=color_layout(te_data_ids,:);

features_tr.edge_histogram=edge_histogram(tr_data_ids,:);
features_te.edge_histogram=edge_histogram(te_data_ids,:);

features_tr.region_shape=region_shape(tr_data_ids,:);
features_te.region_shape=region_shape(te_data_ids,:);


% Set k values of base-layer and meta-layer fuzzy k-NN classifiers:

k_list_base=10*ones(size(feature_names,2),1); % List of k values of base-layer classifiers.
k_meta=10; % k value of the meta-layer classifier.

% Call Fuzzy Stacked Generalization (FSG) method for classification:

[label_predictions,membership_predictions,meta_layer_performance,base_layer_performances] = fsg(features_tr,features_te,labels_tr,labels_te,feature_names,k_list_base,k_meta);














