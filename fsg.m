function [label_predictions,membership_predictions,meta_layer_performance,base_layer_performances] = fsg(features_tr,features_te,labels_tr,labels_te,feature_names,k_list_base,k_meta)
% ==========================================================================
% FSG: Fuzzy Stacked Generalization
% ==========================================================================
% Input:
%       feature_names: 1 by J cell of the set of base-layer feature sets;
%                      Each variable "feature_name" of "feature_names" represents the name of a feature set, e.g. color_histogram.  
%
%       labels_tr: An N by 1 vector of class labels of training samples;
%                  Each variable of the vector represents the label of a training sample.
%
%       features_tr: A struct variable ("features_tr.feature_name") which represents;
%                    the sets of D_j dimensional feature vectors (with name "feature_name")
%                    of N number of training samples.
%
%       labels_te: An N by 1 vector of class labels of test samples;
%                  Each variable of the vector represents the label of a test sample.
%
%       features_te: A struct variable ("features_te.feature_name") which represents;
%                    the sets of D_j dimensional feature vectors (with name "feature_name")
%                    of N number of test samples.
%
%       k_list_base: J by 1 vector of the k values of the base-layer fuzzy k-NN classifiers.
%
%       k_meta: The k value of the meta-layer fuzzy k-NN classifier
% ==========================================================================
%% Initialize the variables:


class_list=unique(labels_tr); % Class list

class_n = length(class_list); % Number of classes.


tr_n = size(labels_tr,1);  % Number of training samples.
te_n = size(labels_te,1);  % Number of test samples.
indices = (1:tr_n)';                % Indices of training samples.

num_features = length(feature_names);   % Number of features.

%Initialize the membership vectors: 

memberships_tr = zeros(tr_n,num_features*class_n); 
memberships_te = zeros(te_n,num_features*class_n);

% Construct temporary label vectors of traning and test samples for
% classification using fknn.m:

labels_tr_temp=labels_tr;
labels_te_temp=labels_te;

for c=1:class_n
   
    labels_tr_temp(find(labels_tr==class_list(c)))=c;
    labels_te_temp(find(labels_te==class_list(c)))=c;    
    
end



%% Base-layer fuzzy k-NN classification:

%  Calculating class membership values of samples in the training dataset;

for f=1:num_features
	desc_name = cell2mat(feature_names(f));
	tr = getfield(features_tr,desc_name);

	for i=1:tr_n
		temp = (indices~=i);
		[Y,X,H]= fknn(tr(temp,:),labels_tr_temp(temp,:),tr(i,:), labels_tr_temp(i,:),k_list_base(f),0);
		memberships_tr(i,((f-1)*class_n+1):(f*class_n)) = X;
    end
    clear tr;
end	

clear X
clear Y

% Calculating class membership values of samples in the test dataset;

for f=1:num_features
    clear H;
    
	desc_name = cell2mat(feature_names(f));
	tr = getfield(features_tr,desc_name);
	te = getfield(features_te,desc_name);
	[Y,X,H] = fknn(tr,labels_tr_temp,te,labels_te_temp,k_list_base(f),0);
    clear tr;
    clear te;
	memberships_te(:,((f-1)*class_n+1):(f*class_n)) = X;
    base_layer_performances(f)=H/size(labels_te,1)*100;

end
clear X
clear Y

%% Meta-layer fuzzy k-NN classification:

[label_predictions,membership_predictions,hits] = fknn(memberships_tr,labels_tr_temp,memberships_te,labels_te_temp,k_meta,0);

% Calculate the meta-layer performance;
meta_layer_performance=hits/size(labels_te,1)*100;

% Construct a vector of label predictions using a meta-layer classifier;
for c=1:class_n
   
    label_predictions(find(label_predictions==c))=class_list(c);
    
end

