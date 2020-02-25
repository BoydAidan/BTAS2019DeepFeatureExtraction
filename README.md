# BTAS2019DeepFeatureExtraction
This repo contains the supplementary material for the 2019 BTAS submission Deep Learning-Based Feature Extraction in Iris Recognition: Use Existing Models, Fine-tune or Train From Scratch?

Trained models can be found at the following links (GitHub has a 100Mb file size limit so I had to upload to Google Drive):

Model trained from scratch using 363,512 iris images from 2000 classes: email aboyd3@nd.edu

Model fine-tuned from ImageNet weights on same iris data as Scratch network: email aboyd3@nd.edu

Model fine-tuned from VGGFace2 weights on same iris data as Scratch network: email aboyd3@nd.edu

This project was written in Python. The code I used to split the CASIA-Iris-Thousand database (found here: http://biometrics.idealtest.org/dbDetailForUser.do?id=4) into the single 70% train/30% test split:

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4242, stratify=y)
  -> the random seed is important of you are trying to replicate my results. 
  
Note: Depending on how successful your segmentation is, you may see slightly varying results. If you want to know my configuration for OSIRIS, send me an email at aboyd3@nd.edu and I can send you my config file so you can replicate the experiments exactly.

If you want to know how features were extracted by each layer, email me and I will send you my extraction code :) 
But this is the important lines in the extraction (this is simplified to one image, you should create a list of all images and loop through them):

  img = image.load_img(image_path, target_size=(64, 512), color_mode="rgb")<br/>
  img = image.img_to_array(img)<br/>
  img = np.expand_dims(img, axis=0)<br/>
  intermediate_layer_model = Model(inputs=model.input,
                                             outputs=model.get_layer(layer).output)<br/>
  features = intermediate_layer_model.predict(img)<br/>
  np.save(PATH, features) # make sure your filename ends with .npy not .png or .jpg!<br/>

I then saved these features to a numpy array which will be loaded by the classification program.

I did Min/Max scaling on the data before PCA, this can be done by:

  from sklearn.preprocessing import MinMaxScaler

  scaler = MinMaxScaler(feature_range=(0, 1))<br/>
  scaler.fit(X_train)<br/>
  X_train = scaler.transform(X_train)<br/>
  X_test = scaler.transform(X_test)<br/>

For PCA I used the library:
from sklearn.decomposition import PCA

and the code to run the PCA is:

  pca = PCA(n_components=2000, svd_solver='randomized')<br/>
  pca.fit(X_train)<br/>
  X_train = pca.transform(X_train)<br/>
  X_test = pca.transform(X_test)<br/>
  coverage = 0<br/>
  num_feats = 0<br/>
  for val in pca.explained_variance_ratio_:<br/>
      coverage = coverage + val<br/>
      num_feats = num_feats + 1<br/>
      if coverage >= 0.9: # This is taking the number of features corresponding to 90% of the variance<br/>
          break<br/>
  X_train = X_train[:, :num_feats]<br/>
  X_test = X_test[:, :num_feats]<br/>
  -> Sometimes 90% of the coverage requires more than the 2000 features, in which case just the 2000 will be used
  
  For classification I used the one-versus-rest SVM from scikit learn, this can be implemented as follows:
  
  clf = multiclass.OneVsRestClassifier(svm.SVC(kernel='linear', gamma='auto'), n_jobs=-1)<br/>
    clf.fit(X_train, y_train)<br/>
    estimates = clf.predict(X_test)<br/>
    

I am more than happy to answer any questions on this work so don't hesitate to email me!
