## Surgical-detection

# Training code
Dataloader
There are different dataloaders for different videos to ensure that images from different videos are not used together(since it might affect LSTM block's learning)
Encoder
The model consists of a Resnet18 backbone which extracts features from the images. After this the feature maps are sent to Tool_WSL which extracts a fclass activation map with 6 channels for each tool corresponding to each frame. The feature maps from Resnet18 are also sent to an LSTM and CAGAM.Before sending to LSTM data shape is converted from (batch, channel, height, width) to (batch-9, 10, channel, height, width). The 10 is basically due to stacking of last 9 frames with present frame. The LSTM extracts temporal features using feature maps of last 9 frames and present frame, and then uses it to predict phase id, phase class activation maps and also extract triplet class activation maps in Phase_WSl. This is done because last frames help in predicting current phase which narrows down the possible triplets. The triplet cam extractor and phase cam extractor share a convolutional layer to learn their corelation. The phase cams are sent to CAGAM block to enhance verb prediction since possible verbs are highly dependent on current phase. The CAGAM block uses attention mechanism to use tool cams for verb and target prediction. 
Decoder
Tool, verb, target and triplet cams are sent to a decoder which consists of 2 layers of mixed attention. Mixed attention learns interactions of tool, verb and target with triplet in addition to decoding self cams. 
Classifier
At the end a fully connected layer is used to predict final triplet logits.

To create logits of tool, verb , target and triplet fom their cams, adaptive average pooling is used.

# Test code
Dataloader
At the beginning of each video 9 black frames are added to be able to match the LSTM requirements.
Test loop
At the beginning of each batch(except the first one), last 9 frames of previous batch are added to be able to generate prediction for each frame, otherwise no prediction for first 9 frames of each batch will be generated. Last 9 frames of each batch are also saved to be used in next batch.
Outputs are also processed to structure in the required json format.
To predict bounding boxes, tool cams are used. The map corresponding to the tool in the triplet is extracted and softmax is applied to obtain probabilities. After that it is binarised using a dynamic threshold = map.mean + 2*map.std. Bounding box coordinates are calculated using this binary mask.
