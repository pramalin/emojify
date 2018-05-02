## Emoji ##

This is an attempt to create DL4J version of Emoji LSTM example from Coursera DL specialization course.

The model is simplified to trouble shoot shape mismatch errors.

### Goal ###
  Given variable sized sentences (max 5) like the following
	"funny lol"
	"lets play baseball"
	"food is ready for you"
   predict the emoji identifier (0 to 4).

      +-----------+
 -->  | Embedding |  // maps word index to - Glove vector 
      +-----------+
            |
            V   
      +-----------+
      |   LSTM    |  // (5)
      +-----------+
            |
            V   
      +-----------+
      | RNN Output|  // many to one softmax
      +-----------+

### Specs: ###
	Use an Embedding layer initialized with prebuilt Glove.
	Time series Tx = 5
	train samples m = 132
	max # of features n_x = 5
	label shape = one hot vector (1,5)


### Issue: ###
   Fails during training, with the shape mismatch error.
	variables:
	output.shapeInformation = [2,660,5,5,1,0,1,99]
	labels.shapeInformation = [2,132,5,5,1,0,1,99] 
        
	code:
	grad = output.subi(labels);

  	---- stack trace ----
	LossMCXENT.computeGradient(INDArray, INDArray, IActivation, INDArray) line: 157	
	RnnOutputLayer(BaseOutputLayer<LayerConfT>).getGradientsAndDelta(INDArray) line: 169	
	RnnOutputLayer(BaseOutputLayer<LayerConfT>).backpropGradient(INDArray) line: 148	
	RnnOutputLayer.backpropGradient(INDArray) line: 63	
	MultiLayerNetwork.calcBackpropGradients(INDArray, boolean) line: 1323	
	MultiLayerNetwork.backprop() line: 1273	
	MultiLayerNetwork.computeGradientAndScore() line: 2237	
	StochasticGradientDescent(BaseOptimizer).gradientAndScore() line: 174	
	StochasticGradientDescent.optimize() line: 60	
	Solver.optimize() line: 53	
	MultiLayerNetwork.fit(INDArray, INDArray, INDArray, INDArray) line: 1780	
	MultiLayerNetwork.fit(INDArray, INDArray) line: 1729	
	MultiLayerNetwork.fit(DataSet) line: 1832	
	Emojify.main(String[]) line: 200	

