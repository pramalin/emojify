package demo.rnn;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Emojify {

	static int Series_Length = 5;

	static Map<String, Integer> word_to_index = new HashMap<String, Integer>();
	static Map<Integer, String> index_to_word = new HashMap<Integer, String>();
	static Map<String, double[]> word_to_vec_map = new HashMap<String, double[]>();

	static void readGloveVecs(String filename) throws IOException {
		List<String> lines = Files.readAllLines(Paths.get(filename));

		int index = -1;
		for (String line : lines) {
			index = index + 1;
			String[] values = line.split(" ");

			String word = values[0];
			double[] vecs = new double[values.length - 1];

			for (int i = 0; i < vecs.length; i++) {
				vecs[i] = Double.parseDouble(values[i + 1]);
			}
			word_to_vec_map.put(word, vecs);

			word_to_index.put(word, index);
			index_to_word.put(index, word);
		}
	}

	static void readCsv(String filename, List<String> X, List<Integer> Y) throws IOException {
   		List<String> lines = Files.readAllLines(Paths.get(filename));

   		for (String line : lines) {
   			String[] tokens = line.split(",");
			X.add(tokens[0].replace("\"","").trim());
			Y.add(Integer.valueOf(tokens[1]));
		}
    }
	  
	static INDArray sentencesToIndices(List<String> X, Map<String, Integer> wordsToIndex, int maxLen) {
		int m = X.size(); // number of training examples

		// # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1
		// line)
		INDArray XIndices = Nd4j.zeros(m, 1, maxLen);

		for (int i = 0; i < m; i++) { // loop over training examples
			// Convert the ith training sentence in lower case and split is into words. You
			// should get a list of words.
			String[] sentenceWords = X.get(i).toLowerCase().split(" ");

			// Initialize j to 0
			int j = 0;

			// # Loop over the words of sentence_words
			for (String w : sentenceWords) {
				// Set the (i,j)th entry of X_indices to the index of the correct word.
				if (j < maxLen) {
					if (wordsToIndex.containsKey(w)) {
						XIndices.putScalar(new int[] { i, 0, j }, wordsToIndex.get(w));
					}
					// Increment j to j + 1
					j = j + 1;
				}
			}

		}
		return XIndices;
	}
	
	static INDArray convertToOneHot(List<Integer>Y, int C, int seriesLen) {
		INDArray labels = Nd4j.create(new int[]{Y.size(), C, seriesLen}, 'f');
		
		for (int i = 0; i < Y.size(); i++) {
			labels.putScalar(new int[]{i, Y.get(i), seriesLen - 1}, 1.0);
		}
		return labels;
	}
	  
	  
	static Map<Integer, String> emoji_dictionary = new HashMap<Integer, String>();
	
	private static MultiLayerNetwork Emojify_V2(Map<String, double[]> word_to_vec_map, Map<Integer, String> index_to_word) {
		int vocab_len = index_to_word.keySet().size(); //

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(0)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.005)
				.list()
				.layer(0, new EmbeddingLayer.Builder().nIn(vocab_len).nOut(50).activation(Activation.IDENTITY).build())
				.layer(1, new GravesLSTM.Builder().nIn(50).nOut(128).activation(Activation.TANH).dropOut(0.5).build())
//				.layer(2, new GravesLSTM.Builder().nIn(128).nOut(5).activation(Activation.TANH).dropOut(0.5).build())
				.layer(2, new RnnOutputLayer.Builder().nIn(128).nOut(5).activation(Activation.SOFTMAX)
				.lossFunction(LossFunctions.LossFunction.MCXENT).build())
				.setInputType(InputType.recurrent(1))
				.pretrain(false)
				.backprop(true)
				.build();

		System.out.println(conf.toJson());
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer embeddingLayer =
			(org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer) net.getLayer(0);

		// Retrieving the weights
		INDArray weights = embeddingLayer.getParam(DefaultParamInitializer.WEIGHT_KEY);

		// putting pre-trained weights into rows
		INDArray rows = Nd4j.createUninitialized(new int[] { vocab_len, weights.size(1) }, 'c');

		for (int i = 0; i < vocab_len; i++) {
			String word = index_to_word.get(i);

			double[] embeddings = word_to_vec_map.get(word);
			if (embeddings != null) {
				INDArray newArray = Nd4j.create(embeddings);
				rows.putRow(i, newArray);
			} else { // if there is no pre-trained embedding value for that specific entry
				rows.putRow(i, weights.getRow(i));
			}
		}

		// finally put rows in place of weights
		embeddingLayer.setParam("W", rows);

		return net;
	}
	  
	public static void main(String[] args) throws Exception {


		// initialize emoji dictionary
		emoji_dictionary. put(0, "\u2764\uFE0F");    // :heart: prints a black instead of red heart depending on the font
		emoji_dictionary. put(1, ":baseball:");
		emoji_dictionary. put(2, ":smile:");
		emoji_dictionary. put(3, ":disappointed:");
		emoji_dictionary. put(4, ":fork_and_knife:");
		
		System.out.println("Read glove file ...");
		readGloveVecs(new ClassPathResource("data/glove.6B.50d.txt").getFile().getPath());

/*
		List<String> X1 = new ArrayList<String>();
		X1.add("funny lol");
		X1.add("lets play baseball");
		X1.add("food is ready for you");

		INDArray X1Indices = sentencesToIndices(X1, word_to_index, Series_Length);
		System.out.println("X1 =" + X1);
		System.out.println("X1_indices =" + X1Indices);
*/
		// model
		MultiLayerNetwork model = Emojify_V2(word_to_vec_map, index_to_word);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
      
        // read train data
		List<String> X_train = new ArrayList<String>();
		List<Integer> Y_train = new ArrayList<Integer>();
		readCsv(new ClassPathResource("data/train_emoji.csv").getFile().getPath(), X_train, Y_train);
		
		INDArray X_train_indices = sentencesToIndices(X_train, word_to_index, Series_Length);
		INDArray Y_train_oh = convertToOneHot(Y_train, 5, Series_Length);

		// many to one output mask - 00001
		INDArray labelsMask = Nd4j.zeros(Y_train.size(), 5);
		INDArray lastColumnMask = Nd4j.ones(Y_train.size(), 1);
		labelsMask.putColumn(4, lastColumnMask);
		 
		// read test data
		List<String> X_test = new ArrayList<String>();
		List<Integer> Y_test = new ArrayList<Integer>();
		readCsv(new ClassPathResource("data/tesss.csv").getFile().getPath(), X_test, Y_test);
		
		INDArray X_test_indices = sentencesToIndices(X_test, word_to_index, Series_Length);
		INDArray Y_test_oh = convertToOneHot(Y_test, 5, Series_Length);

		ListDataSetIterator<DataSet> trainData =
				new ListDataSetIterator<DataSet>((new DataSet(X_train_indices, Y_train_oh, null, labelsMask)).asList(), 50);
		ListDataSetIterator<DataSet> testData =
				new ListDataSetIterator<DataSet>(new DataSet(X_test_indices, Y_test_oh).asList());

		// 50 epochs
		for (int i = 0; i < 100; i++) {
	        String str = "Test set evaluation at epoch %d: Score: %.4f Accuracy = %.2f, F1 = %.2f";
	        
			model.fit(trainData);
            
			//Evaluate on the test set:
            Evaluation evaluation = model.evaluate(testData);
            System.out.println(String.format(str, i, model.score(), evaluation.accuracy(), evaluation.f1()));
		}


		// This code allows you to see the mislabeled examples
		INDArray pred3 = model.output(X_test_indices);

		double miss = 0;
		for (int i = 0; i < X_test.size(); i++) {
		    int num = Nd4j.argMax(pred3.getRow(i).getColumn(4), 0).getInt(0,0);
		    if(num != Y_test.get(i)) {
		      miss = miss + 1;
		      System.out.println("Expected emoji: " + emoji_dictionary.get(Y_test.get(i)) +
		    		  " prediction: " + X_test.get(i) + emoji_dictionary.get(num));
		    }
		  }
		  
		System.out.println("Missed " + miss + " out of " + X_test.size() +
				" acc: " + ((X_test.size() - miss) / X_test.size()) * 100 + " percent"); 
		  
		// Change the sentence below to see your prediction. Make sure all the words are
		// in the Glove embeddings.
		List<String> X_test2 = new ArrayList<String>();
		X_test2.add("not feeling happy");

		X_test_indices = sentencesToIndices(X_test2, word_to_index, 5);
		INDArray pred4 = model.output(X_test_indices);
		int num4 = Nd4j.argMax(pred4.getRow(0).getColumn(4), 0).getInt(0, 0);

		System.out.println(" Test: " + X_test2.get(0) + emoji_dictionary.get(num4));
	}

}
