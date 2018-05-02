package demo.rnn;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Emojify {

	
	// 1.3 - Implementing Emojifier-V1
	// As shown in Figure (2), the first step is to convert an input sentence into
	// the word vector representation,
	// which then get averaged together. Similar to the previous exercise, we will
	// use pretrained 50-dimensional GloVe embeddings.
	// Run the following cell to load the word_to_vec_map, which contains all the
	// vector representations.
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
			X.add(tokens[0]);
			Y.add(Integer.valueOf(tokens[1]));
		}
    }
	  
	static INDArray sentencesToIndices(List<String> X, Map<String, Integer> wordsToIndex, int maxLen) {
		/*
		 * Converts an array of sentences (strings) into an array of indices
		 * corresponding to words in the sentences. The output shape should be such that
		 * it can be given to `Embedding()` (described in Figure 4).
		 * 
		 * Arguments: X -- array of sentences (strings), of shape (m, 1) word_to_index
		 * -- a dictionary containing the each word mapped to its index max_len --
		 * maximum number of words in a sentence. You can assume every sentence in X is
		 * no longer than this.
		 * 
		 * Returns: X_indices -- array of indices corresponding to words in the
		 * sentences from X, of shape (m, max_len)
		 */

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
					XIndices.putScalar(new int[] { i, 0, j }, wordsToIndex.get(w));
					// Increment j to j + 1
					j = j + 1;
				}
			}

		}
		return XIndices;
	}
	
	static INDArray convertToOneHot(List<Integer>Y, int C) {
		return Nd4j.eye(C).getRows(Y.stream().mapToInt(Integer::intValue).toArray());
	}
	  
	private static MultiLayerNetwork Emojify_V2(Map<String, double[]> word_to_vec_map, Map<Integer, String> index_to_word) {
		/*
		 * Function creating the Emojify-v2 model's graph.
		 * 
		 * Arguments: input_shape -- shape of the input, usually (max_len,)
		 * word_to_vec_map -- dictionary mapping every word in a vocabulary into its
		 * 50-dimensional vector representation word_to_index -- dictionary mapping from
		 * words to their indices in the vocabulary (400,001 words)
		 * 
		 * Returns: model -- a model instance
		 */

		int vocab_len = index_to_word.keySet().size(); //

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
				.layer(0, new EmbeddingLayer.Builder().nIn(vocab_len).nOut(50).activation(Activation.IDENTITY).build())
				.layer(1, new GravesLSTM.Builder().nIn(50).nOut(5).activation(Activation.TANH).build())
				.layer(2, new RnnOutputLayer.Builder().nIn(5).nOut(5).activation(Activation.SOFTMAX)
				.lossFunction(LossFunctions.LossFunction.MCXENT).build())
				.inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
				.inputPreProcessor(1, new FeedForwardToRnnPreProcessor()).pretrain(false).backprop(true).build();

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

			double[] embeddings = word_to_vec_map.get(word); // getEmbeddings is my own function
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
		System.out.println("read glove file ...");
		readGloveVecs(new ClassPathResource("data/glove.6B.50d.txt").getFile().getPath());

		List<String> X1 = new ArrayList<String>();
		X1.add("funny lol");
		X1.add("lets play baseball");
		X1.add("food is ready for you");

		INDArray X1Indices = sentencesToIndices(X1, word_to_index, 5);
		System.out.println("X1 =" + X1);
		System.out.println("X1_indices =" + X1Indices);

		MultiLayerNetwork model = Emojify_V2(word_to_vec_map, index_to_word);

		List<String> X_train = new ArrayList<String>();
		List<Integer> Y_train = new ArrayList<Integer>();
		readCsv(new ClassPathResource("data/train_emoji.csv").getFile().getPath(), X_train, Y_train);
		
		INDArray X_train_indices = sentencesToIndices(X_train, word_to_index, 5);
		INDArray Y_train_oh = convertToOneHot(Y_train, 5);

		INDArray labelsMask = Nd4j.zeros(Y_train.size(), 5);
		INDArray lastColumnMask = Nd4j.ones(Y_train.size(), 1);

		labelsMask.putColumn(4, lastColumnMask);

		for (int i = 0; i < 50; i++) {
			model.fit(new DataSet(X_train_indices, Y_train_oh, null, labelsMask));
		}

	}

}
