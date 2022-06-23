import java.util.*;
import java.io.*;
import java.lang.Math;
/******************************************************************
 * An instance of the <code>HMM</code> class represents a
 * hidden Markov model, including the model's states, transitions,
 * and the optimal annotation (i.e., parse) of an observation
 * sequence.
 ******************************************************************/
public class HMM {

    /**************************************************************
     ********************** INSTANCE VARIABLES ********************
     **************************************************************/

    /**
     * A collection of the states in the model.
     */
    private Vector<State> states;

    /**
     * The number of states in the model.
     */
    private int numStates;

    /**
     * A matrix of size (numStates x numStates) corresponding to the
     * probabilities of transitioning be any two states.
     */
    private double[][] transitionsMatrix;

    /**
     * The starting state of the optimal annotation (i.e., parse).
     * It is assumed that the first state (at index 0) will
     * always be the starting state.
     */
    private int startState;

    /**
     * A sequence of observations. This HMM will compute the 
     * optimal annoation of the observationSequence.
     */
    private String observationSequence;

    /**
     * The dynamic programming table used by the Viterbi algorithm.
     * The table has one row for each state in the HMM.
     * The table has one column for each character in the observationSequence.
     * The table entry at row i and column j corresponds to the probability
     * (to be precise, the natural logarithm of the probability)
     * of the best annotation of the first (j+1) characters in the 
     * observationSequence assuming the best annotation of these
     * characters ends in state i. 
     */
    private double[][] table;

    /**
     * The backtracking table used by the Viterbi algorithm.
     * As the Viterbi algorithm fills in the dynamic programming table,
     * it also fills in this backtracking table.
     * While the dynamic programming table enables computation of the
     * "probability" of the optimal annotation, the backtracking table
     * enables computation of the optimal annotation itself.
     * The backtracking table entry at row i and column j corresponds to the
     * state prior to state i in the optimal annotation, assuming state i
     * is the end state after observing the first (j+1) characters in
     * the observationSequence.
     */
    private int[][] backtrackTable;

    /**
     * The score of the optimal annotation of the observationSequence.
     * The score is the natural logarithm of the probability of the annotation.
     */
    private double scoreOfOptimalAnnotation;

    /**
     * The optimal annotation (i.e., parse) of the observationSequence.
     */
    private String optimalAnnotation;

    /**************************************************************
     ********************** CONSTRUCTOR ***************************
     **************************************************************/

    /**
     * Creates a hidden Markov model based on the specified states, transitions,
     * and observation sequence.
     * <P>
     * The states of the HMM are read in from the file specified by the first parameter.
     * The transitions between states of the HMM are read in from the file specified by the second parameter.
     * The observation sequence emitting by the HMM is read in from the file specified by the third parameter.
     * It is assumed that all paths (state sequences) through the HMM start in the starting state, 
     * which is the first state (i.e., the state at index 0).
     * It is assumed that the first state emits characters only of length 1.
     * It is assumed that for any state in the HMM, all characters emitted by that state
     * are the same length.
     * The constructor initializes the HMM, but it does not compute via the Viterbi algorithm the optimal
     * annotation for the specified observation sequence.
     *
     * @param   states_fileName   the name of a tab-delimited file specifying the states of the HMM
     * @param   transitions_fileName   the name of a tab-delimited file specifying the transitions
     * between states of the HMM
     * @param   sequence_fileName   the name of a FASTA formatted file containing an observation sequence,
     * i.e., a sequence emitted by the HMM for which we want to determine the optimal annotation
     */
    public HMM(String states_fileName, String transitions_fileName, String sequence_fileName) {
        this.states = readInInformationAboutStates(states_fileName);
        this.numStates = states.size();
        this.transitionsMatrix = new double[numStates][numStates];
        readInInformationAboutTransitions(transitions_fileName);  // populate transitionsMatrix
        this.startState = 0;
        this.observationSequence = SequenceOps.sequenceFromFastaFile(sequence_fileName);
        this.table = new double[numStates][observationSequence.length()];
        this.backtrackTable = new int[numStates][observationSequence.length()];
        this.optimalAnnotation = "";
    }

    /**************************************************************
     ********************** PUBLIC INSTANCE METHODS ***************
     **************************************************************/
    /**
     * Calculate probability of current emission for use in filling the DP table
     */
    public double get_prob_emission(int i, int j, int char_len){
        // get length of current state's alphabet using getEmissionLength()
        double emission_prob = Double.NEGATIVE_INFINITY;
        
        double temp_emission_prob = states.get(i).getEmissionProbability(observationSequence.substring(j-char_len+1, j+1));
        temp_emission_prob = Math.log(temp_emission_prob);
        if (temp_emission_prob>emission_prob){
             emission_prob=temp_emission_prob;
        }
        return emission_prob;
    }

    /**
     * Find and return max between three doubles
     */
    public double max_3(double a, double b, double c){
        double max = Double.NEGATIVE_INFINITY; //since were working with logs of decimals (aka negatives) we want to start at neg inf not 0
        if (a>max){
            max = a;
        }

        if (b>max){
            max=b;
        }
        if (c>max){
            max = c;
        }
        return max;
    }
    /**
     * Calculate ln of probability of transition from state y to state z
     */
    public double y_z_transition(int y, int z){
        return Math.log(this.transitionsMatrix[y][z]);
    }
    /**
     * Fills in the entry at row <code>row</code> and column <code>col</code> in <em>both</em>
     * the dynamic programming table and the backtracking table.
     * <P>
     * Each entry in the dynamic programming table corresponds to the natural logarithm 
     * of a probability. For the dynamic programming table,
     * the table entry at row <code>row</code> and column <code>col</code> corresponds 
     * to the probability (to be precise, the natural logarithm of the probability)
     * of the best annotation of the first (<code>col</code>+1) characters in the 
     * observationSequence assuming the best annotation of these
     * characters ends in state <code>row</code>. 
     * While the dynamic programming table enables computation of the
     * "probability" of the optimal annotation, the backtracking table
     * enables computation of the optimal annotation itself.
     * The backtracking table entry at row <code>row</code> and column <code>col</code> 
     * corresponds to the state prior to state <code>row</code> in the optimal annotation, 
     * assuming state <code>row</code> is the end state after observing the first 
     * (<code>col</code>+1) characters in the observationSequence.
     * Any dynamic programming table entry corresponding to a state that is unreachable 
     * should be populated with a value of negative infinity corresponding to the 
     * natural logarithm of a probability of zero.
     *
     * @param   row   the row of the table entry to be filled in
     * @param   col   the column of the table entry to be filled in
     */
    public void fillInTableEntry(int row, int col) {
        
        int i = row;
        int j = col;
        // first get the emissions length from states.get(row)
        State emissions =  this.states.get(row);
        int emissionsL = emissions.getEmissionLength();
        double entry = Double.NEGATIVE_INFINITY;
        
        //different cases of self transition vals at first column
        if(emissionsL>j){
        
            if (j==0){ //for first columnSystem.out.println("here at 202");
            double probitoi = this.y_z_transition(i, i)+ this.get_prob_emission(i, j,emissionsL);
            entry = probitoi;
            this.table[i][j] = entry;
            // }
            // if (i==1){//CASE 2:self transistion from state 2 to state 2  
                // double prob2to2 = this.y_z_transition(1,1) + this.get_prob_emission(i, j,emissionsL);
                // entry = prob2to2;
            // }
            // if (i==2){//CASE 3: self transistion from state 3 to state 3
                // double prob3to3 = this.y_z_transition(2,2) + this.get_prob_emission(i, j,emissionsL);
                // entry = prob3to3;
            // }
        }
        else{
            entry = Double.NEGATIVE_INFINITY;
            this.table[i][j] = entry;
            this.backtrackTable[i][j] = -1;
        }
    }
        
        if(emissionsL<=j){// (so for every other column) 
            double max_k_to_i = Double.NEGATIVE_INFINITY;
            int best_k = -1;
            for (int k=0; k<this.numStates; k++){
            State k_emissions =  this.states.get(k);
            int state_char_len = emissions.getEmissionLength();
            
            double k_to_i = this.y_z_transition(k,i) + table[k][j-state_char_len];
            if (k_to_i>max_k_to_i){
                max_k_to_i = k_to_i;
                best_k = k;
            }
            // double two_to_i = this.y_z_transition(1,i) + table[1][j-1];
            // double three_to_i = this.y_z_transition(2,i) + table[2][j-1];
            // double max_transition = this.max_3(one_to_i, two_to_i, three_to_i);
            
        }
        entry = max_k_to_i + this.get_prob_emission(i, j,emissionsL);
        this.backtrackTable[i][j] = best_k;
        this.table[i][j] = entry;
        }
        

    }

    /**
     * Implements the Viterbi algorithm by filling in a dynamic programming table 
     * and a backtracking table.
     * <P>
     * The method populates every entry in a dynamic programming table
     * as well as every entry in a backtracking table.
     * The tables have one row for each state in the HMM.
     * The tables have one column for each character in the observationSequence.
     * For the dynamic programming table,
     * the table entry at row i and column j corresponds to the probability
     * (to be precise, the natural logarithm of the probability)
     * of the best annotation of the first (j+1) characters in the 
     * observationSequence assuming the best annotation of these
     * characters ends in state i. 
     * While the dynamic programming table enables computation of the
     * "probability" of the optimal annotation, the backtracking table
     * enables computation of the optimal annotation itself.
     * The backtracking table entry at row i and column j corresponds to the
     * state prior to state i in the optimal annotation, assuming state i
     * is the end state after observing the first (j+1) characters in
     * the observationSequence.
     * The following assumptions are made:
     * <UL>
     *   <LI>It is assumed that all paths (state sequences) through the HMM start in 
     *   the starting state, which is the first state (i.e., the state at index 0).</LI>
     *   <LI>It is assumed that the first state emits characters only of length 1.</LI>
     *   <LI>It is assumed that for any state in the HMM, all characters emitted by that state
     *   are the same length.</LI>
     * </UL>
     * <P>
     * Since it is assumed that all annotations start in the first state, every other state
     * besides the first state in the first column of the dynamic programming table should 
     * be populated with a value of negative infinity corresponding to the natural logarithm 
     * of a probability of zero.
     */
    public void viterbi() {
        // first fill the dynamic prog. table with negative infinities for the first column
        for (int s = 0; s<numStates; s++){// in the whole column 0 but the different states. 
            this.table[s][0] = Double.NEGATIVE_INFINITY;
            // and set all values in this position of the backtrackingtable to -1
            this.backtrackTable[s][0] = -1;
        }
        //next, fill starting state
        this.table[startState][0] = Math.log(this.states.get(0).getEmissionProbability(this.observationSequence.substring(0,1)));
        
        // start after the first state set i = 1;
        for (int j=1; j<observationSequence.length(); j++){ /// loop must fill each column sequentially
            for(int i=0; i<this.numStates; i++){ // then fill each row in the column
                this.fillInTableEntry(i, j); // fillInTableEntry(row, col)
            }
        }

    }
    /**
     * Reverse String
     */
    public String reverseString(String s){
        StringBuilder s_build = new StringBuilder();
        for(int i=s.length()-1; i>=0; i--){
            s_build.append(s.substring(i,i+1));
        }
        String s_to_return = s_build.toString();
        return s_to_return;
    }
    /**
     * Determines the optimal annotation of the observation sequence.
     * <P>
     * Once the dynamic programming table and backtracking table have
     * been filled in, this method uses the tables to construct a
     * <code>String</code> representation of the optimal annotation.
     * The maximum value in the final column of the dynamic programming
     * table indicates the final state in the optimal annotation. The
     * backtracking table then can be used to backtrack from this final
     * state to determine the optimal state sequence (i.e., annotation).
     * The optimal annotation should be the same length as the 
     * observation sequence and each state in the optimal annotation
     * should be represented by the first character in the state's name.
     */
    public void determineOptimalAnnotation() {
        int best_state = -1;
        double best_score = Double.NEGATIVE_INFINITY;
        int end = this.observationSequence.length()-1;
        for (int i = 0; i < this.numStates; ++i) {
            if (this.table[i][end] >best_score) {
                best_score = this.table[i][end];
                best_state = i;
            }
        }
        //now that best_state holds the state we start backtracking with, we can start backtracking
        //set scoreOfOptimalAnnotation as highest score overall
        this.scoreOfOptimalAnnotation = best_score;
        StringBuilder best_seq = new StringBuilder();
        int curr_ind = best_state;
        int val_to_go_back;
        //backtracking
        for (int j=end; j>0; j-=val_to_go_back) {
            val_to_go_back = states.get(curr_ind).getEmissionLength();
            for (int adding= 0; adding< val_to_go_back; ++adding) { //++adding vs adding++ so that we dont add and extra char at begining
                //add current state's first initial to the building seq 
                best_seq.append(states.get(curr_ind).getName().charAt(0));
            }
            //ind to pick up back track from
            curr_ind = backtrackTable[curr_ind][j];
        }
        best_seq.append(states.get(curr_ind).getName().charAt(0));
        
        optimalAnnotation = best_seq.reverse().toString();
    }

    /**
     * Returns the score (the natural logarithm of the probability) of
     * the optimal annotation of the observation sequence.
     *
     * @return   a <code>double</code> representing the score of the optimal annotation
     */
    public double getScoreOfOptimalAnnotation() {
        return scoreOfOptimalAnnotation;
    }

    /**
     * Returns a String representation of the optimal annotation.
     * The annotation <em>includes</em> the observation sequence.
     * The returned String is in FASTA format.
     *
     * @return   a <code>String</code> representing the optimal annotation.
     */
    public String optimalAnnotationToString() {
        int fastaLineLength = 60;
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < optimalAnnotation.length()) {
            sb.append("\n");
            if (i+fastaLineLength > optimalAnnotation.length()) {  // Last line of FASTA format
                sb.append("\t\t" + observationSequence.substring(i) + "\t" + (optimalAnnotation.length()) + "\n");
                sb.append("\t\t" + optimalAnnotation.substring(i) + "\n");
            } else {
                sb.append("\t\t" + observationSequence.substring(i, i+fastaLineLength) + "\t" + (i+fastaLineLength) + "\n");
                sb.append("\t\t" + optimalAnnotation.substring(i, i+fastaLineLength) + "\n");
            }
            i += fastaLineLength;
        }
        return sb.toString();
    }

    /**
     * Returns a String representation of this HMM.
     * The String representation of the HMM consists of a String
     * representation of each state in the HMM.
     *
     * @return   a <code>String</code> representing this HMM.
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("\nThe HMM consists of the following " + numStates + " states:\n");
        for (int i=0; i<numStates; i++) {
            sb.append("\t" + states.get(i).toString() + "\n");
        }
        return sb.toString();
    }

    /**
     * Returns the dynamic programming table used by the Viterbi algorithm.
     * 
     * @return   a 2D array of <code>doubles</code> corresponding to the dynamic programming table.
     */
    public double[][] getTable() {
        return table;
    }

    /**
     * Returns the backtracking table used by the Viterbi algorithm.
     * 
     * @return   a 2D array of <code>ints</code> corresponding to the backtracking table.
     */
    public int[][] getBacktrackTable() {
        return backtrackTable;
    }

    /**
     * Returns a String representation of a 2D array.
     * <P>
     * A helpful method for debugging. A String representation of 
     * the dynamic programming table can be obtained using this
     * method. However, the method is only practical for 
     * <em>small</em> 2D arrays.
     *
     * @param   a   a 2D array of <code>doubles</code>
     * @return   a <code>String</code> representation of the 2D array
     */
    public String arrayToString(double[][] a) {
        if (a.length == 0) return "";
        if (a[0].length == 0) return "";

        java.text.DecimalFormat df = new java.text.DecimalFormat("0.00");
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++) {
            for (int j=0; j<a[0].length; j++) {
                if (a[i][j] <= -Double.MAX_VALUE+1)
                    sb.append("-Infty" + " ");
                else
                    sb.append(df.format(a[i][j]) + "  ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Returns a String representation of a 2D array.
     * <P>
     * A helpful method for debugging. A String representation of 
     * the backtracking table can be obtained using this
     * method. However, the method is only practical for 
     * <em>small</em> 2D arrays.
     *
     * @param   a   a 2D array of <code>ints</code>
     * @return   a <code>String</code> representation of the 2D array
     */
    public String arrayToString(int[][] a) {
        if (a.length == 0) return "";
        if (a[0].length == 0) return "";

        StringBuilder sb = new StringBuilder();
        for (int i=0; i<a.length; i++) {
            for (int j=0; j<a[0].length; j++) {
                sb.append(a[i][j] + "  ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Outputs the optimal annotation and its score to a file.
     * 
     * @param   filename   the name of the output file
     */
    public void annotationToFile(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.println("\nOptimal annotation:\n" + optimalAnnotationToString() + "\n");
            writer.println("Score of optimal annotation:\t" + getScoreOfOptimalAnnotation() + "\n");        
            writer.close();        
        } catch (FileNotFoundException ex) {
            System.err.println(ex);
        }   
    }

    /**************************************************************
     ********************** PRIVATE INSTANCE METHODS ***************
     **************************************************************/

    /**
     * Reads in information from a file about each state in the HMM.
     *
     * @param   states_fileName   the name of a tab-delimited file specifying the states of the HMM
     * @return   a collection of <code>States</code> as specified by the file
     */
    private Vector<State> readInInformationAboutStates(String states_fileName) {
        Vector<State> v = new Vector<State>();
        try {
            Scanner reader = new Scanner(new File(states_fileName));
            while (reader.hasNextLine()) {  // continue until we reach end of file
                String line = reader.nextLine();
                State s = new State(line);
                v.add(s);
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.err.println("Error - the file " + states_fileName + " could not be found and opened.");
            System.exit(0);
        }
        return v;
    }

    /**
     * Reads in information from a file about the transitions between states in the HMM.
     * Populates a matrix corresponding to the transitions between the states of the HMM.
     *
     * @param   transitions_fileName   the name of a tab-delimited file specifying the transitions
     * between states of the HMM
     */
    private void readInInformationAboutTransitions(String transitions_fileName) {

        try {
            Scanner reader = new Scanner(new File(transitions_fileName));
            int row = 0;
            String line = reader.nextLine();  // Ignore header line
            while (reader.hasNextLine()) {  // continue until we reach end of file
                line = reader.nextLine();
                String[] splitLine = line.split("\\s+");  // split line around any white space

                // Check to ensure that line contains the correct number of tokens
                if (splitLine.length != numStates+1) {
                    System.err.println("Error - the number of states differs from the size of the transitions matrix.");
                    System.exit(0);
                }

                // Populate one row of transitions matrix. Ignore first column header.
                for (int i=0; i<numStates; i++) {
                    transitionsMatrix[row][i] = Double.parseDouble(splitLine[i+1]);
                }
                row++;
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.err.println("Error - the file " + transitions_fileName + " could not be found and opened.");
            System.exit(0);
        }
    }

    /*********************************************************
     ****************** MAIN METHOD **************************
     *********************************************************/

    /**
     * The <code>main</code> method generates a <code>HMM</code> based on the state and
     * transition information in two files specified by command line arguments. An observation
     * sequence is determined from a FASTA formatted file also specified as a command line
     * argument. After the model is built based on the information in the three files,
     * the Viterbi algorithm along with backtracking is used to determine the optimal
     * annotation of the observation sequence. The optimal annotation, as well as its score,
     * is output to the screen and to a file named annotation.txt. The application can be
     * executed as follows:
     * <BR>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<code>java HMM states.txt
     * transitions.txt observation.txt</code>
     * 
     * @param   args   an array of <code>Strings</code> representing any command line arguments
     */
    public static void main(String[] args) {
        

        HMM hmm = new HMM(args[0], args[1], args[2]);
        //System.out.println(hmm.toString());
        hmm.viterbi();
        //System.out.println(hmm.arrayToString(hmm.getTable()));
        //System.out.println(hmm.arrayToString(hmm.getBacktrackTable()));
        hmm.determineOptimalAnnotation();
        //System.out.println("\nOptimal annotation:\n" + hmm.optimalAnnotationToString() + "\n");
        System.out.println("Score of optimal annotation:\t" + hmm.getScoreOfOptimalAnnotation() + "\n");
        hmm.annotationToFile("annotation.txt");

    }

}
