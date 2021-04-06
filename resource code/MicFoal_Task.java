//First of all, the trained model is used for prediction, and then active learning is carried out, 
//The whole process is evaluated prequentially

package moa.tasks;


import java.io.File;
import java.io.FileOutputStream;

import java.io.PrintStream;

import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.TimingUtils;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.evaluation.EWMAClassificationPerformanceEvaluator;
import moa.evaluation.FadingFactorClassificationPerformanceEvaluator;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.streams.ExampleStream;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.Utils;

import moa.classifiers.active.MicFoal;


public class MicFoal_Task extends ClassificationMainTask implements CapabilitiesHandler {

    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by testing then training with each example in sequence.";
    }

    private static final long serialVersionUID = 1L;

    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "First predict then active learn.", MultiClassClassifier.class, "moa.classifiers.active.MicFoal");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "WindowClassificationPerformanceEvaluator");

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'j',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",  'f',
            "How many instances between samples of the learning performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(  "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpoverallEvaluateResOption = new FileOption("dumpOverallEvaluateFile", 'd',
            "CSV File to output overall evaluation result for whole data stream.", null, "csv", true);
    
	public FileOption dumpprobVectorFileOption = new FileOption("dumpProbVectorFile", 'z',
            "CSV File to output prediction probability vector for each instance.", null, "csv", true);

    public FileOption dumppredLabelFileOption = new FileOption("dumpPredLabelFile", 'x',
            "CSV File to output prediction label for each instance.", null, "csv", true);

    public FileOption dumpSelfEvaluateFileOption = new FileOption("dumpSelfEvaluateFile", 'v',
            "CSV File to output self evaluation result on the real label sliding window.", null, "csv", true);

    //New for prequential method DEPRECATED
    public IntOption widthOption = new IntOption("width",  'w', "Size of Window", 1000);

    public FloatOption alphaOption = new FloatOption("alpha", 'a', "Fading factor or exponential smoothing factor", .01);
    //End New for prequential methods

    public IntOption initInstanceNumOption = new IntOption("initInstanceNumber", 'i', "The number of instances to init model at the begin.", 
    		500, 1, Integer.MAX_VALUE);

  
    public FloatOption activelearningbudget = new FloatOption("activebudget", 	'b',"The percentenge of random selection size.",
    		0.1, 0.0, 1.0);
    
    public FloatOption randRatioOption = new FloatOption("randratio",  'r', "random ratio to select instances for imbalance sampling.",
            0.05, 0.0, 1.0);
    
    public IntOption justInitalParamsOption = new IntOption("justInitalParams", 'y',
    		"if learner has be trained, just need to inital learner's params.", 
    		0, 0, 1); //0: Learner is not trained; 1: learner has be trained, just need to inital learner's params


    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }

    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
    	MicFoal learner = (MicFoal) getPreparedClassOption(this.learnerOption);

        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);
        LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");

        //New for prequential methods
        if (evaluator instanceof WindowClassificationPerformanceEvaluator) {
            //((WindowClassificationPerformanceEvaluator) evaluator).setWindowWidth(widthOption.getValue());
            if (widthOption.getValue() != 1000) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (WindowClassificationPerformanceEvaluator -w " + widthOption.getValue() + ")");
                 return learningCurve;
            }
        }
        if (evaluator instanceof EWMAClassificationPerformanceEvaluator) {
            //((EWMAClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (EWMAClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        if (evaluator instanceof FadingFactorClassificationPerformanceEvaluator) {
            //((FadingFactorClassificationPerformanceEvaluator) evaluator).setalpha(alphaOption.getValue());
            if (alphaOption.getValue() != .01) {
                System.out.println("DEPRECATED! Use EvaluatePrequential -e (FadingFactorClassificationPerformanceEvaluator -a " + alphaOption.getValue() + ")");
                return learningCurve;
            }
        }
        //End New for prequential methods

        learner.setModelContext(stream.getHeader());
        if(justInitalParamsOption.getValue() ==1)
        {//just need to set the learner's params, do not need to initial learner itself !!
	        int numclasses = stream.getHeader().numClasses();
	        double budget = this.activelearningbudget.getValue();
	        double randratio = this.randRatioOption.getValue();
	    	int numInitInstance=this.initInstanceNumOption.getValue();
	        learner.initParameters(numclasses,budget, randratio, numInitInstance);
        }
        
        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);
        
        File dumpOverallEvaluateFile = this.dumpoverallEvaluateResOption.getFile();
        PrintStream overallEvaluationResultStream = null;
        if (dumpOverallEvaluateFile != null) {
            try {
                if (dumpOverallEvaluateFile.exists()) {
                    overallEvaluationResultStream = new PrintStream(
                            new FileOutputStream(dumpOverallEvaluateFile, false), true);
                } else {
                    overallEvaluationResultStream = new PrintStream(
                            new FileOutputStream(dumpOverallEvaluateFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open the overall evaluation result file: " + dumpOverallEvaluateFile, ex);
            }
        }
 
    	File predProbdumpFile = this.dumpprobVectorFileOption.getFile();
        PrintStream predProbVectorStream = null;
        if (predProbdumpFile != null) {
            try {
                if (predProbdumpFile.exists()) {
                	predProbVectorStream = new PrintStream(
                            new FileOutputStream(predProbdumpFile, false), true);
                } else {
                	predProbVectorStream = new PrintStream(
                            new FileOutputStream(predProbdumpFile), true);
                }
            	String strheader = "RealLabel";
                for (int ii=0;ii<stream.getHeader().numClasses();ii++) {
                	strheader = strheader + ",Prob_"+ii;
                } 
                predProbVectorStream.println(strheader); //写CSV的文件头
                
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction probability vector file: " + predProbdumpFile, ex);
            }
            
        }

    	File predLabeldumpFile = this.dumppredLabelFileOption.getFile();
        PrintStream predLabelStream = null;
        if (predLabeldumpFile != null) {
            try {
                if (predLabeldumpFile.exists()) {
                	predLabelStream = new PrintStream(
                            new FileOutputStream(predLabeldumpFile, false), true);
                } else {
                	predLabelStream = new PrintStream(
                            new FileOutputStream(predLabeldumpFile), true);                	
                }
                predLabelStream.println("InstancesNo , Realclassindex , Predictclassindex , Selectedby, CurBudget, CurInstanceWeight" );//写CSV的文件头
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction lable file: " + predLabeldumpFile, ex);
            }
        }

        
        
        //File for output evaluation results by random sampling
    	File selfEvaluateDumpFile = this.dumpSelfEvaluateFileOption.getFile();
        PrintStream selfEvaluateStream = null;
        if (selfEvaluateDumpFile != null) {
            try {
                if (selfEvaluateDumpFile.exists()) {
                	selfEvaluateStream = new PrintStream(
                            new FileOutputStream(selfEvaluateDumpFile, false), true);
                } else {
                	selfEvaluateStream = new PrintStream(
                            new FileOutputStream(selfEvaluateDumpFile), true);                	
                }
                selfEvaluateStream.println("InstancesNo , avgPrecision , avgRecall , avgF1, numRowsofCMatrix, newBudget" );//写CSV的文件头
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open selfEvaluateDumpFile file: " + selfEvaluateDumpFile, ex);
            }
        }

        
        boolean firstDump = true;

        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;
        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            Example trainInst = stream.nextInstance();
            Example testInst = (Example) trainInst; //.copy();
            
            
            //testInst.setClassMissing();
            double[] prediction = learner.getVotesForInstance(testInst);
            // Output prediction
            int trueClass = (int) ((Instance) trainInst.getData()).classValue();

            evaluator.addResult(testInst, prediction);
            
         	Instance inst = (Instance) trainInst.getData();
        	String strTrueClass = String.valueOf(trueClass);
        	int predClassindex = -2;
        	if(prediction.length>0)
        		predClassindex=Utils.maxIndex(prediction);


            if (predProbVectorStream != null) {

	            
            	String strprobvalues = strTrueClass;
            	
            	DoubleVector vote = new DoubleVector(prediction);
            	if (vote.sumOfValues() > 0.0)
            	{
            		vote.normalize();
            		double[] votedouble = vote.getArrayRef();
    	            for (int ii=0; ii<votedouble.length; ii++) 
    	            {
    	            	strprobvalues =strprobvalues+ ","+ String.valueOf(votedouble[ii]);
     	            } 
    	            for (int ii=votedouble.length; ii<learner.getnumClasses(); ii++) 
    	            {
    	            	strprobvalues =strprobvalues+ ",0";
    	            } 

            	}
            	else
            	{
    	            for (int ii=0;ii<learner.getnumClasses();ii++)
    	            	strprobvalues =strprobvalues+ ","+ String.valueOf(1.0/(double)learner.getnumClasses());
           	   }
                 
            	predProbVectorStream.println(strprobvalues);
            }
            
	        if (predLabelStream != null) {
            	String realClassindex =  ((Instance) testInst.getData()).classIsMissing() == true ? " ? " : strTrueClass;
	        	predLabelStream.println(instancesProcessed +", "+ realClassindex + ", " + predClassindex + ", " + 
            	learner.instSelectedby()+", " + learner.getbudget()+", " + learner.gettrainingWeight());
	        }
	        
            learner.trainOnInstance(trainInst);
            instancesProcessed++;
            if ((instancesProcessed % learner.getsizeofWindow() == 0
                    || stream.hasMoreInstances() == false) && (selfEvaluateStream != null) && (instancesProcessed > 1*learner.getsizeofWindow())) {
            	String strvalues = String.valueOf(instancesProcessed);

        		double[] selfevaluateResult = learner.PeriodicEvaluate();
	            for (int ii=0; ii<selfevaluateResult.length; ii++) 
	            {
	            	strvalues =strvalues+ ","+ String.valueOf(selfevaluateResult[ii]);
 	            } 
	            selfEvaluateStream.println(strvalues);
            }
            
            
            if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || stream.hasMoreInstances() == false) {
                long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);
                double RAMHoursIncrement = learner.measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                RAMHours += RAMHoursIncrement;
                lastEvaluateStartTime = evaluateTime;
                learningCurve.insertEntry(new LearningEvaluation(
                        new Measurement[]{
                            new Measurement(
                            "learning evaluation instances",
                            instancesProcessed),
                            new Measurement(
                            "evaluation time ("
                            + (preciseCPUTiming ? "cpu "
                            : "") + "seconds)",
                            time),
                            new Measurement(
                            "model cost (RAM-Hours)",
                            RAMHours)
                        },
                        evaluator, learner));

                if (overallEvaluationResultStream != null) {
                    if (firstDump) {
                        overallEvaluationResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    overallEvaluationResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    overallEvaluationResultStream.flush();
                }
            }
            if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                if (monitor.taskShouldAbort()) {
                    return null;
                }
                long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                if (maxInstances > 0) {
                    long maxRemaining = maxInstances - instancesProcessed;
                    if ((estimatedRemainingInstances < 0)
                            || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                        : (double) instancesProcessed
                        / (double) (instancesProcessed + estimatedRemainingInstances));
                if (monitor.resultPreviewRequested()) {
                    monitor.setLatestResultPreview(learningCurve.copy());
                }
                secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                        - evaluateStartTime);
            }               
        }
        
        if (overallEvaluationResultStream != null) {
            overallEvaluationResultStream.close();
        }
        if (predLabelStream != null) {
        	predLabelStream.close();
        }
        if (predProbVectorStream != null) {
        	predProbVectorStream.close();
        }
        if (selfEvaluateStream != null) {
        	selfEvaluateStream.close();
        }

        return learner;
   }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == MicFoal_Task.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
