REM Creating two big tsv files to train/test on. Also creating the prop file.
python ..\ws\tsv_converter.py

REM Running the training on the created tsv and prop file. 2>training_log.txt
java -mx1g -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -prop classifier.prop 2>training_log.txt

REM Testing the classifier on the test tsv file.
java -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -testFile test_tsv\test_data 1>matching_log.txt 2>test_summary.txt

REM Outputs: training_log.txt -> Results of training.
REM 		 test_summary.txt -> Summary about the success of the classifier.
REM 		 matching_log.txt -> A long list of tokens with the predicted and the real labels.

pause