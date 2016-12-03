REM Creating two big tsv files to train/test on. Also creating the prop file.
python ..\ws\tsv_converter.py

REM Running the training on the created tsv and prop file. 2>training_log.txt
java -mx1g -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -prop classifier.prop 2>training_log.txt

REM Testing the classifier on the test tsv file. Matching log contains the probabilities. -printLabelValue > dump.txt -printClassifier HighWeight -map word=0,answer=1
java -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -testFile test_tsv\test_data 1>test_matching_log.txt -printProbs 2> nul

REM running the previous command again, without printing the probabilities. This way, the test_summary will contain the overall score.
java -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -testFile test_tsv\test_data 2>test_summary.txt 1> nul

REM Using the classifier on unseen data. outputFormat: xml, inlineXML, tsv, taggedEntities, slashTags
java -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -textFile unlabelled_data\unlabelled_data 2>unseen_labelling_info.txt -outputFormat tsv > unseen_labelled.txt

REM Outputs: training_log.txt -> Results of training.
REM          test_summary.txt -> Summary about the success of the classifier.
REM          test_matching_log.txt -> A long list of tokens with the predicted and the real labels.
REM          unseen_labelling_info.txt -> Some general information about the labelling of unseen data.
REM          unseen_labelled.txt -> Labelled unseen data.

pause