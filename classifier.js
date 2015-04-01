// naive bayes classifier

var fs = require('fs');
process.stdin.resume();
process.stdin.setEncoding('utf8');
var _input = "";
var dictionary = {};
var english_gkeywords = [];
var count_samples = [];
var count_words = [];
var featureMap = [];
var propability_class = [];
var probability_result = [];
var _count_totalClasses = 8;
var _weightageFactor = 1900;

var writeLine = function(data){
	process.stdout.write(data.toString());
	process.stdout.write("\r\n");
}
var init_english_gkeywords = function(dataset){
	var temp = [];
	//adding indefinite pronouns
	temp.push(["any", "anybody", "anyone", "anything", "both", "each", "either", "everybody", "everyone", "everything", "many", "neither", "nobody", "none", "no one", "one", "other", "others", "some", "somebody", "someone"]);
	//adding helping verbs
	temp.push(["is", "am", "are", "was", "were", "be", "being", "been", "has", "have", "had", "do", "does", "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could"]);
	//adding adverbs
	temp.push(["a", "an", "the" , "my", "our", "your", "their"]);
	//adding adverbs clause
	temp.push(["after", "although", "as", "as if", "before", "because", "if", "since", "so that", "than", "though", "unless", "until", "when", "where", "while"]);
	//adding interrogative pronouns
	temp.push(["how", "when", "where", "how much", "why"]);
	temp.push(["and", "but", "or", "nor", "for", "yet"]);
	temp.push(["I", "she", "he", "we", "they", "my", "mine", "your", "yours", "his", "her", "hers", "its", "our", "ours", "your", "yours", "their", "theirs"])
	temp.push(["myself", "yourself", "yourselves", "himself", "herself", "itself", "ourselves"]);
	temp.push(["us", "that", "those", "in", "to", "also", "of", "said", "not", "it" , "too", "from","for","in","at","by"]);
	temp.every(function(ele,idx,array){
		ele.every(function(ele,idx,array){
			dataset.push(ele);
			return true;
		});
		return true;
	});
	dataset = dataset.sort();
	return dataset;
}	
var sanitize_learnedData = function(tokenList,gWords){
	for(var o in tokenList){
		if(gWords.indexOf(o) > -1 )
			delete tokenList[o];
	}
	return tokenList;
}
var sanitize_check = function(word,gWords){
	var flag = true;
	if(gWords.indexOf(word) > -1){
		flag = false;
	}
	return flag;
}
var normalize_featureSet = function(featureSet,divisor){
	
	for(var feature in featureSet){
		featureSet[feature] = (featureSet[feature] * _weightageFactor)/ (divisor); 
	}
}

process.stdin.on('data',function(data){
	_input += data;
});
process.stdin.on('end',function(){
	var file = fs.readFileSync("trainingdata.txt");
	
	var sentences = file.toString().split('\r\n');
	// count the number of classified text sentences used for the learning phase
	var count_sentences = sentences.shift();
	//writeLine(count_sentences);
	for(var i = 0 ; i < count_sentences ; i++){ 
		// parsing the sentence considering the syntax className_sampleSentence
		// get the class category
		var index_class = parseInt(sentences[i].split(' ')[0]);
		var dictionary = null;
		if(typeof featureMap[index_class] == "undefined") { 
			featureMap[index_class] = {};
			count_samples[index_class] = 0;
			count_words[index_class] = 0;
		}
		dictionary = featureMap[index_class];
		// update the feature set of that class category
		sentences[i].split(' ').every(function(ele,idx,array){
			if(idx == 0) return true; 
			if(typeof dictionary[ele] == "undefined") dictionary[ele.toLowerCase()] = 1;
			else dictionary[ele.toLowerCase()] += 1;
			count_words[index_class] += 1;
			return true;
		});
		// increase the number of count on the samples of a given class category
		count_samples[index_class] += 1; 
	}
	

	var count_totalSamples = 0;
	
	// sanitize each class category and reduces the length of feature Set 
	init_english_gkeywords(english_gkeywords);
	for(var i = 1; i <= _count_totalClasses ; i++) {
		sanitize_learnedData(featureMap[i],english_gkeywords);
		normalize_featureSet(featureMap[i],count_words[i]);

		// same as the total number of sample classified sentences
		count_totalSamples += count_samples[i];
	}

	// calculate the probability of the given classes

	for(var i = 1; i <= _count_totalClasses ; i++) {
		//writeLine(Object.keys(featureMap[i]).length);
		//writeLine("No of Samples collected for " + i + ":" + count_samples[i]);
		//writeLine("length of feature Set:" + Object.keys(featureMap[i]).length);
		propability_class[i] = count_samples[i] / count_totalSamples;
		//writeLine("probabilty of class:" + i + " is "+ propability_class[i]);
		//writeLine("No of Total Words in " + i + ":" + count_words[i]);
	}
	
	// ************running the supervised model on live data, phase begins********************
	var input_sentences = _input.split('\r\n');
	var count_input = input_sentences.shift();
	var count = 0;
	var count_failure = 0;
	input_sentences.every(function(sele,sidx,sarray){
		//evaluate the given string against all the featureset and compare the score 
		
		var scoreboard = [];

		// calculate naive bayes probabilty of a given document for each class 
		for(var classIdx = 1; classIdx <= _count_totalClasses ; classIdx++){
			var set = featureMap[classIdx];
			var prob_score = 1;
			sele.split(' ').every(function(ele,idx,array){
				// santize the input with reduction
				if(!sanitize_check(ele,english_gkeywords)) return true;
				if(typeof set[ele] != "undefined") 
					prob_score = prob_score * set[ele];
				else 
					prob_score = prob_score * (1 /(count_words[classIdx]));
				//writeLine("Prob_score: "+ classIdx + "-" + prob_score);
				return true;
			});
			scoreboard[classIdx] = prob_score * propability_class[classIdx];
		}

		// find the maximum naive bayes class probability for the given document
		var maxScore = 0;
		var idx = 0;
		for(var i = 1; i <= _count_totalClasses; i++){
			if(scoreboard[i] > maxScore) {
				maxScore = scoreboard[i];
				idx = i;
			}
		}
		if(idx == 0) count_failure++;
		count++;
		writeLine(idx);
		
		return true;
	});
	writeLine("Total Sentences classified:" + (count - count_failure));
});
