import os
import json
import re
import nltk
import copy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.text import Text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
import itertools
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from collections import Counter
import objectpath

filteredContent = {}
lemmatizer = WordNetLemmatizer()

# Method for Lemmatization:


def lemmatizeWord(word, posTag):
    return lemmatizer.lemmatize(word, pos=posTag)


# Method to trim Dictionary
def trimDictionary(dictionary, count):
    newDictionary = {}
    newDictionary = copy.deepcopy(dictionary)
    for key, value in dictionary.items():
        k = 0
        while k < count:
            if key[k] in stopwords.words('english'):
                newDictionary.pop(key, None)
            k += 1
    return newDictionary


# Method to construct Bi-gram Frequency Distribution
def collocations(text):
    # collocationDistribution = {}
    bigramDict = {}
    unigramDict = {}
    sentences = text.split('.')
    length = len(sentences)
    k = 0
    for sentence in sentences:
        tokenizedSentence = nltk.word_tokenize(sentence)
        # print("TOKEN SEN = ", tokenizedSentence)
        bigrams = list(nltk.bigrams(tokenizedSentence))

        # print("BIGRAM = ", bigram)

        for entry in bigrams:
            # keyBi = str(entry).lower()
            keyBi = entry
            if keyBi not in bigramDict:
                bigramDict[keyBi] = 1
            bigramDict[keyBi] += 1

        for word in tokenizedSentence:
            # keyUni = str(word).lower()
            keyUni = word
            if keyUni not in unigramDict:
                unigramDict[keyUni] = 1
            unigramDict[keyUni] += 1

    dictTotal = {}
    dictTotal['uni'] = unigramDict
    dictTotal['bi'] = bigramDict

    return dictTotal


# Method 'jsonToText' to extract text values from the JSON Object
def jsonToText(object):
    lengthOfJson = len(dataPointJson)
    j = 0
    text = []
    nameList = []
    idList = []

    while j < lengthOfJson:
        # Extracting valid sentences and text data into 'text' for each Data
        # Point (JSON Object)
        nameValue = dataPointJson[j].get("name")
        idValue = dataPointJson[j].get("id")
        descriptionValue = dataPointJson[j].get("description")
        definitionValue = dataPointJson[j].get("definition")
        if idValue is not None and nameValue is not None:
            nameList.append(dataPointJson[j].get("name"))
            idList.append(dataPointJson[j].get("id"))
        if nameValue[-1:] != '.':
            nameValue += ". "
        if descriptionValue[-1:] != '.':
            descriptionValue += ". "

        # idValue = dataPointJson[j].get("id")
        text.append(nameValue)
        text.append(descriptionValue)
        if definitionValue is not None:
            if definitionValue[-1:] != '.':
                definitionValue += " . "
            text.append(definitionValue)

        # nameNidDict['id'].append(dataPointJson[j].get("id"))
        # nameNidDict['name'].append(dataPointJson[j].get("name"))

        j += 1

    return text, nameList, idList


# Method 'dataPointTextExtractor' to import json content
def jsonExtractor(filePath):
    source = open(filePath, 'r')
    jsonDataPoint = json.load(source)
    return jsonDataPoint

# Method 'regexExtractor' to remove non-conforming patterns


def regexExtractor(text):
    excludedNumerals = set()
    excludedNotAlphanumeric = set()
    excludedSeeMissingHyphenated = set()
    excludedSawMissingHyphenated = set()
    cleansedText = []

    for line in text:

        numeralRegex = re.compile(r' [0-9]+ ')
        postNumeralRegex = re.sub(numeralRegex, ' ', line)
        if len(numeralRegex.findall(line)) > 0:
            excludedNumerals.update(numeralRegex.findall(line))
    # print("Excluded Numerals = ", excludedNumerals)

        notAlphanumericRegex = re.compile(r'[^A-Z0-9a-z-\. ]')
        postNotAlphanumericRegex = re.sub(
            notAlphanumericRegex, ' ', postNumeralRegex)
        # print(notAlphanumericRegex.findall(postNumeralRegex))
        if len(notAlphanumericRegex.findall(postNumeralRegex)) > 0:
            excludedNotAlphanumeric.update(
                notAlphanumericRegex.findall(postNumeralRegex))

    # print("Excluded Non Alphanumeric content = ", excludedNotAlphanumeric)

        seeMissingRegex = re.compile(r' -')
        postSeeMissingRegex = re.sub(
            seeMissingRegex, ' ', postNotAlphanumericRegex)
        if len(seeMissingRegex.findall(postNotAlphanumericRegex)) > 0:
            excludedSeeMissingHyphenated.update(
                seeMissingRegex.findall(postNotAlphanumericRegex))
    # print("Excluded See Missing Hyphenations = ",
    # excludedSeeMissingHyphenated)

        sawMissingRegex = re.compile(r'- ')
        postSawMissingRegex = re.sub(sawMissingRegex, ' ', postSeeMissingRegex)
        if len(sawMissingRegex.findall(postSeeMissingRegex)) > 0:
            excludedSawMissingHyphenated.update(
                sawMissingRegex.findall(postSeeMissingRegex))

    # print("Excluded Saw Missing Hyphenations = ",
    # excludedSawMissingHyphenated)

        cleansedText.append(re.sub('\s+', ' ', postSawMissingRegex))
    # cleansedText = re.sub('\s+', ' ', postSawMissingRegex)

    # print("VALUE = ", cleansedText)
    filteredContent['excludedNumerals'] = excludedNumerals
    filteredContent['excludedNotAlphanumeric'] = excludedNotAlphanumeric
    filteredContent[
        'excludedSeeMissingHyphenated'] = excludedSeeMissingHyphenated
    filteredContent[
        'excludedSawMissingHyphenated'] = excludedSawMissingHyphenated
    cleanRegex = ' '.join(cleansedText)
    filteredContent['cleansedText'] = cleanRegex
    return filteredContent


def removeDuplicate(dpVocab):
    j = 0
    while j < len(dpVocab):
        k = len(dpVocab) - 1
        while k > j:
            if dpVocab[j]['word'].lower() == dpVocab[k]['word'].lower() and dpVocab[j]['postag'].lower() == dpVocab[k]['postag'].lower():
                dpVocab.pop(k)
            k = k - 1
        j = j + 1
    return dpVocab


def splitString(joinedString):
    fo = re.compile(r'[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    fi = fo.findall(joinedString)
    result = ''
    for var in fi:
        result += var + ' '
    return result


def namedEntityRecognition(pos):
    chunked_token = ne_chunk(pos)
    named_entity = tree2conlltags(chunked_token)
    return named_entity


def stopWordsRemove(word):
    if word.lower() in stopwords.words('english') and re.search(r'no+', word.lower()) is None:
        return word


def preProcessVocab(clean_text):
    nounList = ["NN", "NNS"]
    adverbList = ["RB", "RBR", "RBS"]
    adjectiveList = ["JJ", "JJR", "JJS"]
    verbList = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    stemmer = PorterStemmer()

    vocabularyEntry = {}
    dpVocab = []

    stopWordsList = set()

    sentences = nltk.sent_tokenize(clean_text)
    j = 0
    for sentence in sentences:
        # sentenceNew = splitString(sentence)
        tokenList = nltk.word_tokenize(sentence)
        partOfSpeech = nltk.pos_tag(tokenList)
        ne_token = namedEntityRecognition(partOfSpeech)
        # print(ne_token)
        # print(partOfSpeech)
        k = 0
        for token in tokenList:
            stpWord = stopWordsRemove(token)

            if stpWord is not None:
                stopWordsList.add(stpWord)
                k += 1
                continue

            isNamedEntity = False

            identifier = str(j + 1) + "." + str(k + 1)

            stemValue = stemmer.stem(token)
            posValue = ne_token[k][1]

            if posValue in nounList:
                lemmaValue = lemmatizeWord(token, 'n')
            elif posValue in adverbList:
                lemmaValue = lemmatizeWord(token, 'r')
            elif posValue in adjectiveList:
                lemmaValue = lemmatizeWord(token, 'a')
            elif posValue in verbList:
                lemmaValue = lemmatizeWord(token, 'v')
            else:
                lemmaValue = lemmatizer.lemmatize(token)
            if posValue == 'NNP' or posValue == 'NNPS':
                isNamedEntity = True

            vocabularyEntry['id'] = identifier
            vocabularyEntry['word'] = token
            vocabularyEntry['postag'] = posValue
            vocabularyEntry['lemma'] = lemmaValue
            vocabularyEntry['stem'] = stemValue
            vocabularyEntry['isnamedentity'] = isNamedEntity

            # Creating Vocabulary Entry Object. We can further reduce the JSON
            # Output by removing repeating JSON Objects, which is not
            # implemented yet.
            dpVocab.append(dict(vocabularyEntry))
            k += 1
        j += 1
    return dpVocab, stopWordsList


def nameIndexing(outFile, nameList, idList):
    dpVocabOutputJson = jsonExtractor(output_file)
    # newText = regexExtractor(nameList)
    # nameClean = newText.get('cleansedText')
    # dpNameVocab, stopNameWordsList = preProcessVocab(nameClean)
    tree = objectpath.Tree(dpVocabOutputJson)
    if len(nameList) == len(idList):
        j = 0
        nameIndexDict = {}
        while j < len(idList):
            nameIndexDict['dp-id'] = idList[j]
            nameIndexDict['original-phrase'] = nameList[j]
            nameTokens = [word
                          for word in nltk.word_tokenize(nameList[j])
                          if word not in string.punctuation and word.lower() not in stopwords.words('english')]
            idTokens = []
            for name in nameTokens:
                print(name, " , ")
                result = tree.execute("$..*[@.word is " + name + "]")
                for entry in result:
                    print(entry["id"], " ++++ ")
                    if entry["id"] is None:
                        break
                    idTokens.append(entry["id"])
            nameIndexDict['encoded-phrase'] = idTokens

            j += 1
    return

    # print(len(dpNameVocab))
    # dictName = {dpNameVocab[i]: dpNameVocab[i + 1] for i in range(0, len(dpNameVocab), 2)}
    # print(dictName["word"])
    ###################################################
    # bigram = Phrases()
    # sentences = []
    # for name in nameList:
    #     sentence = [word
    #                 for word in nltk.word_tokenize(name.lower())
    #                 if word not in string.punctuation]
    #     sentences.append(sentence)
    #     bigram.add_vocab([sentence])
    # print(list(bigram[sentences])[:5])
    # bigram_model_counter = Counter()

    # bigram_model = Word2Vec(bigram[sentences], size=100)

    # for key in bigram_model.wv.vocab.keys():
    #     if key not in stopwords.words("english"):
    #         if len(key.split("_")) > 1:
    #             bigram_model_counter[key] += bigram_model.wv.vocab[key].count

    # for key, counts in bigram_model_counter.most_common(50):
    #     print('{0: <20} {1}'.format(key, counts))
    #############################################################


if __name__ == '__main__':

    # input_path = input("\nEnter the location of the input file: \n")
    # output_path = input("\nEnter the location of the output file: \n")
    input_path = "C:\\Users\\adity\\Documents\\GitHub\\MSTR_\\week-4-oct14"
    # print(input_path)
    output_path = "C:\\Users\\adity\\Documents\\GitHub\\MSTR_\\week-4-oct14"
    input_file = os.path.join(input_path, 'dp-docs.json')
    output_file = os.path.join(output_path, 'dp-vocab_a.json')

    Filtered_file = os.path.join(output_path, 'Ommision_list_a.txt')

    # Data Points info of Json content imported into 'dataPointJson'
    dataPointJson = jsonExtractor(input_file)
    allText, nameList, idList = jsonToText(dataPointJson)
    parsedOutput = regexExtractor(allText)
    clean = parsedOutput.get('cleansedText')
    dpVocab, stopWordsList = preProcessVocab(clean)
    dpVocab = removeDuplicate(dpVocab)

    filteredContent['stopWords'] = stopWordsList
    # print(filteredContent['stopWords'])

    dpVocabJson = json.dumps(dpVocab, indent=4)
    with open(output_file, 'w') as outfile:
        outfile.write(dpVocabJson)

    with open(Filtered_file, 'w') as outfile:
        outfile.write(str(filteredContent['excludedNumerals']))
        outfile.write(str(filteredContent['excludedSeeMissingHyphenated']))
        outfile.write(str(filteredContent['excludedNotAlphanumeric']))
        outfile.write(str(filteredContent['excludedSawMissingHyphenated']))
        outfile.write(str(filteredContent['stopWords']))

    nameIndexing(output_file, nameList, idList)

    # print("lemmatized = ", lemmatized)
    # print("POS = ", partOfSpeech)

    """
	NOUN = NN, NNS
	Adjective = JJ, JJR, JJS
	Adverb = RB, RBR, RBS
	Verb = VB, VBD, VBG, VBN, VBP, VBZ

	dictTotal = collocations(clean)
	uniList = dictTotal.get('uni')
	biList = dictTotal.get('bi')

	cleanBiList = trimDictionary(biList, 2)
	cleanUniList = trimDictionary(uniList, 1)
	# print("UNI = ", uniList)
	print("BI = ", sorted(cleanBiList.items(), key=lambda kv: kv[1], reverse=True))
	# print("UNI = ", sorted(cleanUniList.items(), key=lambda kv: kv[1], reverse=True))
	"""
