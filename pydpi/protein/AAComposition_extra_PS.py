# -*- coding: utf-8 -*-
"""
###############################################################################

The module is used for computing the composition of amino acids, dipetide and

3-mers (tri-peptide) for a given protein sequence. You can get 8420 descriptors

for a given protein sequence. You can freely use and distribute it. If you hava

any problem, you could contact with us timely!

References:

[1]: Reczko, M. and Bohr, H. (1994) The DEF data base of sequence based protein

fold class predictions. Nucleic Acids Res, 22, 3616-3619.

[2]: Hua, S. and Sun, Z. (2001) Support vector machine approach for protein

subcellular localization prediction. Bioinformatics, 17, 721-728.


[3]:Grassmann, J., Reczko, M., Suhai, S. and Edler, L. (1999) Protein fold class

prediction: new methods of statistical classification. Proc Int Conf Intell Syst Mol

Biol, 106-112.

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.3.27

Email: oriental-cds@163.com

###############################################################################
"""

import re

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
#############################################################################################
def ComputeMeanVar(dict1, dict2, str):
	#dict 1 is AAC
	#dict 2 is values

	mean = 0
	for k, v in dict1.items():
  		mean = mean + v * dict2[k] / 100
	var = 0
	for k, v in dict1.items():
  		var = var + v*((dict2[k] - mean)**2) / 100

	return {"mean" + str: mean, "var" + str: var}

def CalculateAAComposition(ProteinSequence):

	"""
	########################################################################
	Calculate the composition of Amino acids

	for a given protein sequence.

	Usage:

	result=CalculateAAComposition(protein)

	Input: protein is a pure protein sequence.

	Output: result is a dict form containing the composition of

	20 amino acids.
	########################################################################
	"""
	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		Result[i]=round(float(ProteinSequence.count(i))/LengthSequence*100,3)
	#modificaton so that it adds some more features
	_Hydrophobicity={"A":0.62,"R":-2.53,"N":-0.78,"D":-0.90,"C":0.29,"Q":-0.85,"E":-0.74,"G":0.48,"H":-0.40,"I":1.38,"L":1.06,"K":-1.50,"M":0.64,"F":1.19,"P":0.12,"S":-0.18,"T":-0.05,"W":0.81,"Y":0.26,"V":1.08}
	res = ComputeMeanVar(Result,_Hydrophobicity,'Hydrophob')
	_hydrophilicity={"A":-0.5,"R":3.0,"N":0.2,"D":3.0,"C":-1.0,"Q":0.2,"E":3.0,"G":0.0,"H":-0.5,"I":-1.8,"L":-1.8,"K":3.0,"M":-1.3,"F":-2.5,"P":0.0,"S":0.3,"T":-0.4,"W":-3.4,"Y":-2.3,"V":-1.5}
	res.update(ComputeMeanVar(Result,_hydrophilicity,'Hydrophil'))
	_pK1={"A":2.35,"C":1.71,"D":1.88,"E":2.19,"F":2.58,"G":2.34,"H":1.78,"I":2.32,"K":2.20,"L":2.36,"M":2.28,"N":2.18,"P":1.99,"Q":2.17,"R":2.18,"S":2.21,"T":2.15,"V":2.29,"W":2.38,"Y":2.20}
	res.update(ComputeMeanVar(Result,_pK1,'_pK1'))
	_pK2={"A":9.87,"C":10.78,"D":9.60,"E":9.67,"F":9.24,"G":9.60,"H":8.97,"I":9.76,"K":8.90,"L":9.60,"M":9.21,"N":9.09,"P":10.6,"Q":9.13,"R":9.09,"S":9.15,"T":9.12,"V":9.74,"W":9.39,"Y":9.11}
	res.update(ComputeMeanVar(Result,_pK2,'_pK2'))
	_pI={"A":6.11,"C":5.02,"D":2.98,"E":3.08,"F":5.91,"G":6.06,"H":7.64,"I":6.04,"K":9.47,"L":6.04,"M":5.74,"N":10.76,"P":6.30,"Q":5.65,"R":10.76,"S":5.68,"T":5.60,"V":6.02,"W":5.88,"Y":5.63}
	res.update(ComputeMeanVar(Result,_pI,'_pI'))
	res.update(Result)

	return res

	#modification ends
	#return Result

#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):
	"""
	########################################################################
	Calculate the composition of dipeptidefor a given protein sequence.

	Usage:

	result=CalculateDipeptideComposition(protein)

	Input: protein is a pure protein sequence.

	Output: result is a dict form containing the composition of

	400 dipeptides.
	########################################################################
	"""

	LengthSequence=len(ProteinSequence)
	Result={}
	for i in AALetter:
		for j in AALetter:
			Dipeptide=i+j
			Result[Dipeptide]=round(float(ProteinSequence.count(Dipeptide))/(LengthSequence-1)*100,2)
	return Result



#############################################################################################

def Getkmers():
	"""
	########################################################################
	Get the amino acid list of 3-mers.

	Usage:

	result=Getkmers()

	Output: result is a list form containing 8000 tri-peptides.

	########################################################################
	"""
	kmers=list()
	for i in AALetter:
		for j in AALetter:
			for k in AALetter:
				kmers.append(i+j+k)
	return kmers

#############################################################################################
def GetSpectrumDict(proteinsequence):
	"""
	########################################################################
	Calcualte the spectrum descriptors of 3-mers for a given protein.

	Usage:

	result=GetSpectrumDict(protein)

	Input: protein is a pure protein sequence.

	Output: result is a dict form containing the composition values of 8000

	3-mers.
	########################################################################
	"""
	result={}
	kmers=Getkmers()
	for i in kmers:
		result[i]=len(re.findall(i,proteinsequence))
	return result

#############################################################################################
def CalculateAADipeptideComposition(ProteinSequence):

	"""
	########################################################################
	Calculate the composition of AADs, dipeptide and 3-mers for a

	given protein sequence.

	Usage:

	result=CalculateAADipeptideComposition(protein)

	Input: protein is a pure protein sequence.

	Output: result is a dict form containing all composition values of

	AADs, dipeptide and 3-mers (8420).
	########################################################################
	"""

	result={}
	result.update(CalculateAAComposition(ProteinSequence))
	result.update(CalculateDipeptideComposition(ProteinSequence))
	result.update(GetSpectrumDict(ProteinSequence))

	return result
#############################################################################################
if __name__=="__main__":

	protein="ADGCGVGEGTGQGPMCNCMCMKWVYADEDAADLESDSFADEDASLESDSFPWSNQRVFCSFADEDAS"

	AAC=CalculateAAComposition(protein)
	print (AAC)
	DIP=CalculateDipeptideComposition(protein)
	print (DIP)
	spectrum=GetSpectrumDict(protein)
	print (spectrum)
	res=CalculateAADipeptideComposition(protein)
	print (len(res))
