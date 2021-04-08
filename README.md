# The Crisis of Bureaucracy
Repo for EUI Graduate Conference Proposal

### Case
Bureaucratic arguments and the origins of the neoliberal turn in British political thought between 1957 and 1985. 

### Hypothesis
A convergence in the different types of arguments involving "bureaucracy" appears from the years of the first Wilson gov. (1964-1970) onwards. This entails:    - 
- [ ] the integration of for example bureaucracy-as-totalitarianism and bureaucracy-as-inefficient into a master narrative structured by the notion of crisis
- [ ] the spread of these arguments across the political spectrum (to be more specific: not only conservative actors)
- [ ] the association of these arguments with concrete examples that refer back to a state of general crisis, instead of the more volatile and rhetoric use of "bureaucracy".

### Methods

- [x] Frequencies
  - [x] unigrams bureaucra*
	  - _show correlation with coalition/opposition role and changes in this relation in the 1970s_
  - [x] bigrams bureaucra*  
- [x] PMI
	- _Top terms show the 'statification' of bureaucracy-arguments_
- [ ] Word2Vec distribution of bureauc*, with party differentation
  - [ ] semantic shift per year (general)
  - [ ] semantic neighbourhood shift per year [code](https://gist.github.com/quadrismegistus/15cafbdd878a98b060ef910c843fcf5a)
- [ ] An argument classifier.

### Approach
1. Map frequencies of bureaucra* ngrams per party
2. PMI similarities
3. Word2Vec similarities
4. Argument Classification
