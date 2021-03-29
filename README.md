# The Crisis of Bureaucracy
Repo for EUI Graduate Conference Proposal

### Case
Bureaucratic arguments and the origins of the neoliberal turn in British political thought between 1957 and 1985. 

### Hypothesis
A convergence in the different types of arguments involving "bureaucracy" appears from the years of the first Wilson gov. (1964-1970) onwards. This entails:    - 
- [ ] the integration of for example bureaucracy-as-totalitarianism and bureaucracy-as-inefficient into a master narrative structured by the notion of crisis
- [ ] the spread of these arguments across the political spectrum (to be more specific: not only conservative actors)
- [ ] the association of these arguments with concrete examples that refer back to a state of general crisis, instead of the more volatile and rhetoric use of "bureaucracy" (perhaps measured by looking at share of adverbs/adjectives in context?)

### Methods

- [ ] Frequencies
  - [ ] unigrams bureaucra*
  - [ ] bigrams bureaucra*  
- [ ] PMI (corrected, see [link](https://www.scitepress.org/Papers/2011/36551/36551.pdf))
  - [ ] bureaucra* context
- [ ] Word2Vec distribution of bureauc*, with party differentation
  - [ ] semantic shift per year (general)
  - [ ] semantic neighbourhood shift per year [code](https://gist.github.com/quadrismegistus/15cafbdd878a98b060ef910c843fcf5a)
- [ ] Sentence Embeddings (USE) for classification.

### Approach
1. Map frequencies of bureaucra* ngrams per party
2. Close read samples of bureauc* windows / year to categorize arguments
3. Verify findings with PMI similarities
4. Verify findings with Word2Vec similarities
5. Train USE (sentence embeddings) classifier on argumentation types.
