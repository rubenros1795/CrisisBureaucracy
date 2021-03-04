# The Crisis of Bureaucracy
Repo for EUI Graduate Conference Proposal

### Case

### Methods

- [ ] Frequencies
  - [ ] unigrams bureaucra*
  - [ ] bigrams bureaucra*  
- [ ] PMI (corrected, see [link](https://www.scitepress.org/Papers/2011/36551/36551.pdf))
  - [ ] bureaucra* context
- [ ] Word2Vec distribution of bureauc*, with party differentation
  - [ ] semantic shift per year (general)
  - [ ] semantic neighbourhood shift per year [code](https://gist.github.com/quadrismegistus/15cafbdd878a98b060ef910c843fcf5a)

### Approach
1. Map frequencies of bureaucra* ngrams per party
2. Close read samples of bureauc* windows / year to categorize arguments
3. Verify findings with PMI similarities
4. Verify findings with Word2Vec similarities
5. Train USE (sentence embeddings) classifier on argumentation types.
