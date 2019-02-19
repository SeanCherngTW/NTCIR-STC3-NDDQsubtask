from gensim.models.keyedvectors import KeyedVectors
doclen = 150
embedding_path = "../embedding/STCWiki/STCWiki_mincount0.model.bin"
# embsize = KeyedVectors.load(embedding_path)['a'].shape[0]
embsize = 100
max_sent = 7
NDclasses = 7
DQclasses = 5
# sentembsize = 4800
sentembsize = 2400
