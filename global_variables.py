

data_dir = 'data'
mappings_dir = 'mappings'


#in_file = 'first-500.pkl'
in_file = 'first-5000.pkl'
#in_file = 'first-10000.pkl'
#in_file = 'whole-pt.pkl'

min_word_count = 2
max_word_len = 30

total_skip_top = 0
total_keep_words = 1000

keep_tags = 20

start_char = 1
#oov_char = 2
index_from = 3
max_seq_len = 200
truncating = 'pre'
padding = 'pre'

vocab_size = total_keep_words - total_skip_top + index_from
