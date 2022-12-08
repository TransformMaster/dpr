Since the original dataset and dense embedding files are too large, it takes too much time on my machine to compress them. So I only uploaded the final result files in tsv format in the repository (in ./result folder).

>You can generate dense embeddings through bert_model.py, and you need to change the name of dataset in bert_model.py (if your file have different filename).

>You can train FAISS using dense embeddings through generate_guess.py, and generate guesses on test dataset use that file.



For the result files:

>For 256centroid.tsv, it is generate with 1probe setting with 256 cell. Each column in the tsv file means:
>qid, pid, rank, cell_id

>For pqivf.tsv, it is generate with 1probe setting with 256 cell and product quantization. Each column in the tsv file means:
>qid, pid, rank, cell_id

>For other file, each column means:
>qid, pid, rank

The analyze.py file contains the function (check_cell) to compute how many correct answers are in different cells with FAISS search, and it also contains the function(check_rank) to check how many FAISS answers have lower rank than exact search.

  
