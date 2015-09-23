# gLM 0.1
gLM the GPU based Language Model is an ngram language model implementation that takes in an arpa file as an input, binarizes it and queries it in batch.

## Build
```bash
git clone https://github.com/XapaJIaMnu/gLM.git
cd gLM
mkdir release_build
cd release_build
cmake .. -DBUILDTYPE=release #Necessary for -O3 compilation otherwise it would be slow
make -j4
make test #Requires CUDA for GPU testing
```

## Binarize arpa files
```bash
cd path_to_glm/release_build/bin
./binarize path_to_arpa_file output_path [btree_node_size] [[CompactMode]]
```
*btree_node_size* should be an odd number. Personally I found that 31 works best, but you should experiment. The number could vary with different size arpa files and different GPUs

If *compactMode* is not set, or is set to *0* gLM will use memory mapped IO for the model. If *compactMode* is set, the model is going to use boost.serialization over bzip2 compressed stream, which will result in about 3 times more compact model, but the loading time could increase 10-80 fold depending on your CPU and your IO system.

## Batch query
To benchmark gLM in batch setting do:
```bash
cd path_to_glm/release_build/bin
./batch_query path_to_binary_lm path_to_text_file [add_begin_end_markers]
```
This will calculate the perplexity of a text file. If *add_begin_end_markers* is set to 0, the begin of sentence and end of sentence tokens (<s> and </s>) will not surround every sentence.

## Preliminary results
So... Everything started running correctly. A (preliminary) benchmark against KenLM (Titan X vs core i7 4720HQ)

| LM  | ngram queries per second | model info          |
|.....|:........................:| -------------------:|
|KenLM| 10 274 237               |3.3G, 88720517 ngrams|
|gLM  | 34 011 267               |3.3G, 88720517 ngrams|


Please bear in mind that this is not a completely fair benchmark as KenLM's benchmark is single threaded and the queries are being batched, which is usually not the case. However in batch setting gLM should outperform kenLM.