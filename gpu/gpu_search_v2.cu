#include "gpu_search_v2.hh"
#include "gpu_common.h"
#include "memory_management.hh"

#define big_entry 16
#define small_entry 8

struct identity {
    __device__ void operator()(float& num) {
        return;
    }
};

struct exponentify {
    __device__ void operator()(float& num) {
        num = expf(num);
    }
};


template<unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram, class Functor>
__global__ void gpuSearchBtree(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, float * results, Functor fn) {

    __shared__ unsigned int offsets[max_num_children/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries_actual[entries_per_node + 1];
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[max_ngram]; //Each block fetches from shared memory the max necessary number of keys

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < max_ngram) {
       keys_shared[i] = keys[(blockIdx.x*max_ngram) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    if (i == 0) {
        //Initialize shared memory for search. We write the entries (keys) from position 1 to n and put
        //0 at position 0 of the actual array. This allows us to skip a case when doing an nary search.
        //Potentially we could set the entries_actual[num_entries +1] element to UINT_MAX and then by
        //using an extra thread skip another divergence case (which will be moved to the memory copy part)
        //Not sure if it's worth it cause it requires a rewrite of the btree part
        entries_actual[0] = 0;
    }
    __syncthreads();

    unsigned int * entries = &entries_actual[1];

    //Setup global memory for convenience
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];

    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];

    unsigned int * next_level = &payload[0];
    float * prob = (float *)&payload[1];
    float * backoff = (float *)&payload[2];

    //Backoff variables
    unsigned int match_length_found = 0; //To check what was our longest match so we know what to backoff to
    float accumulated_score = 0;
    bool get_backoff = false; //Are we looking to extract backoff or prob from our ngram

    //First get the value from first_lvl
    unsigned int current_ngram = 0;
    unsigned int key = keys_shared[current_ngram];

    /* When using gLM to score ngrams for NMT frequently sentences in batches are padded with zeroes so they can be at the same length
    * the easiest way to get corresponding behaviour is to allow gLM to submit bogus scores (e.g. 0) for them in case the first ngram
    * in the query is zero. Hence this ugly goto which will bypass the btree code. Unfortunately we pay for this with about 0.001% drop
    * in throughput ;/
    */
    if (key != 0) {

        //Backoff logic
        backoff_part2:
        if (get_backoff) {
            accumulated_score += *prob; //We add the longest match of probability we found.
            match_length_found = current_ngram - 1; //The length of the match found. We need to backoff from toplevel to here
            current_ngram = 1; //Set backoff in -1. If we run into this case again we need to do nothing
            key = keys_shared[current_ngram];
            get_backoff = true;
        }
        __syncthreads(); //Needed!
        if (i < 3) {
            payload[i] = first_lvl[(key-1)*3 + i];
            //If this is the last non zero ngram  no need to go to the btree_trie. We already have
            //the payload value
        }
        __syncthreads();
        if (i == 0) {
            if (get_backoff && match_length_found <= current_ngram) {
                accumulated_score += *backoff;
            } else if (keys_shared[current_ngram + 1] == 0) {
                accumulated_score += *prob;
            }
        }
        __syncthreads();

        //Set the start index
        uint64_t current_btree_start = *next_level*4;
        current_ngram++;
        key = keys_shared[current_ngram];

        //Some necessary variables
        uint64_t updated_idx;
        unsigned int size;

        //Current_btree_start == 0 means that we had UNK key (vocabID 1 which wasn't found, so we should directly go to backoff
        //@TODO we can check if key[0] == 1 when we get the score too
        if (current_btree_start == 0 && key != 0) {
            goto backoff_notriecont;
        }

        while ((key != 0 && current_ngram < max_ngram - 1 && current_btree_start != 0) || 
            (get_backoff && key != 0 && current_ngram < max_ngram && current_btree_start != 0)) {
            current_ngram++;
            updated_idx = current_btree_start + 4; //Update the index for the while loop
            //@TODO consider this for shared memory as oppposed to global mem broadcast to register
            size = *(unsigned int *)&btree_trie_mem[current_btree_start]; //The size of the current node to process.

            //Initialize shared variable
            if (i < 2) {
                booleans[i] = false; //Uset *exact_match and *is_last
            }
            __syncthreads();

            //Traverse the current Btree
            while (!*exact_match) {
                //First warp divergence here. We are reading in from global memory
                if (i == 0) {
                    //@TODO: Replace this with a mod check
                    int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(big_entry + sizeof(unsigned short));
                    *is_last = !(entries_per_node == cur_node_entries);
                }
                __syncthreads();

                int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.
                if (*is_last) {
                    //The number of entries in the bottom most nodes may be smaller than the size
                    num_entries = size/big_entry;
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                } else {
                    num_entries = entries_per_node;
                    //Now load the entries
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }

                    //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                    if (i < (max_num_children/2) + 1) {
                        offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
                    }
                }
                __syncthreads();

                //NOW search
                if (i < num_entries) {
                    if (key > entries_actual[i] && key <= entries_actual[i + 1]){
                        found_idx = i;
                        if (key == entries_actual[i + 1]) {
                            *exact_match = true;
                        }
                    }
                } else if (i == num_entries) {
                    //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                    if (key > entries_actual[i]) {
                        found_idx = i;
                    }
                }
                __syncthreads();

                //We found either an exact match (so we can access next level) or at least an address to next btree level
                if (!*exact_match && !*is_last) {
                    //Calculate the address and the size of the next child
                    updated_idx += *first_child_offset*4;
                    if (found_idx == 0) {
                       size = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it
                    } else {
                        updated_idx += offests_incremental[found_idx - 1]*4;
                        size = (offests_incremental[found_idx] - offests_incremental[found_idx - 1])*4;
                    }
                    __syncthreads();
                } else if (*is_last && !*exact_match) {
                    //In this case we didn't find the key that we were looking for
                    //What we should do is get the probability of the last node that we found
                    //The last node that we found's probability should be in shared memory
                    backoff_notriecont:
                    if (get_backoff) {
                        current_ngram = max_ngram;
                        break; //If we didn't find a backoff, the value is zero; //We should go to end now, because any further backoffs
                        // will be missing from the trie
                    } else {
                        get_backoff = true;
                        __syncthreads(); //Necessary
                        goto backoff_part2;
                    }
                } else {
                    //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
                    if (i < 3) {
                        //What we are doing here is reading the correct memory location for our payload. The payload is found
                        //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                        if (*is_last) {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                    + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                        } else {
                            payload[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                                + num_entries*sizeof(unsigned int) //Skip the keys
                                    + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                        + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                        }
                    }

                    key = keys_shared[current_ngram]; //@TODO this might be illegal memory access
                    __syncthreads();

                    current_btree_start = current_btree_start + *next_level*4;

                    //Very rarely, mostly when having big datasets with small vocabulary
                    //we will have tries that don't go to the last level. In this case
                    //we just need to initiate backoff
                    if (*next_level == 0 && key != 0) {
                        current_ngram++; //We need to add one to the current_ngram because we actually found a match on this trie level
                        goto backoff_notriecont; //it's the next trie level we are missing so in effect we say that this is the longest
                    }                            //match and we need to calculate the backoff for the rest, similar to the case in the last_level


                    if (get_backoff) {
                        if (match_length_found < current_ngram) {
                            accumulated_score += *backoff;
                        }
                    } else if (key == 0) {
                        accumulated_score += *prob;
                    }

                    break;
                }
            }
        }
        //Now fetch the last level if the key is not 0 or we backed off
        //key = keys_shared[current_ngram]; We already set the next key
        if (!get_backoff && key != 0) {
            updated_idx = current_btree_start + 4; //Update the index for the while loop
            //@TODO consider this for shared memory as oppposed to global mem broadcast to register
            size = *(unsigned int *)&btree_trie_mem[current_btree_start]; //The size of the current node to process.

            //Initialize shared variable
            if (i < 2) {
                booleans[i] = false;
            }
            __syncthreads();

            //Traverse the current Btree
            while (!*exact_match) {
                //First warp divergence here. We are reading in from global memory
                if (i == 0) {
                    //@TODO: Replace this with a mod check
                    int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(small_entry + sizeof(unsigned short));
                    *is_last = !(entries_per_node == cur_node_entries);
                }
                __syncthreads();

                int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.
                if (*is_last) {
                    //The number of entries in the bottom most nodes may be smaller than the size
                    num_entries = size/small_entry;
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                } else {
                    num_entries = entries_per_node;
                    //Now load the entries
                    if (i < num_entries) {
                        entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                    }
                    //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                    if (i < (max_num_children/2) + 1) {
                        offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) + i*sizeof(unsigned int)]);
                    }
                }
                __syncthreads();

                //NOW search
                if (i < num_entries) {
                    if (key > entries_actual[i] && key <= entries_actual[i + 1]){
                        found_idx = i;
                        if (key == entries_actual[i + 1]) {
                            *exact_match = true;
                        }
                    }
                } else if (i == num_entries) {
                    //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                    if (key > entries_actual[i]) {
                        found_idx = i;
                    }
                }
                __syncthreads();

                //We found either an exact match (so we can access next level) or at least an address to next btree level
                if (!*exact_match && !*is_last) {
                    //Calculate the address and the size of the next child
                    updated_idx += *first_child_offset*4;
                    if (found_idx == 0) {
                       size = offests_incremental[0]*4; //0 being found_idx but a bit faster cause we hardcode it
                    } else {
                        updated_idx += offests_incremental[found_idx - 1]*4;
                        size = (offests_incremental[found_idx] - offests_incremental[found_idx - 1])*4;
                    }
                    __syncthreads();
                } else if (!*exact_match && is_last) {
                    current_ngram++; //This is necessary so that longest match logic is kept correct since in the while loop we
                    goto backoff_notriecont; //Increment this before actually finding the next level
                } else {
                    // We have an exact match, so we just need to add it to the payload and be done with it
                    if (i == 0) {
                        //What we are doing here is reading the correct memory location for our payload. The payload is found
                        //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                        if (*is_last) {
                            accumulated_score += *(float *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(float))]); //Skip the previous keys' payload
                        } else {
                            accumulated_score += *(float *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                                + num_entries*sizeof(unsigned int) //Skip the keys
                                    + found_idx*(sizeof(float))]); //Skip the previous keys' payload
                        }
                    }
                }
            }
        }

    } //key != 0
    //Write the correct result at the end
    if (i == 0) {
        fn(accumulated_score); //This is basically either identity or exp, depending on what we need
        results[blockIdx.x] = accumulated_score;
    }
}

/*
    We have to do this to provide some degree of flexibility, whilst maintaining performance
    http://stackoverflow.com/questions/32534371/cuda-most-efficient-way-to-store-constants-that-need-to-be-parsed-as-arguments?noredirect=1#comment52933276_32534371
    http://stackoverflow.com/questions/6179295/if-statement-inside-a-cuda-kernel/6179580#6179580
    http://stackoverflow.com/questions/31569401/fastest-or-most-elegant-way-of-passing-constant-arguments-to-a-cuda-kernel?rq=1
    Instantiate templates for known things:
*/
inline void kernelTemplateWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_num_children,
  unsigned int max_ngram, cudaStream_t& stream, bool make_exp){
    if (max_ngram == 6) {
        if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 6><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 6><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else { 
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if (max_ngram == 5) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if (max_ngram == 4) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 4><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else if ( max_ngram == 3) {
        if (entries_per_node == 7) {
            if (make_exp) {
                gpuSearchBtree<8, 7, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<8, 7, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 31) {
            if (make_exp) {
                gpuSearchBtree<32, 31, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());
            } else {
                gpuSearchBtree<32, 31, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());
            }
        } else if (entries_per_node == 127) {
            if (make_exp) {
                gpuSearchBtree<128, 127, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, exponentify());    
            } else {
                gpuSearchBtree<128, 127, 3><<<num_ngram_queries, max_num_children, 0, stream>>>(btree_trie_mem, first_lvl, keys, results, identity());            
            }
        } else {
            printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    } else {
        printf("No template argument for node size %d and number of ngrams %d. If you want to use this configuration add it in %s:%d.\n",
             entries_per_node, max_ngram, __FILE__, __LINE__);
            exit(EXIT_FAILURE);
    }
}

inline void kernelTemplateWrapperDebug(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_num_children,
  unsigned int max_ngram, cudaStream_t& stream, cudaEvent_t &start, cudaEvent_t &stop, bool make_exp){
    cudaEventRecord(start);
    kernelTemplateWrapper(btree_trie_mem, first_lvl,  keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
}

void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, bool make_exp, bool debug) {

    unsigned int max_num_children = entries_per_node + 1;
    cudaStream_t stream;
    CHECK_CALL(cudaStreamCreate(&stream));

    if (debug) {
        //Time the kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        kernelTemplateWrapperDebug(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node,
         max_num_children, max_ngram, stream, start, stop, make_exp);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
        printf("Throughput: %d queries per second.\n", (int)((num_ngram_queries/(milliseconds))*1000));
    } else {
        kernelTemplateWrapper(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    }
    CHECK_CALL(cudaStreamDestroy(stream));
}

void searchWrapperStream(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram, cudaStream_t& stream, bool make_exp, bool debug) {


    unsigned int max_num_children = entries_per_node + 1;

    if (debug) {
        //Time the kernel execution @TODO remove once its working
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        kernelTemplateWrapperDebug(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node,
         max_num_children, max_ngram, stream, start, stop, make_exp);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
        printf("Throughput: %d queries per second.\n", (int)((num_ngram_queries/(milliseconds))*1000));
    } else {
        kernelTemplateWrapper(btree_trie_mem, first_lvl, keys, num_ngram_queries, results, entries_per_node, max_num_children, max_ngram, stream, make_exp);
    }
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID) {
    CHECK_CALL(cudaSetDevice(deviceID));
}

void GPUSearcher::search(unsigned int * keys, unsigned int num_ngram_queries, float * results, int streamID, bool debug) {
    if (streamID > num_streams - 1) {
        std::cerr << "Provided stream greater than the available ones. Using stream 0 as default. Fix your code!" << std::endl;
        streamID = 0;
    }

    searchWrapperStream(btree_trie_gpu, first_lvl_gpu, keys, num_ngram_queries, results, lm.metadata.btree_node_size,
     lm.metadata.max_ngram_order, streams[streamID], make_exp, debug);
}

std::vector<float> GPUSearcher::search(std::vector<unsigned int>& queries, int streamID, bool debug) {
    if (streamID > num_streams - 1) {
        std::cerr << "Provided stream greater than the available ones. Using stream 0 as default. Fix your code!" << std::endl;
        streamID = 0;
    }

    unsigned int num_ngram_queries = queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;
    allocateGPUMem(num_ngram_queries, &results);

    searchWrapperStream(btree_trie_gpu, first_lvl_gpu, gpuKeys, num_ngram_queries, results, lm.metadata.btree_node_size,
     lm.metadata.max_ngram_order, streams[streamID], make_exp, debug);

    std::vector<float> cpuResults(num_ngram_queries);

    copyToHostMemory(results, cpuResults.data(), num_ngram_queries);

    //Free memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    return cpuResults;
}

void GPUSearcher::gpuInit() {
    //Init GPU memory
    btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());

    if (num_streams < 1) {
        std::cerr << "You have specified " << num_streams << " number of streams however it must be at least 1. Using 1 stream as default. Fix your code!" << std::endl;
        num_streams = 1;
    }
    streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CHECK_CALL(cudaStreamCreate(&streams[i]));
    }
}

GPUSearcher::GPUSearcher(int num, LM& lm_, bool make_exp_) : lm(lm_), num_streams(num), make_exp(make_exp_) {
    gpuInit();
}

GPUSearcher::GPUSearcher(int num, LM& lm_, int gpuDeviceID, bool make_exp_) : lm(lm_), num_streams(num), make_exp(make_exp_) {
    setGPUDevice(gpuDeviceID);
    //Init GPU memory
    gpuInit();
}

GPUSearcher::~GPUSearcher() {
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);
    for (int i = 0; i < num_streams; i++) {
        CHECK_CALL(cudaStreamDestroy(streams[i]));
    }
}
