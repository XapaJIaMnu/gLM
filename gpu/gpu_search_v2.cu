#include "gpu_search.hh" 
#include <cuda_runtime.h>
#include "gpu_common.h"

#define big_entry 16
#define small_entry 8

template<unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram>
__global__ void gpuSearchBtree(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys, float * results) {

    __shared__ unsigned int offsets[max_num_children/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[entries_per_node];
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[max_ngram]; //Each block fetches from shared memory the max necessary number of keys

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < max_ngram) {
       keys_shared[i] = keys[(blockIdx.x*max_ngram) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    __syncthreads();

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
    if (i < 3) {
        payload[i] = first_lvl[(keys_shared[0]-1)*3 + i];
    }
    __syncthreads();

    //Set the start index
    unsigned int current_btree_start = *next_level*4;
    unsigned int current_ngram = 1;
    unsigned int key = keys_shared[current_ngram];

    //Some necessary variables
    unsigned int updated_idx;
    unsigned int size;

    while (key != 0 && current_ngram < max_ngram - 1) {
        current_ngram++;
        updated_idx = current_btree_start + 4; //Update the index for the while loop
        //@TODO consider this for shared memory as oppposed to global mem broadcast to register
        size = *(unsigned int *)&btree_trie_mem[current_btree_start];; //The size of the current node to process.

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
                //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                if (i < (max_num_children/2) + 1) {
                    offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                }
                //Now load the entries
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) + i*sizeof(unsigned int)]);
                }
            }
            __syncthreads();

            //NOW search
            if (i == 0) {
                if (key <= entries[i]) {
                    found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i < num_entries) {
                if (key > entries[i-1] && key <= entries[i]){
                    found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i == num_entries) {
                //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                if (key > entries[i-1]) {
                    found_idx = i;
                }
            }
            __syncthreads();

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Calculate the address and the size of the next child
                updated_idx += *first_child_offset*4;
                if (found_idx == 0) {
                   size = offests_incremental[0]; //0 being found_idx but a bit faster cause we hardcode it
                } else {
                    updated_idx += offests_incremental[found_idx - 1];
                    size = offests_incremental[found_idx] - offests_incremental[found_idx - 1];
                }
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
                    accumulated_score += *prob; //We add the longest match of probability we found.
                }
                match_length_found = current_ngram - 1; //The length of the match found. We need to backoff from toplevel to here
                current_ngram = 1; //Set backoff in -1. If we run into this case again we need to do nothing
                key = keys_shared[current_ngram];
                current_btree_start = 4;
                get_backoff = true;
                __syncthreads(); //Needed!
                break;

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

                key = keys_shared[current_ngram]; //@TODO Out of bounds memory access here probably
                //__syncthreads(); Not a proper fix! Replace that with proper check for invalid next level of btree
                //if (*next_level == 0 && key != 0 && !get_backoff) { //key == 0 attempts to prevent <s> from falling here
                //    goto unktoken; //If we have invalid "next_level" it's going to be indexed 0. True for unk
                //}
                __syncthreads();
                if (current_ngram < max_ngram && key != 0) {
                    //@TODO this might not be necessary in this case: only in the last_btree_level so the logic can be simplified
                    if (*next_level == 0) {
                        //STOP. We are in the case of a trie that doesn't continue further. In this case we should basically
                        //do the backoff procuedure
                        goto backoff_notriecont;
                    }
                    current_btree_start = current_btree_start + *next_level*4;
                    if (get_backoff && (match_length_found < current_ngram)) { //Get all necessary backoffs on the way
                        accumulated_score += *backoff;
                    }
                } else {
                    if (get_backoff && (match_length_found < current_ngram)) {
                        accumulated_score += *backoff;
                    } else { //Actual match here
                        accumulated_score += *prob;
                    }
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
        size = *(unsigned int *)&btree_trie_mem[current_btree_start];; //The size of the current node to process.

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
                int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(big_entry + sizeof(unsigned short));
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
                //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                if (i < (max_num_children/2) + 1) {
                    offsets[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + i*sizeof(unsigned int)]);
                }
                //Now load the entries
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) + i*sizeof(unsigned int)]);
                }
            }
            __syncthreads();

            //NOW search
            if (i == 0) {
                if (key <= entries[i]) {
                    found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i < num_entries) {
                if (key > entries[i-1] && key <= entries[i]){
                    found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i == num_entries) {
                //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                if (key > entries[i-1]) {
                    found_idx = i;
                }
            }
            __syncthreads();

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Calculate the address and the size of the next child
                updated_idx += *first_child_offset*4;
                if (found_idx == 0) {
                   size = offests_incremental[0]; //0 being found_idx but a bit faster cause we hardcode it
                } else {
                    updated_idx += offests_incremental[found_idx - 1];
                    size = offests_incremental[found_idx] - offests_incremental[found_idx - 1];
                }
            } else if (!*exact_match && is_last) {
                goto backoff_notriecont;
            } else {
                // We have an exact match, so we just need to add it to the payload and be done with it
                if (i == 0) {
                    //What we are doing here is reading the correct memory location for our payload. The payload is found
                    //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                    if (*is_last) {
                        accumulated_score += *(unsigned int *)(&btree_trie_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                            + found_idx*(sizeof(float))]); //Skip the previous keys' payload
                    } else {
                        accumulated_score += *(unsigned int *)(&btree_trie_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                            + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(float))]); //Skip the previous keys' payload
                    }
                }
            }
        }
    }

    //@TODO this might need to be inside the while loop, but I don't think so
    //Write the correct result at the end
    if (i == 0) {
        results[blockIdx.x] = accumulated_score;
    }
}

void searchWrapper(unsigned char * btree_trie_mem, unsigned int * first_lvl, unsigned int * keys,
 unsigned int num_ngram_queries, float * results, unsigned int entries_per_node, unsigned int max_ngram) {

    unsigned int max_num_children = entries_per_node + 1;

    //Time the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /*
    We have to do this to provide some degree of flexibility, whilst maintaining performance
    http://stackoverflow.com/questions/32534371/cuda-most-efficient-way-to-store-constants-that-need-to-be-parsed-as-arguments?noredirect=1#comment52933276_32534371
    http://stackoverflow.com/questions/6179295/if-statement-inside-a-cuda-kernel/6179580#6179580
    http://stackoverflow.com/questions/31569401/fastest-or-most-elegant-way-of-passing-constant-arguments-to-a-cuda-kernel?rq=1
    Instantiate templates for known things:
    */
    if (entries_per_node == 3) {
        cudaEventRecord(start);
        gpuSearchBtree<4, 3, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 7) {
        cudaEventRecord(start);
        gpuSearchBtree<8, 7, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 15) {
        cudaEventRecord(start);
        gpuSearchBtree<16, 15, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 23) {
        cudaEventRecord(start);
        gpuSearchBtree<24, 23, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 31) {
        cudaEventRecord(start);
        gpuSearchBtree<32, 31, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 63) {
        cudaEventRecord(start);
        gpuSearchBtree<64, 63, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 127) {
        cudaEventRecord(start);
        gpuSearchBtree<128, 127, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 255) {
        cudaEventRecord(start);
        gpuSearchBtree<256, 255, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    } else if (entries_per_node == 511) {
        cudaEventRecord(start);
        gpuSearchBtree<512, 511, 5><<<num_ngram_queries, max_num_children>>>(btree_trie_mem, first_lvl, keys, results);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}

/*Tells the code to execute on a particular device. Useful on multiGPU systems*/
void setGPUDevice(int deviceID) {
    CHECK_CALL(cudaSetDevice(deviceID));
} 
