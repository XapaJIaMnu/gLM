#include "gpu_search.hh" 
#include <cuda_runtime.h>

extern __shared__ unsigned int shared_mem[];

//We want to copy a whole BTree node to shared memory. We will know the size in advance, we need to distribute the copying between
//our threads. We might end up copying more than we need, but that is fine, as long as we avoid warp divergence.
__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int * keys, float * results,
    unsigned int entry_size, unsigned int max_num_children, unsigned int entries_per_node, unsigned int max_ngram){

    unsigned int * offsets = shared_mem; //Reads in the first child offset + the shorts
    unsigned int * entries = &shared_mem[max_num_children/2 +1];
    unsigned int * prefix_sum = &entries[entries_per_node]; //Prefix sum gives us next node size
    unsigned int * found_idx = &prefix_sum[1];
    unsigned int * booleans = &found_idx[1]; //booleans[0] = is_last; booleans[1] = exact_match
    unsigned int * payload = &booleans[2];   //After we find the correct entry, load the payload here
    unsigned int * keys_shared = &payload[3]; //Each block fetches from shared memory the max necessary number of keys
    
    /*Since it's a bit hard to read the above shared memory alocations, so here's the static shared memory version
    __shared__ unsigned int offsets[MAX_NUM_CHILDREN/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[ENTRIES_PER_NODE];
    __shared__ unsigned int prefix_sum; //Prefix sum gives us next node size
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[MAX_NGRAM]; //Each block fetches from shared memory the max necessary number of keys
    */

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < max_ngram) {
       keys_shared[i] = keys[(blockIdx.x*max_ngram) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    __syncthreads();
    //if (i == 0) { //Maybe necessary to prevent OutOfBounds reading on the shared memory
    //    keys_shared[MAX_NGRAM] = 0; //Sometimes we are going out of bounds
    //}
    //__syncthreads();

    //Split some of the shared memory onto more comfrotable places
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];

    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];

    unsigned int * next_level = &payload[0];
    float * prob = (float *)&payload[1];
    float * backoff = (float *)&payload[2];

    int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.

    //Backoff variables
    unsigned int match_length_found = 0; //To check what was our longest match so we know what to backoff to
    float accumulated_score = 0;
    bool get_backoff = false; //Are we looking to extract backoff or prob from our ngram

    //Set the start index
    unsigned int current_btree_start = 0;
    unsigned int current_ngram = 0;
    unsigned int key = keys_shared[current_ngram];
    while (key != 0 && current_ngram < max_ngram) {
        current_ngram++;
        unsigned int updated_idx = current_btree_start + 4; //Update the index for the while loop
        unsigned int size = *(unsigned int *)&global_mem[current_btree_start];; //The size of the current node to process. @TODO This causes misaligned memory
        //Move to register to avoid sychronizationIs it better to do this in shared memory                                 access when using cuda-memcheck. Why?

        //Initialize shared variable
        if (i < 2) {
            booleans[i] = false;
        }
        __syncthreads();

        while (!*exact_match) {
            //First warp divergence here. We are reading in from global memory
            if (i == 0) {
                //@TODO: Replace this with a mod check
                int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(entry_size + sizeof(unsigned short));
                *is_last = !(entries_per_node == cur_node_entries);
                //@TODO. Fix this to be more efficient. Maybe move it with entries?
                //As per cuda memory model at least one write will succeed. We are clearing this value
                //So it doesn't interfere with the future values
                *prefix_sum = 0;
            }
            __syncthreads();


            if (*is_last) {
                //The number of entries in the bottom most nodes may be smaller than the size
                num_entries = size/entry_size;
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                    //printf("Entries i: %d, value %d\n", i, entries[i]);
                }
                //printf("Num entries: %d size: %d\n", num_entries, size);
            } else {
                num_entries = entries_per_node;
                //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                if (i < (max_num_children/2) + 1) {
                    offsets[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                }
                //Now load the entries
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) + i*sizeof(unsigned int)]);
                }
            }
            __syncthreads();

            //NOW search
            if (i == 0) {
                if (key <= entries[i]) {
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i < num_entries) {
                if (key > entries[i-1] && key <= entries[i]){
                    *found_idx = i;
                    if (key == entries[i]) {
                        *exact_match = true;
                    }
                }
            } else if (i == num_entries) {
                //Case where our key is greater than the last available entry. We need to do a prefix sum of i+1 elements.
                if (key > entries[i-1]) {
                    *found_idx = i;
                }
            }
            __syncthreads();
            //if (i == 0) {
            //    printf("Key %d, index %d, result_there %d num_entries %d\n", key, found_idx, entries[found_idx], num_entries);
            //}

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Do a prefix sum on the offsets here
                //@TODO optimize later. Do a proper prefix sum rather than atomic add
                if (i < *found_idx) {
                   atomicAdd(prefix_sum, (unsigned int)offests_incremental[i]);
                }
                __syncthreads(); //This is not necessary? It is necssary because the threads that don't take the if
                //path may write to the updated idx
                //As per the cuda memory model at least one write will succeed. since they all write the same we don't care
                size = (unsigned int)offests_incremental[*found_idx];
                updated_idx = *first_child_offset + *prefix_sum + current_btree_start; //*first_child_offset + prefix_sum only gives score since beginning
                                                                                    // of this btree. If we want index from the byte_arr start we need to add
                                                                                    // current_btree_start
                __syncthreads(); //Data hazard fix on size
                
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
                current_btree_start = 0;
                get_backoff = true;
                __syncthreads(); //Needed!
                break;

            } else {
                //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
                if (i < 3) {
                    //What we are doing here is reading the correct memory location for our payload. The payload is found
                    //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                    if (*is_last) {
                        payload[i] = *(unsigned int *)(&global_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                            + *found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                    } else {
                        payload[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + max_num_children*sizeof(unsigned short) //Skip the offsets and first offset
                            + num_entries*sizeof(unsigned int) //Skip the keys
                                + *found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
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
                    current_btree_start = *next_level; //@TODO maybe we need to sync here as well
                    if (current_btree_start == 0) {
                        //STOP. We are in the case of a trie that doesn't continue further. In this case we should basically
                        //do the backoff procuedure
                        goto backoff_notriecont;
                    }
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
        if (i == 0) {
            results[blockIdx.x] = accumulated_score;
        }
    }
}

//num_keys is the number of blocks we are going to launch. It is actually the number of ngrams queries
void searchWrapper(unsigned char * global_mem, unsigned int * keys, unsigned int num_ngram_queries, float * results,
    unsigned int entries_per_node, unsigned int entry_size, unsigned int max_ngram) {

    /*the shared_memory_size is equal to the combined size of all shared memory entries
    __shared__ unsigned int offsets[MAX_NUM_CHILDREN/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[ENTRIES_PER_NODE];
    __shared__ unsigned int prefix_sum; //Prefix sum gives us next node size
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[MAX_NGRAM]; //Each block fetches from shared memory the max necessary number of keys
    */
    unsigned int max_num_children = entries_per_node + 1;
    unsigned int shared_memory_size = (max_num_children/2 + 1 + entries_per_node + 1 + 1 + 2 + 3 + max_ngram)*sizeof(unsigned int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpuSearchBtree<<<num_ngram_queries, max_num_children, shared_memory_size>>>(global_mem, keys, results, entry_size,
        max_num_children, entries_per_node, max_ngram);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Searched for %d ngrams in: %f milliseconds.\n", num_ngram_queries, milliseconds);
}

void cudaDevSync() {
    cudaDeviceSynchronize();
}
