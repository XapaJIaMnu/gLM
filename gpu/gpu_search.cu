#include "gpu_search.hh" 
//#include "entry_structs.hh"
#include <cuda_runtime.h>

#define MAX_NUM_CHILDREN 128
#define ENTRIES_PER_NODE (MAX_NUM_CHILDREN - 1)
#define ENTRY_SIZE (sizeof(unsigned int) + sizeof(unsigned int) + 2*sizeof(float)) //Same as the getEntrySize(true)
#define MAX_NGRAM 5
//Assume working with 256 thread DON'T RELY ENTIRERLY ON THOSE! Size may be smaller. need a parameter.
//Requires two more threads then num of entries per node


//We want to copy a whole BTree node to shared memory. We will know the size in advance, we need to distribute the copying between
//our threads. We might end up copying more than we need, but that is fine, as long as we avoid warp divergence.
__global__ void gpuSearchBtree(unsigned char * global_mem, unsigned int * keys, float * results){

    __shared__ unsigned int offsets[MAX_NUM_CHILDREN/2 +1]; //Reads in the first child offset + the shorts
    __shared__ unsigned int entries[ENTRIES_PER_NODE];
    __shared__ unsigned int prefix_sum; //Prefix sum gives us next node size
    __shared__ unsigned int found_idx;
    __shared__ unsigned int booleans[2]; //booleans[0] = is_last; booleans[1] = exact_match
    __shared__ unsigned int payload[3]; //After we find the correct entry, load the payload here
    __shared__ unsigned int keys_shared[MAX_NGRAM]; //Each block fetches from shared memory the max necessary number of keys

    //Maybe we need to issue shared memory here to optimize it
    int i = threadIdx.x;
    if (i < MAX_NGRAM) {
       keys_shared[i] = keys[(blockIdx.x*MAX_NGRAM) + i]; //Shared memory read here for up NUM_NGRAM keys 
    }
    __syncthreads();

    //Split some of the shared memory onto more comfrotable places
    unsigned short * offests_incremental = (unsigned short *)&offsets[1];
    unsigned int * first_child_offset = &offsets[0];

    unsigned int * is_last = &booleans[0];
    unsigned int * exact_match = &booleans[1];

    unsigned int * next_level = &payload[0];
    float * prob = (float *)&payload[1];
    float * backoff = (float *)&payload[2];

    int num_entries; //Set the number of entries per node depending on whether we are internal or leaf.

    //Backoof variables
    unsigned int backing_off = 0; //To check whether we are backing off
    unsigned int num_backoff = 0; //To check how many items min we have to backoff to.
    float accumulated_score = 0;
    bool get_backoff = false; //Are we looking to extract backoff or prob from our ngram

    //Set the start index
    unsigned int current_btree_start = 0;
    unsigned int current_ngram = 0;
    unsigned int key = keys_shared[current_ngram];
    while (key != 0 && current_ngram < MAX_NGRAM) {
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
                int cur_node_entries = (size - sizeof(unsigned int) - sizeof(unsigned short))/(ENTRY_SIZE + sizeof(unsigned short));
                *is_last = !(ENTRIES_PER_NODE == cur_node_entries);
                //@TODO. Fix this to be more efficient. Maybe move it with entries?
                //As per cuda memory model at least one write will succeed. We are clearing this value
                //So it doesn't interfere with the future values
                prefix_sum = 0;
            }
            __syncthreads();


            if (*is_last) {
                //The number of entries in the bottom most nodes may be smaller than the size
                num_entries = size/ENTRY_SIZE;
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                    //printf("Entries i: %d, value %d\n", i, entries[i]);
                }
                //printf("Num entries: %d size: %d\n", num_entries, size);
            } else {
                num_entries = ENTRIES_PER_NODE;
                //Load the unsigned int start offset together with the accumulated offsets to avoid warp divergence
                if (i < (MAX_NUM_CHILDREN/2) + 1) {
                    offsets[i] = *(unsigned int *)(&global_mem[updated_idx + i*sizeof(unsigned int)]);
                }
                //Now load the entries
                if (i < num_entries) {
                    entries[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + MAX_NUM_CHILDREN*sizeof(unsigned short) + i*sizeof(unsigned int)]);
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
            //if (i == 0) {
            //    printf("Key %d, index %d, result_there %d num_entries %d\n", key, found_idx, entries[found_idx], num_entries);
            //}

            //We found either an exact match (so we can access next level) or at least an address to next btree level
            if (!*exact_match && !*is_last) {
                //Do a prefix sum on the offsets here
                //@TODO optimize later. Do a proper prefix sum rather than atomic add
                if (i < found_idx) {
                   atomicAdd(&prefix_sum, (unsigned int)offests_incremental[i]);
                }
                __syncthreads(); //This is not necessary? It is necssary because the threads that don't take the if
                //path may write to the updated idx
                //As per the cuda memory model at least one write will succeed. since they all write the same we don't care
                size = (unsigned int)offests_incremental[found_idx];
                updated_idx = *first_child_offset + prefix_sum + current_btree_start; //*first_child_offset + prefix_sum only gives score since beginning
                                                                                    // of this btree. If we want index from the byte_arr start we need to add
                                                                                    // current_btree_start
                __syncthreads(); //Data hazard fix on size
                
            } else if (*is_last && !*exact_match) {
                //In this case we didn't find the key that we were looking for
                //What we should do is get the probability of the last node that we found
                //The last node that we found's probability should be in shared memory

                if (get_backoff) {
                    break; //If we didn't find a get_backoff value
                } else {
                    accumulated_score += *prob; //Multiply by the probability of n-1 gram. WE are working in log space so multiply -> add 
                }
                backing_off++; //Start from further ahead in the backoff
                num_backoff = current_ngram - 1; //How many terms are in our min backoff
                                                //Current_ngram - 1 shows how many words we matched from the ngram and we need to backoff to exactly as many token
                                                //as the minimum level backoff
                current_ngram = backing_off; //Start trie lookup further ahead
                key = keys_shared[current_ngram];
                get_backoff = true;
                current_btree_start = 0; //Go back to the root of the tries
                __syncthreads();
                //printf("We are here!\n");
                break;

            } else {
                //Locate the rest of the data for the entry (i.e. the payload - backoff, prob, next offset)
                if (i < 3) {
                    //What we are doing here is reading the correct memory location for our payload. The payload is found
                    //After the offsets and the keys, so we skip them and then we skip to the correct payload using found_idx
                    if (*is_last) {
                        payload[i] = *(unsigned int *)(&global_mem[updated_idx + num_entries*sizeof(unsigned int) //Skip the keys
                            + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                + i*sizeof(unsigned int)]); //Get next_level/prob/backoff
                    } else {
                        payload[i] = *(unsigned int *)(&global_mem[updated_idx + sizeof(unsigned int) + MAX_NUM_CHILDREN*sizeof(unsigned short) //Skip the offsets and first offset
                            + num_entries*sizeof(unsigned int) //Skip the keys
                                + found_idx*(sizeof(unsigned int) + sizeof(float) + sizeof(float)) //Skip the previous keys' payload
                                    + i*sizeof(unsigned int)]);  //Get next_level/prob/backoff
                    }
                }

                key = keys_shared[current_ngram]; //@TODO Out of bounds memory access here probably

                if (current_ngram < MAX_NGRAM && key != 0) {
                    __syncthreads();
                    current_btree_start = *next_level; //@TODO maybe we need to sync here as well
                    if (get_backoff && (num_backoff < current_ngram)) { //Get all necessary backoffs on the way
                        accumulated_score += *backoff;
                    }
                } else {
                    if (get_backoff) {
                        accumulated_score += *backoff;
                    } else {
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

void searchWrapper(unsigned char * global_mem, unsigned int * keys, unsigned int num_keys, float * results) {
    //Block size should always be MAX_NUM_CHILDREN for best efficiency when searching the btree
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpuSearchBtree<<<num_keys, MAX_NUM_CHILDREN>>>(global_mem, keys, results);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Searched %d keys in: %f milliseconds.\n", num_keys, milliseconds);
}

/* Can't compile easily with cmake. Maybe there's a better way
__global__ void searchInBulk(unsigned int * keys_array, unsigned char * btree_trie) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    gpuSearchBtree<<<1, MAX_NUM_CHILDREN>>>(btree_trie, 0, keys_array[i]);
}
*/

void cudaDevSync() {
    cudaDeviceSynchronize();
}
