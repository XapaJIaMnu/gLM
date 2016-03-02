#include <vector>
#include <iostream>
#include <math.h>
#include <memory>
#include <assert.h>
#include <deque>
#include <climits>
#include "btree_v2.hh"
#include <sstream>
#include <set>
#include <algorithm>

//Note that the ARRAY vector will be cleared during execution so that memory usage doesn't get too out of hand.
void array2balancedBtree(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &array, unsigned short BtreeNodeSize, bool lastNgram) {
    /*Idea: First we have the current BTree constructed as an array.
      We convert that array to a BTree using the following algorithm:
      1) Divide the length of the array by the BTreeNodeSize and divide the array in n even parts (approximately)
      2) Put in the root node one element every n locations from the array.
      3) Reccursively do the same for children node.
      The node configuration should be the following:
      |vocabID1,VocabID2...||OffsetToChild1,OffsetToChild2...OffsetToChildN,SizeOfChildN||PayloadForChild1,PayloadForCHild2...|
      Sizes and offsets should be of type unsigned short, except OffseToChild1, which is unsigned int.
      To get offsetToChild1, one needs to add OffsetToChild1 to the current start offset and get M as the starting point for the first child.
      In order to get the size of the first child one needs to look at offsetToChild2.
      To get offsetToChild2, one needs to add OffsetToChild1 to the current start offset and add to that OffsetToChild2. To get the size one needs
      to subtract OffsetToChild2 from OffsetToChild3 and so on. Since the optimal number of childre is 32 and we won't be having too many more
      than that it is feasible to fit the offsets in unsigned shorts. Since all sizes are multiple of fours (only odd BTreeNodeSizes allowed).
      and first BTree node has extra 4 bytes at the beginning (unsigned int) in order to tell the size of the root node. The first child offset,
      as well as the offsets to consecutive trie levels need to be multiplied by four to get the correct locations.

      Size of BTree node (in bytes) 4*BTreeNodeSize + 4 + 4*BtreeNodeSize/2 + 4*BtreeNodeSize*12
      If it is last level of Trie the size is: 4*BTreeNodeSize + 4 + 4*BtreeNodeSize/2 + 4*BtreeNodeSize*4, because there's no next_level offset
      or backoff.
    */

    //Determine the size of the node.
    int payload_size;
    if (lastNgram) {
        payload_size = 4;
    } else {
        payload_size = 12;
    }
    unsigned int root_size;

    if (array.size() <= BtreeNodeSize) {
        root_size = (4 + payload_size)*array.size();
    } else {
        root_size = (4 + payload_size)*BtreeNodeSize + 4 + (4*(BtreeNodeSize + 1))/2;
    }

    //Put the size of the root at the beginning of the array.
    byte_arr.resize(byte_arr.size() + 4);
    std::memcpy(&byte_arr[byte_arr.size() - 4], &root_size, sizeof(root_size));

    //Set up for a while loop
    std::deque<std::vector<Entry_v2> > future_nodes;
    future_nodes.push_back(array);
    array.clear(); //We are not goint to use the original input array anymore, so clear it in order to use less memory

    while (!future_nodes.empty()) {
        std::vector<Entry_v2> cur_array = future_nodes.front();
        //Initialize offsets vector
        std::vector<unsigned int> offsets;
        offsets.reserve(BtreeNodeSize + 2);

        if (cur_array.size() <= BtreeNodeSize) {
            if (cur_array.size() != 0) { //Only insert non empty entries
                entry_v2_to_node(byte_arr, cur_array, offsets, payload_size);
            }
            future_nodes.pop_front();
        } else {

            //Calculate first child offset here
            offsets.push_back(0); //Initial value for future offsets
            for (auto future_node : future_nodes) {
                offsets[0] += futureSizeCalculator(future_node.size(), BtreeNodeSize, payload_size);
            }
            assert(offsets[0] % 4 == 0); //Verify that we indeed have an address divisible by 4.
            offsets[0] = offsets[0]/4;

            future_nodes.pop_front(); //Remove the node that we are currently processing from the queue

            //Choose elements to put into the node.
            std::vector<unsigned int> splits = createEvenSplits(cur_array.size(), BtreeNodeSize);

            std::vector<Entry_v2> entries_to_insert;
            entries_to_insert.reserve(BtreeNodeSize);
            
            unsigned int accumulated_entry_number = 0; //Keeps track of which entries from the array we need to access
            for (auto split : splits) {
                std::vector<Entry_v2> children; //Initialize the children vector
                children.resize(split);

                //Copy the entries which will belong and descend from that child
                std::memcpy(&children[0], &cur_array[0 + accumulated_entry_number], split*sizeof(children[0]));

                //Calculate the size of the top node and use it to calculate the offset to consecutive child
                offsets.push_back(futureSizeCalculator(children.size(), BtreeNodeSize, payload_size));

                //Push it onto the queue for processing in the future
                future_nodes.push_back(children);
                accumulated_entry_number += split; //Account for the progress in the array.

                //This is necessary to prevent adding to entries_to_insert too many times
                if (accumulated_entry_number < cur_array.size()) {
                    entries_to_insert.push_back(cur_array[accumulated_entry_number]);
                    accumulated_entry_number++;
                }
            }

            assert(entries_to_insert.size() == BtreeNodeSize); //Something's wrong with the algorithm otherwise.

            //Do prefix sums for future offsets and add the last
            for (unsigned int i = 2; i < offsets.size(); i++){
                offsets[i] += offsets[i-1]; //This will effectively compute prefix sum starting from first element.
            }

            //for (auto entry : entries_to_insert) {
            //    if (entry.vocabID == 67715) {
            //        std::cout << "Here" << std::endl;
            //    }
            //}

            entry_v2_to_node(byte_arr, entries_to_insert, offsets, payload_size);
        }
    } 

}

/*Given a subsection of the array representation of the btree, calculate the size of the top node*/
unsigned int futureSizeCalculator(unsigned int size, unsigned short BtreeNodeSize, int payload_size) {
    unsigned int node_size;
    if (size <= BtreeNodeSize) {
        node_size = (4 + payload_size)*size;
    } else {
        node_size = (4 + payload_size)*BtreeNodeSize + 4 + (4*(BtreeNodeSize + 1))/2;
    }

    return node_size;
}

void entry_v2_to_node(std::vector<unsigned char> &byte_arr, std::vector<Entry_v2> &entries, std::vector<unsigned int> offsets, unsigned int payload_size) {
    /* Case 1: inner: |vocabIDs|OFFSETS|PAYLOADS|
       Case 2: leaf   |vocabIDs|Payloads|
       payload size: 4 for last level ngrams, 12 for every other case
    */

    unsigned int node_size;
    if (offsets.size() > 0) {
        node_size = entries.size()*(4 + payload_size) + 4 + (offsets.size() - 1)*2;
    } else {
        node_size = entries.size()*(4 + payload_size);
    }
    
    size_t cur_byte_arr_idx = byte_arr.size();
    byte_arr.resize(byte_arr.size() + node_size); //New size for byte array.

    std::unique_ptr<unsigned char[]> payloads(new unsigned char[payload_size*entries.size()]); //Payload size is already in bytes

    for (unsigned int i = 0; i < entries.size(); i++) {
        std::memcpy(&byte_arr[cur_byte_arr_idx], &entries[i].vocabID, sizeof(entries[i].vocabID));
        cur_byte_arr_idx += sizeof(entries[i].vocabID);

        //Put the payloads into tmp array.
        if (payload_size == 4) {
            std::memcpy(&payloads[0] + 4*i, &entries[i].prob, sizeof(entries[i].prob));
        } else {
            memset(&payloads[0] + 12*i, 0, 4); //Empty next_level_offset
            std::memcpy(&payloads[0] + 12*i + 4, &entries[i].prob, sizeof(entries[i].prob));
            std::memcpy(&payloads[0] + 12*i + 8, &entries[i].backoff, sizeof(entries[i].backoff));
        }
    }

    if (offsets.size() > 0) {
        //Copy the first child offset which is actually unsigned int
        std::memcpy(&byte_arr[cur_byte_arr_idx], &offsets[0], sizeof(offsets[0]));
        cur_byte_arr_idx += sizeof(offsets[0]);

        //The rest of the elements are just unsigned shorts so cast them and copy them.
        for (unsigned int i = 1; i < offsets.size(); i++) {
            if (offsets[i]/4 > USHRT_MAX) {
                std::cerr << "The btree node size chosen will cause an overflow in the btree. Sorry, but you will have to use a smaller value ; (" << std::endl;
                exit(EXIT_FAILURE);
            }
            if (offsets[i] == 186680 || offsets[i] == 186680*4) {
                std::cout << "here" << std::endl;
            }
            assert(offsets[i] % 4 == 0); //Sanity check
            unsigned short tmpnum = (unsigned short)(offsets[i]/4);
            std::memcpy(&byte_arr[cur_byte_arr_idx], &tmpnum, sizeof(tmpnum));
            cur_byte_arr_idx += sizeof(tmpnum);
        }
    }

    std::memcpy(&byte_arr[cur_byte_arr_idx], &payloads[0], payload_size*entries.size()); //Copy the payloads

}

//Idea: if we have BtreeNodeSize, we want BtreeNodeSize+1 almost equal children.
//To do that we get remainder elements and we split them into even groups.
std::vector<unsigned int> createEvenSplits(unsigned int array_size, unsigned short BtreeNodeSize) {
    std::vector<unsigned int> equal_parts;
    equal_parts.reserve(BtreeNodeSize + 1);

    if (array_size > 2*BtreeNodeSize) {
        unsigned int remaining_elements = (array_size - BtreeNodeSize); //The elements minus the ones that will go to the node
        unsigned int equal_parts_size = remaining_elements/(BtreeNodeSize + 1);
        unsigned int equal_remainders = remaining_elements % (BtreeNodeSize + 1);

        for (unsigned short i = 0; i < BtreeNodeSize + 1; i++) {
            if (equal_remainders > 0) {
                equal_parts.push_back(equal_parts_size + 1); //Take out from the remainder
                equal_remainders--;
            } else {
                equal_parts.push_back(equal_parts_size);
            }
        }
    } else {
        //In order for this algorithm to work we need array_size > 2*BtreeNodeSize
        //If however we have BtreeNodeSize < array_size <= 2*BtreeNodeSize we will just take
        //the first n elements for the current node, and stick the rest at the end to one leaf node
        //To achieve this with our algorithm the return vector will contain 0 for the first n elements
        //and array_size - BtreeNodeSize for the final one.
        equal_parts.resize(BtreeNodeSize);
        memset(&equal_parts[0], 0, BtreeNodeSize*sizeof(equal_parts[0]));
        equal_parts.push_back(array_size - BtreeNodeSize);
    }
    

    return equal_parts;

}

Entry_with_offset searchBtree(std::vector<unsigned char> &byte_arr, size_t BtreeStartPosition, unsigned short BtreeNodeSize, unsigned int vocabID, bool lastNgram) {
    unsigned short payload_size;
    if (lastNgram) {
        payload_size = 4;
    } else {
        payload_size = 12;
    }

    Entry_with_offset result;

    //Get the size of the first node:
    unsigned int node_size;
    size_t current_start_pos = BtreeStartPosition + 4; //Acount for the uint in the beginning of the BTree
    std::memcpy(&node_size, &byte_arr[BtreeStartPosition], sizeof(node_size));

    while (true) {
        result = searchNode(byte_arr, current_start_pos, node_size, vocabID, payload_size, BtreeNodeSize);
        current_start_pos = result.next_child_offset;
        node_size = result.next_child_size;
        if (result.found) {
            result.currentBtreeStart = BtreeStartPosition;
            return result;
        } else if ((current_start_pos == 0) && (node_size == 0)){
            //We have not found the vocabID and we have reached a leaf.
            result.currentBtreeStart = BtreeStartPosition;
            return result;
        }
    }
}

Entry_with_offset searchNode(std::vector<unsigned char> &byte_arr, size_t StartPosition, unsigned int node_size, unsigned int vocabID,
 unsigned short payload_size, unsigned short BtreeNodeSize) {
    Entry_with_offset result = {0, 0, 0.0, 0.0, 0, 0, false, 0, 0};

    unsigned int entry_size = 4 + payload_size;
    //Determine whether we have a leaf or internal node. If the node is internal it's going to be fully saturated. Otherwise
    //its a leaf node. We can check if we have an internal node if we have the expected number of entries.
    unsigned int cur_node_entries = (node_size - sizeof(unsigned int) - sizeof(unsigned short))/(entry_size + sizeof(unsigned short));
    bool is_leaf = !(BtreeNodeSize == cur_node_entries);

    //Some helper variables:
    unsigned int * payloads;
    std::pair<unsigned int, bool> found;

    if (is_leaf) {
        assert(node_size % entry_size == 0); //Sanity check
        cur_node_entries = node_size/entry_size;
        //Perform linear seach on the vocabIDs of the entry
        unsigned int * vocabIDs = reinterpret_cast<unsigned int *>(&byte_arr[StartPosition]);
        found = linearSearch(vocabIDs, cur_node_entries, vocabID);
        result.found = found.second;
        result.found_idx = found.first;

        if (found.second) {
            result.vocabID = vocabIDs[found.first]; //set the vocabID
            //Set the payload beginning locaiton.
            payloads = reinterpret_cast<unsigned int *>(&byte_arr[StartPosition + cur_node_entries*sizeof(vocabID)]);
        }
    } else {
        //We have a saturated internal node, perform search on it.
        cur_node_entries = BtreeNodeSize;
        unsigned int * vocabIDs = reinterpret_cast<unsigned int *>(&byte_arr[StartPosition]);
        found = linearSearch(vocabIDs, cur_node_entries, vocabID);
        result.found = found.second;
        result.found_idx = found.first;

        if (found.second) {
            result.vocabID = vocabIDs[found.first]; //set the vocabID
            //Set the payload beginning locaiton.
            unsigned int payload_extra_offset = 
                cur_node_entries*sizeof(vocabID) + cur_node_entries*sizeof(unsigned short) + sizeof(unsigned short) + sizeof(unsigned int);
            payloads = reinterpret_cast<unsigned int *>(&byte_arr[StartPosition + payload_extra_offset]);
        } else {
            //We have not found the vocabID but we have found the continuation where it could be
            unsigned int * first_child_offset = reinterpret_cast<unsigned int *>(&byte_arr[StartPosition + cur_node_entries*sizeof(vocabID)]);
            unsigned short * next_children_offsets = reinterpret_cast<unsigned short *>(&byte_arr[StartPosition + cur_node_entries*sizeof(vocabID) + sizeof(unsigned int)]);
            size_t first_child_full_offset = StartPosition + *first_child_offset*4;

            //Get the size of the resulting node and additional offset for every consecutive child.
            unsigned int additional_offset;
            unsigned int next_node_size;
            if (found.first == 0) {
                additional_offset = 0;
                next_node_size = next_children_offsets[found.first]*4;
            } else {
                additional_offset = next_children_offsets[found.first - 1]*4;
                next_node_size = next_children_offsets[found.first]*4 - next_children_offsets[found.first - 1]*4;
            }
            first_child_full_offset += additional_offset;
            result.next_child_offset = first_child_full_offset;
            result.next_child_size = next_node_size;
            //Sanity check
            if (next_node_size == 186680) {
                std::cout << "here" << std::endl;
            }
            assert(first_child_full_offset < byte_arr.size());
        }
    }
    //Extract payloads if we have found the entry.
    if (found.second) {
        float * prob_tmp;
        float * backoff_tmp;
        if (payload_size == 4) {
            prob_tmp = reinterpret_cast<float *>(&payloads[found.first]);
            result.prob = *prob_tmp;
        } else {
            result.next_level = &payloads[found.first*3];

            prob_tmp = reinterpret_cast<float *>(&payloads[found.first*3+1]);
            result.prob = *prob_tmp;

            backoff_tmp = reinterpret_cast<float *>(&payloads[found.first*3+2]);
            result.backoff = *backoff_tmp;
        }
    }

    return result;
}

//Finds either the matching entry or the continuation position
std::pair<unsigned int, bool> linearSearch(unsigned int * arr_to_search, unsigned int size, unsigned int vocabID) {
    std::pair<unsigned int, bool> ret;
    bool set = false; //Checks if we set anything
    for (unsigned int i = 0; i < size; i++) {
        if (vocabID == arr_to_search[i]) {
            set = true;
            ret.first = i;
            ret.second = true;
            break;
        } else if (vocabID < arr_to_search[i]) {
            ret.first = i;
            ret.second = false;
            set = true;
            break;
        }
    }

    if (!set) {
        //We didn't trigger any condition which means the vocabID we are looking forward is on the right of the last child
        ret.first = size;
        ret.second = false;
    }

    return ret;
}

std::pair<bool, std::string> test_btree_v2(unsigned int num_elements, unsigned short BtreeNodeSize, bool lastNgram) {
    std::stringstream error;
    bool passes = true;

    //Create the Btree
    std::set<unsigned int> prev_nums; //Used to see if we have duplicating nums
    std::vector<Entry_v2> array;
    while (prev_nums.size() < num_elements) {
        unsigned int new_entry = 1 + (rand() % (num_elements*10));
        if (prev_nums.count(new_entry) == 0){
            Entry_v2 new_entry_actual = {new_entry, prev_nums.size() + 0.0f, prev_nums.size() + 0.5f};
            array.push_back(new_entry_actual);
            prev_nums.insert(new_entry);
        }
    }

    std::sort(array.begin(), array.end()); 

    std::vector<unsigned char> btree_byte_arr;
    array2balancedBtree(btree_byte_arr, array, BtreeNodeSize, lastNgram);

    for (auto entry : array) {
        Entry_with_offset test = searchBtree(btree_byte_arr, 0, BtreeNodeSize, entry.vocabID, lastNgram);
        if (lastNgram) {
            if (entry.vocabID != test.vocabID || entry.prob != test.prob) {
                error << "Expected vocabID:" << entry.vocabID << " prob: " << entry.prob << ", got: " << test.vocabID << " " << test.prob << std::endl;
                passes = false;
                break;
            }
        } else {
            //assign a dummy next level offset just to test that it works:
            *test.next_level = entry.vocabID;
            if (entry.vocabID != test.vocabID || entry.prob != test.prob || entry.backoff != test.backoff) {
                error << "Expected vocabID:" << entry.vocabID << " prob: " << entry.prob << " backoff: " << entry.backoff <<
                ", got: " << test.vocabID << " " << test.prob << " " << test.backoff << std::endl;
                passes = false;
                break;
            }
        }  
    }

    //Test if next_level was saved successfully
    if (!lastNgram && passes) {
        for (auto entry : array) {
            Entry_with_offset test = searchBtree(btree_byte_arr, 0, BtreeNodeSize, entry.vocabID, lastNgram);
            if (entry.vocabID != *test.next_level) {
                error << "Expected next_level to be set to: " << entry.vocabID << ", got: " << *test.next_level << std::endl;
                passes = false;
                break;
            }
        }
    }

    return std::pair<bool, std::string>(passes, error.str());
}
