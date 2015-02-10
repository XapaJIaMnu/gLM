#include <vector>
#include <utility>

class B_tree {
    public:
        void * root_node; //it should be B_tree_node. Avoid cyclical definition
        B_tree(unsigned short);
        ~B_tree();
};

class B_tree_node {
    
    public:
        unsigned short max_elem;
        std::vector<int> values;
        std::vector<B_tree_node *> children;
        B_tree_node * parent;
        B_tree * container; //Accessible only from the root node
        bool is_root;

        //Operations
        B_tree_node(unsigned short, B_tree_node *);
        B_tree_node(unsigned short, B_tree *);
        ~B_tree_node();
        //Insertion to vector location is index to be inserted (before the old one)
        void insert(B_tree_node * new_node, int location);
        void insert(int new_val, int location);
        std::pair<B_tree_node *, int> find_position(int new_value);
        void split();
};

B_tree::B_tree(unsigned short num_max_elem) {
    B_tree_node * root = new B_tree_node(num_max_elem, this); //This should be safe, I am just need "this" for the address.
    root_node = reinterpret_cast<void *>(root);
}

B_tree::~B_tree() {
    B_tree_node * actual_node = reinterpret_cast<B_tree_node *>(root_node);
    delete actual_node;
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree_node * parent_node) {
    max_elem = num_max_elem;
    is_root = false;
    parent = parent_node;
    container = nullptr;
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree * container_orig) {
    max_elem = num_max_elem;
    is_root = true;
    parent = nullptr;
    container = container_orig;
}

B_tree_node::~B_tree_node(){
    values.clear();
    for (int i = 0; i < children.size(); i++) {
        delete children[i];
    }
}

void B_tree_node::insert(B_tree_node * new_node, int location){
    if (children.size() == 0) {
        children.push_back(new_node);
    } else {
        std::vector<B_tree_node *> new_vec(children.size() + 1);
        for (int i = 0; i<children.size(); i++){
            if (i < location) {
                new_vec[i] = children[i];
            } else if (i == location) {
                new_vec[i] = new_node;
                new_vec[i+1] = children[i];
            } else {
                new_vec[i+1] = children[i];
            }
        }
        children = new_vec;
    }

}

void B_tree_node::insert(int new_val, int location){
    if (values.size() == 0) {
        values.push_back(new_val);
    } else {
        std::vector<int> new_vec(values.size() + 1);
        for (int i = 0; i<values.size(); i++){
            if (i < location) {
                new_vec[i] = values[i];
            } else if (i == location) {
                new_vec[i] = new_val;
                new_vec[i+1] = values[i];
            } else {
                new_vec[i+1] = values[i];
            }
        }
        values = new_vec;
    }

}

std::pair<B_tree_node *, int> B_tree_node::find_position(int new_value) {

    int candidate_position = values.size(); //Assume last position

    for (int i = 0; i < values.size(); i++){
        if (values[i] > new_value) {
            //We can never have two nodes with the same value as per specification.
            candidate_position = i;
        }
    }

    if (children.size() == 0){
        return std::pair<B_tree_node *, int>(this, candidate_position);
    } else {
        return children[candidate_position]->find_position(new_value);
    }

}

void B_tree_node::split() {
    if (values.size() < max_elem){
        //No need to split the node. It's balanced.
        return;
    }

    int middle_idx = values.size()/2; //Integer division here

    //We if we need to split we take our current node to become the left node
    //by trimming it and we will create a new node which will be the right node
    //from the elements that we previously cut out.

    //Save the middle value;
    int middle_value = values[middle_idx];

    //Create the right node
    B_tree_node * right_node = new B_tree_node(max_elem, parent);

    //Populate it
    std::copy(values.begin() + middle_idx + 1, values.end(), right_node->values.begin());
    std::copy(children.begin() + middle_idx, children.end(), right_node->children.begin());

    //Trim the left node.
    this->values.resize(middle_idx);
    this->children.resize(middle_idx+1);

    //Assign parent node and change root if necessary
    if (parent == nullptr) {
        //We are the root node, we need to make a new root.
        B_tree_node * new_root = new B_tree_node(max_elem, container);
        new_root->insert(middle_value, 0);

        this->is_root = false;
        this->container = nullptr;
        this->parent = new_root;
        right_node->parent = new_root;

        //Now assign child pointers to the new root.
        new_root->insert(right_node, 0);
        new_root->insert(this, 0);
        //We are done, the tree is balanced and split;
    } else {
        //Find the location of the middle_value in the parent
        int new_location = parent->values.size();
        for (int i = 0; i< parent->values.size(); i++) {
            if (parent->values.size() > middle_value) {
                new_location = middle_value;
            }
        }
        //insert the middle_value and the right node (the left was there beforehand)
        parent->insert(middle_value, new_location);
        parent->insert(right_node, new_location+1);

        //Check if parent is balanced
        parent->split();
    }


}