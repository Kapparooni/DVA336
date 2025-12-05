#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct bstnode {
    bstnode(int key) : key(key) { left = right = nullptr; }
    ~bstnode() { delete left; delete right; }
    int key;
    bstnode *left, *right;
};

inline int max(int a, int b) { return a>b ? a : b; }

//parallel height using tasks
int height(bstnode* root) {
    if(!root) return 0;
    
    int left_height = 0, right_height = 0;
    
    #pragma omp task shared(left_height)
    left_height = height(root->left);
    
    #pragma omp task shared(right_height)
    right_height = height(root->right);
    
    #pragma omp taskwait
    
    return max(left_height, right_height) + 1;
}

//parallel isbalanced using tasks  
bool isbalanced(bstnode* root) {
    if(!root) return true;
    
    int left_height = 0, right_height = 0;
    bool left_balanced = true, right_balanced = true;
    
    #pragma omp task shared(left_balanced, left_height)
    {
        left_balanced = isbalanced(root->left);
        left_height = height(root->left);
    }
    
    #pragma omp task shared(right_balanced, right_height)
    {
        right_balanced = isbalanced(root->right);
        right_height = height(root->right);
    }
    
    #pragma omp taskwait
    
    return (abs(left_height - right_height) < 2) && 
           left_balanced && right_balanced;
}

#define NDF -1

int lcg(int seed = NDF) {
    constexpr int m = 134456;
    constexpr int a = 8121;
    constexpr int c = 28411;
    static int x = 0;
    return x = seed==NDF ? (a*x+c)%m : seed;
}


//return a pointer to a random BST of which keys are in [lower,upper]
bstnode* buildbalanced(int lower, int upper, const int threshold, int &node_counter) {
    if (upper<lower) return nullptr;
    int key = (upper+lower)/2;
    bstnode* root = new bstnode(key);
    node_counter = node_counter+1;
    if(lcg()%100<threshold) root->left = buildbalanced(lower, key-1, threshold, node_counter);
    if(lcg()%100<threshold) root->right = buildbalanced(key+1, upper, threshold, node_counter);
    return root;
}

int main(int argc, char *argv[]) {
   
    #pragma omp parallel
    #pragma omp single
    {
        {
            const int h = atoi(argv[1]);
            const int threshold = atoi(argv[2]);
            int nnodes = 0;
            lcg(264433);
            bstnode *root = buildbalanced(1, (1<<h)-1, threshold, nnodes);
            printf("T#0, %d nodes, isbalanced = %c\n", nnodes, isbalanced(root)?'Y':'N');
            delete root;
        }
        {
            int nnodes = 0;
            lcg(264433);
            bstnode *root = buildbalanced(1, 511, 90, nnodes);
            printf("T#1, %d nodes, isbalanced = %c\n", nnodes, isbalanced(root)?'Y':'N');
            delete root;
        }
        {
            int nnodes = 0;
            lcg(123);
            bstnode *root = buildbalanced(1, 32767, 90, nnodes);
            printf("T#2, %d nodes, isbalanced = %c\n", nnodes, isbalanced(root)?'Y':'N');
            delete root;
        }
    }

    return EXIT_SUCCESS;
}