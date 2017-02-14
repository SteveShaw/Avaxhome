//<--> 80. Remove Duplicates from Sorted Array II
/*
Follow up for "Remove Duplicates":
What if duplicates are allowed at most twice?

For example,
Given sorted array nums = [1,1,1,2,2,3],

Your function should return length = 5,
with the first five elements of nums being 1, 1, 2, 2 and 3.
It doesn't matter what you leave beyond the new length.
*/

class Solution {
public:
    int removeDuplicates(vector<int>& nums)
    {
        int n = nums.size();
        int pre = 0, cur = 1, count  = 1;
        while(cur < n)
        {
            if(nums[pre]==nums[cur] && count == 0)
            {
                ++cur;
            }
            else
            {
                if(nums[pre]==nums[cur])
                {
                    --count;
                }
                else
                {
                    count = 1;
                }
                
                nums[pre++] = nums[++cur];
            }
        }
        
        return pre+1;
    }
};

//<--> 81. Search in Rotated Sorted Array II
/*
Suppose an array sorted in
ascending order is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Write a function to determine if a given target is in the array.

The array may contain duplicates.
*/
/*
 A[mid] =  target, 返回mid，否则

(1) A[mid] < A[end]: A[mid+1 : end] sorted
A[mid] < target <= A[end]  右半，否则左半。

(2) A[mid] > A[end] : A[start : mid-1] sorted
A[start] <= target < A[mid] 左半，否则右半。

当有重复数字，会存在A[mid] = A[end]的情况

A[mid] = A[end] != target时：搜寻A[start : end-1]

 */

class Solution {
public:
    bool search(vector<int>& nums, int target)
    {
        if(nums.empty())
        {
            return false;
        }
        
        int left = 0, right = nums.size() - 1;
        
        while(left<=right)
        {
            int mid = left + (right-left)/2;
            if(nums[mid] == target)
            {
                return true;
            }
            
            if(nums[mid] < nums[right]) // key: right half is sorted
            {
                if(target > nums[mid] && target<= nums[right] )
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid-1;
                }
            }
            else if(nums[mid] > nums[right]) //key: left half is sorted
            {
                if(target>=nums[left] && target<nums[mid])
                {
                    right = mid-1;
                }
                else
                {
                    left = mid + 1;
                }
            }
            else
            {
                --right;
            }
        }
        
        return false;
    }
};

//<--> 82. Remove Duplicates from Sorted List II
/*
Given a sorted linked list,
delete all nodes that have duplicate numbers,
leaving only distinct numbers from the original list.

For example,
Given 1->2->3->3->4->4->5, return 1->2->5.
Given 1->1->1->2->3, return 2->3.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head)
    {
        //my method need a count to count the duplicates
        
         if(!head)
        {
            return nullptr;
        }
        
        auto new_head = new ListNode(-1);
        auto pre = new_head;
        pre->next = head;
        auto cur = head;
        auto next = cur->next;
        
        int count = 0;
        
        while(next)
        {
            if(cur->val==next->val)
            {
                ++count;
            }
            else
            {
                if(count > 0)
                {
                    pre->next = next;
                    count = 0;
                }
                else
                {
                    pre = cur;
                }
                
                cur = next;
            }
            
            next = next->next;
        }
        
        if(count>0)
        {
            pre->next = next;
        }
        else
        {
            cur->next = next;
        }
        
        return new_head->next;
    }
};

//<--> 83. Remove Duplicates from Sorted List
/*
Given a sorted linked list,
delete all duplicates such that each element appear only once.

For example,
Given 1->1->2, return 1->2.
Given 1->1->2->3->3, return 1->2->3.
*/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head)
    {
        if(!head) return nullptr;
        
        auto cur = head;
        auto next = cur->next;
        
        while(next)
        {
            if(cur->val!=next->val)
            {
                cur->next = next;
                cur = next;
            }
            
            next = next->next;
        }
        
        cur->next = next;
        
        return head;
    }
};
//<--> 84. Largest Rectangle in Histogram
/*
Given n non-negative integers representing the histogram's bar height
where the width of each bar is 1,
find the area of largest rectangle in the histogram.
*/
class Solution {
public:
    //first method: using stack
    int largestRectangleArea(vector<int>& heights)
    {
        heights.push_back(0);//important
        int n = heights.size();
        stack<int> s;
        
        int max_area = 0;
        
        for(int i = 0; i<n; ++i)
        {
            if(s.empty() || heights[s.top()] <= heights[i])
            {
                s.push(i);
            }
            else
            {
                auto cur = s.top();
                s.pop();
                
                int w = s.empty() ? i : (i-s.top()-1); //important: i - s.top() - 1
                max_area = max(max_area, heights[cur]*w);
                --i; //very important
            }
        }
        
        return max_area;
    }
    
    //not using stack
    int largestRectangleArea(vector<int>& heights)
    {
        auto& h = heights; //for keyboard input simpicity
        h.push_back(0); //important
        
        int n = h.size();
        
        int max_area = 0;
        
        for( int i = 0; i<n; ++i)
        {
            if(h[i]<=h[i+1])
            {
                continue;
            }
            
            auto min_h = h[i];
            for(int j = i; j>=0; --j)
            {
                min_h = min(min_h, h[j]);
                int area = min_h * (i-j+1); //important: i - j + 1;
                max_area = max(max_area, area);
            }
        }
        
        return max_area;
    }
    
};


//<--> 85. Maximal Rectangle
/*
Given a 2D binary matrix filled with 0's and 1's,
find the largest rectangle containing only 1's and return its area.

For example, given the following matrix:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Return 6.
*/
class Solution {
public:
	//think of each col is a histo
	int maximalRectangle( vector<vector<char>>& matrix )
	{
		int res = 0;

		auto& m = matrix;

		if ( m.empty() || m[0].empty() )
		{
			return res;
		}

		int r = m.size(), c = m[0].size();

		vector<int> h( c + 1, 0 );

		for ( int i = 0; i<r; ++i )
		{
			for ( int j = 0; j<c; ++j )
			{
				if ( m[i][j] == '1' ) //important : m[i][j] == 0 will reset height to zero
				{
					h[j] += 1; 
				}
				else
				{
					h[j] = 0;
				}
			}

			stack<int> s;
			int cur_max_area = 0;
			for ( int i = 0; i <= c; ++i )
			{
				if ( s.empty() || h[s.top()] <= h[i] )
				{
					s.push( i );
				}
				else
				{
					auto cur = s.top();
					s.pop();

					int w = s.empty() ? i : (i - s.top() - 1);
					cur_max_area = max( cur_max_area, w*h[cur] );
					--i;
				}
			}

			res = max( res, cur_max_area );
		}

		return res;
	}
};

//<--> 86. Partition List
/*
Given a linked list and a value x,
partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

For example,
Given 1->4->3->2->5->2 and x = 3,
return 1->2->2->4->3->5.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    // first method : first search the first item that no less than x,
    // than put all items less than x before this item.
	ListNode* partition( ListNode* head, int x )
	{
		ListNode* dummy = new ListNode( -1 );

		dummy->next = head;

		ListNode* pre = dummy;

		while ( pre->next && pre->next->val < x )
		{
			pre = pre->next;
		}

		auto cur = pre;

		while ( cur->next )
		{
			if ( cur->next->val < x )
			{
				auto tmp = cur->next;

				cur->next = tmp->next;

				tmp->next = pre->next;
				pre->next = tmp;

				pre = pre->next;
			}
			else
			{
				cur = cur->next;
			}
		}

		auto new_head = dummy->next;
		delete dummy;

		return new_head;
	}
    
    // second method: using two headers: one point to less than x, and another to opposite
   ListNode* partition( ListNode* head, int x )
   {
       if ( !head )
       {
           return head;
       }
       auto big_head = new ListNode( -1 );
       big_head->next = head;
   
       auto small_head = new ListNode( -1 );
   
       auto b = big_head, s = small_head;
   
       while ( b->next )
       {
           if ( b->next->val < x )
           {
               s->next = b->next;
               s = s->next;
   
               b->next = b->next->next; // key: skip b->next;
               s->next = nullptr; // key: broke the big chain.
           }
           else
           {
               b = b->next;
           }
   
       }
   
       s->next = big_head->next;
       delete big_head;
   
       s = small_head->next;
       delete small_head;
   
       return s;
   }
};

//<--> 87. Scramble String
/*
Given a string s1, we may represent it as a binary tree
by partitioning it to two non-empty substrings recursively.

Below is one possible representation of s1 = "great":

    great
   /    \
  gr    eat
 / \    /  \
g   r  e   at
           / \
          a   t
To scramble the string, we may choose any non-leaf node and swap its two children.

For example, if we choose the node "gr" and swap its two children,
it produces a scrambled string "rgeat".

    rgeat
   /    \
  rg    eat
 / \    /  \
r   g  e   at
           / \
          a   t
We say that "rgeat" is a scrambled string of "great".

Similarly, if we continue to swap the children of nodes "eat" and "at",
it produces a scrambled string "rgtae".

    rgtae
   /    \
  rg    tae
 / \    /  \
r   g  ta  e
       / \
      t   a
We say that "rgtae" is a scrambled string of "great".

Given two strings s1 and s2 of the same length,
determine if s2 is a scrambled string of s1.
*/

class Solution {
public:
	bool isScramble( string s1, string s2 ) {

		if ( s1.size() != s2.size() )
		{
			return false;
		}

		if ( s1 == s2 )
		{
			return true;
		}

		vector<int> m( 26, 0 );
		int n = s1.size();

		for ( int i = 0; i< n; ++i )
		{
			++m[s1[i] - 'a'];
			--m[s2[i] - 'a'];
		}

		for ( auto v : m )
		{
			if ( v != 0 )
			{
				return false;
			}
		}

		for ( int i = 1; i<n; ++i ) // key: i start from 1, not from 0
		{
			string s11 = s1.substr( 0, i );
			string s12 = s1.substr( i );
			string s21 = s2.substr( 0, i );
			string s22 = s2.substr( i );

			if ( isScramble( s11, s21 ) && isScramble( s12, s22 ) ) return true;

			s21 = s2.substr( n - i ); //length = i (n - (n-i));
			s22 = s2.substr( 0, n - i );//length = n-i (n-i - 0)

			if ( isScramble( s11, s21 ) && isScramble( s12, s22 ) ) return true;
		}

		return false;
	}
};

//<--> 88. Merge Sorted Array
/*
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n)
to hold additional elements from nums2.
The number of elements initialized in nums1 and nums2 are m and n respectively.
*/
class Solution {
public:
    //start from end (m + n -1)
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
    {
        auto & A = nums1;
        auto & B = nums2;
        
        int s = m + n - 1;
        
        m -= 1; //key: need to put end to m-1/n-1
        n -= 1;
        
        while(m>=0 && n>=0)
        {
            A[s--] = A[m] > B[n] ? A[m--] : B[n--];
            //note: cannot write as follows:
            /*
             *  A[s] = A[m] > B[n] ? A[m] : B[n];
             *  --s;
             *  --m;
             *  --n;
             *  Because --m or --n is not happening at the same time
             */
        }
        
        while(n>=0)     //important: nums2 may have additionl items.
        {
            A[s] = B[n];
            --s;
            --n;
        }
    }
};

//<--> 89. Gray Code
/*
The gray code is a binary numeral system where two successive values differ in only one bit.

Given a non-negative integer n representing the total number of bits in the code,
print the sequence of gray code. A gray code sequence must begin with 0.
For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:

00 - 0
01 - 1
11 - 3
10 - 2

Note:
For a given n, a gray code sequence is not uniquely defined.

For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.
*/
class Solution {
public:
    //Gray code property: mirror symmertic
    /*0  ->   00  ->   000
      1  ->   01  ->   001
      -- ->   --
      1  ->   11  ->   011
      0  ->   10  ->   010
              
              ------------
              10  ->   110
              11  ->   111
              01  ->   101
              00  ->   100
    */
    vector<int> grayCode(int n)
    {
        vector<int> res{0};
        
        for(int i = 0; i<n; ++i)
        {
            int cur_sz = res.size();
            for(int j = cur_sz; j>=0; --j) //key: this is mirror: from end to start
            {
                res.push_back(res[j] | (1<<i));
            }
        }
        
        return res;
    }
};

//<--> 90. Subsets II
/*
Given a collection of integers that might contain duplicates, nums,

return all possible subsets.

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,2], a solution is:

[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
*/

class Solution {
public:
    // recursive method
	vector<vector<int>> subsetsWithDup( vector<int>& nums ) {

		vector<vector<int>> res;
		vector<int> out;

		sort( nums.begin(), nums.end() );

		dfs( nums, res, out, 0 );

		return res;

	}

	void dfs( vector<int>& A, vector<vector<int>> &res, vector<int> &out, int start )
	{
		res.push_back( out );

		int n = A.size();

		for ( int i = start; i<A.size(); ++i )
		{
			if ( i>start && A[i] == A[i - 1] ) continue;  //important: avoid duplicate
			out.push_back( A[i] );
			dfs( A, res, out, i + 1 );  //important: avoid using current item again.
			out.pop_back();
		}
	}
    
    // iterative
    vector<vector<int>> subsetsWithDup( vector<int>& nums )
    {
        vector<vector<int>> res(1); //key: need to set size to 1 to let the loop can run
        
        if(nums.empty())
        {
            return res;
        }
        
        sort(nums.begin(), nums.end()); //key: sort is required
        
        int n = nums.size();
        
        int last = nums[0], last_size = 1;
        
        for(int i = 0; i<n; ++i)
        {
            if(last!=nums[i])
            {
                last = nums[i];
                last_size = res.size();
            }
            
            int cur_size = res.size();
            
            for(int j = cur_size - last_size; j< cur_size; ++j)
            {
                res.push_back(res[j]);
                res.back().push_back(nums[i]);
            }
        }
        
        return res;
    }
};

//<--> 91. Decode Ways
/*
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given an encoded message containing digits,

determine the total number of ways to decode it.

For example,
Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

The number of ways decoding "12" is 2.
*/
class Solution {
public:
    // using dp
    // dp[i] = number of ways to decode for s[0..i]
    // if s[i-1] = '1' to '9', dp[i] = dp[i-1]
    // if s[i-2..i-1] = '10' to '26' dp[i] += dp[i-2]
    // dp[0][0] = 1: this is the key
    int numDecodings(string s) 
    {
        if(s.empty()) return 0;
        
        int n = s.size();
        
        vector<int> dp(n+1, 0);
        
        dp[0] = 1; //key: empty string only have 1 way to decode
        
        for(int i = 1; i<=n; ++i)
        {
            if(s[i-1] != '0')  //s[i-1] = '1' to '9'
            {
                dp[i] = dp[i-1];
            }
            
            if(i>1)
            {
                if(s[i-2] == '1' || ( s[i-2]=='2' && s[i-1] >= '0' && s[i-1]<= '6' ) ) //s[i-2..i-1] = '10' to '26'
                {
                    dp[i] += dp[i-2];
                }
            }
        }
        
        return dp[n];
    }
};

//<--> 92. Reverse Linked List II
/*
Reverse a linked list from position m to n. Do it in-place and in one-pass.

For example:
Given 1->2->3->4->5->NULL, m = 2 and n = 4,

return 1->4->3->2->5->NULL.

Note:
Given m, n satisfy the following condition:
1 <= m <= n <= length of list.
*/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    // 一个比较普适的方法是，固定起始点前面的那个节点pre，然后不停的
    // 把翻转范围内的节点一个个的放到pre->next上
    // example: 1->2->3->4->5->null, m=2, n = 4
    // pre = 1, so we will have
    // 1) : 1->3->2->4->5->null
    // 2) : 1->4->3->2->5->null
    // the required steps is n - m;
    ListNode* reverseBetween(ListNode* head, int m, int n)
    {
        ListNode* dummy = new ListNode(-1);
        dummy->next = head;
        ListNode* prev = dummy;
        
        for(int i = 1; i<=m-1; ++i) //key: i from 1 to m-1
        {
            prev = prev->next;
        }
        
        //now pre is the previous node of the range start node.
        auto cur = prev->next;
        
        for(int i = m; i<n; ++i) //number of steps = n - m
        {
            auto tmp = cur->next;
            cur->next = tmp->next; //set to cur->next->next;
            
            tmp->next = prev->next; //put 3 to pre->next for the first loop.
            pre->next = tmp; // now 1->3 for the first loop.
        }
        
        head = dummy->next;
        delete dummy;
        return head;
    }
};

//<--> 93. Restore IP Addresses
/*
Given a string containing only digits,
restore it by returning all possible valid IP address combinations.

For example:
Given "25525511135",

return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
*/
class Solution {
public:
    //my own solution: using recursive.
	vector<string> restoreIpAddresses( string s )
    {

		vector<string> res;

		if ( s.empty() || s.size()<4 || s.size() > 16 )
		{
			return res;
		}


		vector<string> out;

		split_addr( res, out, s, 0 );

		return res;
	}

	void split_addr( vector<string>& res, vector<string>& out, string& s, int start )
	{
		if ( start > s.size() )
		{
			return;
		}

		int n = out.size();

		if ( 4 == n && start == s.size() )
		{
			string ip;
			for ( const auto& s : out )
			{
				ip += s;
				ip += ".";
			}
			ip.pop_back();
			res.push_back( ip );
			return;
		}

		if ( s.size() >= start + 1 )
		{
			string num = s.substr( start, 1 );
			out.push_back( num );
			split_addr( res, out, s, start + 1 );
			out.pop_back();
		}


		if ( s.size() >= start + 2 )
		{
			string num = s.substr( start, 2 );
			if ( isValidIP( num ) )
			{
				out.push_back( num );
				split_addr( res, out, s, start + 2 );
				out.pop_back();
			}
		}


		if ( s.size() >= start + 3 )
		{
			string num = s.substr( start, 3 );
			if ( isValidIP( num ) )
			{
				out.push_back( num );
				split_addr( res, out, s, start + 3 );
				out.pop_back();
			}
		}

	}

	bool isValidIP( string& s )
	{
		if ( s.size() == 3 )
		{
			if ( s[0] == '1' ) return true;

			if ( s[0] == '2' )
			{
				if ( s[1] >= '0' && s[1] <= '4' )
				{
					return true;
				}

				if ( s[1] == '5' && s[2] >= '0' && s[2] <= '5' )
				{
					return true;
				}
			}

			return false;
		}

		if ( s.size() == 2 )
		{
			return s[0] != '0';
		}

		if ( s.size() == 1 )
		{
			return true;
		}

		return false;
	}
};

//<--> 94. Binary Tree Inorder Traversal
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //recursive
    vector<int> inorderTraversal(TreeNode* root)
    {
        vector<int> res;
        traverse(root, res);
        
        return res;
    }
    
    void traverse(TreeNode* root, vector<int>& out)
    {
        if(root)
        {
            traverse(root->left, out);
            out.push_back(root->val);
            traverse(root->right, out);
        }
    }
    
    //iterative
    vector<int> inorderTraversal(TreeNode* root)
    {
        auto p = root;
        stack<TreeNode*> s;
        vector<int> res;
        
        while(p || !s.empty()) //key: if p is not null or stack is not empty
        {
            while(p) //important: push all left child into stack first
            {
                s.push(p);
                p = p->left; 
            }
            
            p = s.top(); //key: get lowest level left node
            s.pop();
            
            res.push_back(p->val);
            
            p = p->right; //key: turn to right child
        }
        
        return res;
    }
    
    //using O(1) space
    vector<int> inorderTraversal(TreeNode* root)
    {
        if(!root)
        {
            return {};
        }
        
        vector<int> res;
        
        auto cur = root;
        TreeNode* pre = nullptr;
        
        while(cur)
        {
            if(!cur->left)
            {
                res.push_back(cur->val);
                cur = cur->right;
            }
            else
            {
                pre = cur->left;
                while(pre->right && pre->right!=cur)
                {
                    pre = pre->right;
                }
                
                if(pre->right==nullptr) //key: if right is not set, then cur is not visited now
                {
                    pre->right = cur;
                    cur = cur->left;
                }
                else
                {
                    pre->right = nullptr; //key: if right is set, then visit cur now.
                    res.push_back(cur->val);
                    cur = cur->right;
                }
            }
        }
        
        return res;
    }
};

//<--> 95. Unique Binary Search Trees II
/*
Given an integer n, generate all structurally unique BST's (binary search trees)
that store values 1...n.

For example,
Given n = 3, your program should return all 5 unique BST's shown below.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int n)
    {
        if(n==0)
        {
            return {};
        }
        
        vector<TreeNode*> res;
        dfs(1,n,res);
        
        return res;
    }
    
    void dfs(int start, int end, vector<TreeNode*>& out)
    {
        if(start>end)
        {
            out.push_back(nullptr); //add null pointer
            return;
        }
        
        for(int i = start; i<=end; ++i)
        {
            vector<TreeNode*> left;
            vector<TreeNode*> right;
            
            dfs(start, i-1, left); //key: create tree for [start, i-1];
            dfs(i+1, end, right); //key: create right tree of i for [i+1, end];
            
            
            for(size_t j = 0; j<left.size(); ++j)
            {
                for(size_t k = 0; k<right.size(); ++k)
                {
                    TreeNode* tn = new TreeNode(i);
                    tn->left = left[j]; //important: 对产生的left子树集合的每一个sub tree，接到节点的 left
                    tn->right = right[j]; //important: 对产生的right子树集合的每一个sub tree，接到节点的 right
                    
                    out.push_back(tn);
                }
            }
        }
    }
};

//<--> 96. Unique Binary Search Trees
/*
Given n, how many structurally unique BST's (binary search trees) that store values 1...n?

For example,
Given n = 3, there are a total of 5 unique BST's.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3
*/

/*
 定义f(n)为unique BST的数量，以n = 3为例：

构造的BST的根节点可以取{1, 2, 3}中的任一数字。

如以1为节点，则left subtree只能有0个节点，
而right subtree有2, 3两个节点。所以left/right subtree一共的combination数量为：f(0) * f(2) = 2

以2为节点，则left subtree只能为1，right subtree只能为3：f(1) * f(1) = 1

以3为节点，则left subtree有1, 2两个节点，right subtree有0个节点：f(2)*f(0) = 2

总结规律：
f(0) = 1
f(n) = f(0)*f(n-1) + f(1)*f(n-2) + ... + f(n-2)*f(1) + f(n-1)*f(0)
 */

class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1,0);
        
        dp[0] = 1;
        
        for(int i = 1; i<=n; ++i)
        {
            for(int j = 0; j<i; ++j)
            {
                dp[i] = dp[j] * dp[i-j-1];
            }
        }
        
        return dp[n];
    }
};

//<--> 97. Interleaving String
/*
Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.

For example,
Given:
s1 = "aabcc",
s2 = "dbbca",

When s3 = "aadbbcbcac", return true.
When s3 = "aadbbbaccc", return false.
*/

class Solution {
public:
    // dp based: recursive will exceed time limit.
    // dp[i][j] : if s1[0..i-1] and s2[0..j-1] are interleaving to s3[0...i+j-1]
    
    bool isInterleave(string s1, string s2, string s3)
    {
        if(s1.size()+s2.size()!=s3.size())
        {
            return false;
        }
        
        vector<vector<int>> dp(s1.size()+1, vector<int>(s2.size()+1,0));
        
        dp[0][0] = 1;
        
        for(size_t i = 1; i<=s1.size();++i)
        {
            if( s1[i-1] == s3[i-1] )
            {
                dp[i][0] = dp[i-1][0];
            }
        }
        
        for(size_t j = 1; j<=s2.size(); ++j)
        {
            if( s2[j-1] == s3[j-1] )
            {
                dp[0][j] = dp[0][j-1];
            }
        }
        
        for(size_t i = 1; i<=s1.size();++i)
        {
            for(size_t j = 1; j<=s2.size(); ++j)
            {
                if(s1[i-1] == s3[i-1+j])
                {
                    dp[i][j] = dp[i-1][j];
                }
                
                if(s2[j-1] == s3[j-1+i])
                {
                    dp[i][j] = dp[i][j] | dp[i][j-1];
                }
            }
        }
        
        return dp.back().back() == 1;
    }
};

//<--> 98. Validate Binary Search Tree
/*
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
Example 1:
    2
   / \
  1   3
Binary tree [2,1,3], return true.
Example 2:
    1
   / \
  2   3
Binary tree [1,2,3], return false.
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //recursive
    bool isValidBST(TreeNode* root)
    {
        return f(root, LONG_MIN, LONG_MAX);
    }
    
    bool f( TreeNode* root, long min, long max ) //key: using long to avoid int type overflow
    {
        if(!root)
        {
            return true;
        }
        
        long val = root->val;
        
        if( val <= min && val >= max )
        {
            return false;
        }
        
        return f(root->left, min, val) && f(root->right, val, max);
        
    }
    
    //using inorder iterative
    
    bool isValidBST(TreeNode* root)
    {
        auto p = root;
        
        TreeNode* pre = nullptr;
        
        stack<TreeNode*> s;
        
        while( p || !s.empty() )
        {
            while(p)
            {
                s.push(p);
                p = p->left;
            }
            
            p = s.top();
            s.pop();
            if(pre && p.val <= pre.val)
            {
                return false;
            }
            
            pre = p;
            p = p->right;
        }
        
        return true;
    }
};

//<--> 99. Recover Binary Search Tree
/*
Two elements of a binary search tree (BST) are swapped by mistake.

Recover the tree without changing its structure.

Note:
A solution using O(n) space is pretty straight forward.
Could you devise a constant space solution?
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //using inorder (recursive) because inorder gives ascending orders
    void recoverTree(TreeNode* root)
    {
        if(!root)
        {
            return;
        }
        
        vector<TreeNode*> node_list;
        vector<int> val_list;
        
        inorder(root, node_list, val_list);
        sort(val_list.begin(), val_list.end()); //key : must sort.
        
        for(size_t i = 0; i<node_list.size(); ++i)
        {
            node_list[i]->val = val_list[i];
        }
    }
    
    void inorder( TreeNode* root, vector<TreeNode*>& nodes, vector<int>& vals )
    {
        if(root)
        {
            inorder(root->left, nodes, vals);
            nodes.push_back(root);
            vals.push_back(root->val);
            inorder(root->right, nodes, vals);
        }
    }
    
    //using morris method: O(1) space
    void recoverTree(TreeNode* root)
    {
        TreeNode* first = nullptr;
        TreeNode* second = nullptr;
        TreeNode* parent = nullptr;
        
        auto cur = root;
        TreeNode* pre = nullptr;
        
        while(cur)
        {
            if(!cur->left)
            {
                if(parent && parent->val > cur->val)
                {
                    if(!first)
                    {
                        first = parent;
                    }
                    
                    second = cur;
                }
                
                parent = cur; //key: visit cur --> set parent to cur
                cur = cur->right;
            }
            else
            {
                pre = cur->left;
                while(pre->right && pre->right != cur) //key: the condition is pre->right is not null, not pre is not null
                {
                    pre = pre->right;
                }
                
                if(!pre->right)
                {
                    pre->right = cur;
                    cur = cur->left;
                }
                else
                {
                    pre->right = nullptr;
                    if(parent->val > cur->val)
                    {
                        if(!first) first = parent;
                        second = cur;
                    }
                    
                    parent = cur; //key: visit cur --> set parent to cur
                    cur = cur->right;
                }
            }
        }
        
        if(first && second)
        {
            swap(first->val, second->val);
        }
    }
};

//<--> 100. Same Tree
/*
Given two binary trees, write a function to check if they are equal or not.

Two binary trees are considered equal
if they are structurally identical
and the nodes have the same value.
*/
class Solution {
public:
    //this is a very easy question
    bool isSameTree(TreeNode* p, TreeNode* q)
    {
        if(!p && !q)
        {
            return true;
        }
        
        if(p && q)
        {
            if(p->val != q->val)
            {
                return false;
            }
            
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }
        
        return false;
    }
};

//<--> 101. Symmetric Tree
/*
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3

But the following [1,2,2,null,3,null,3] is not:
    1
   / \
  2   2
   \   \
   3    3
Note:
Bonus points if you could solve it both recursively and iteratively.
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //using recursive
    bool isSymmetric(TreeNode* root)
    {
        if(!root) return true;
        return compare(root->left, root->right);
    }
    
    bool compare(TreeNode* l, TreeNode* r)
    {
        if(!l && !r)
        {
            //both null, is same
            return true;
        }
        
        if(l && r)
        {
            if(l->val != r->val)
            {
                return false;
            }
            
            return compare(l->left, r->right) && compare(l->right, r->left);
        }
        
        return false;
    }
    
    //iterative: using two queues
    bool isSymmetric(TreeNode* root)
    {
        if(!root)
        {
            return true;
        }
        
        queue<TreeNode*> q1;
        queue<TreeNode*> q2;
        
        q1.push(root->left);
        q2.push(root->right);
        
        while(!q1.empty() && !q2.empty())
        {
            auto node1 = q1.front();
            auto node2 = q2.front();
            q1.pop();
            q2.pop();
            
            if(!node1 && !node2)
            {
                continue; //important: need to continue since other branches need to visit
            }
            
            if(node1 && node2)
            {
                if(node1->val != node2->val)
                {
                    return false;
                }
                
                q1.push(node1->left);
                q1.push(node1->right);
                q2.push(node2->right);
                q2.push(node2->left);
            }
            else
            {
                return false;
            }
        }
        
        return true;
    }
};

//<--> 102. Binary Tree Level Order Traversal
/*
Given a binary tree, return the level order traversal of its nodes' values.

(ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
*/
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root)
    {
        if(!root)
        {
            return {};
        }
        
        queue<TreeNode*> q;
        vector<vector<int>> res;
        
        q.push(root);
        
        while(!q.empty())
        {
            auto cur_sz = q.size();
            
            vector<int> level;
            
            for(size_t i = 0; i<cur_sz; ++i)
            {
                auto node = q.front();
                q.pop();
                
                if(node)
                {
                    level.push_back(node->val);
                    q.push(node->left);
                    q.push(node->right);
                }
            }
            
            if(!level.empty())
            {
                res.push_back(level);
            }
        }
        
        return res;
    }
};

//<--> 103. Binary Tree Zigzag Level Order Traversal
/*
Given a binary tree, return the zigzag level order traversal of its nodes' values.

(ie, from left to right, then right to left for the next level and alternate between).

For example:

Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
	vector<vector<int>> zigzagLevelOrder( TreeNode* root ) {

		if ( !root )
		{
			return{};
		}

		bool bL2R = true;

		queue<TreeNode*> q;
		q.push( root );

		vector<vector<int>> res;

		while ( !q.empty() )
		{
			int cur_sz = q.size();

			vector<int> level;

			if ( bL2R )
			{
				for ( int i = 0; i<cur_sz; ++i )
				{
					auto node = q.front();
					q.pop();

					if ( node )
					{
						level.push_back( node->val );
						q.push( node->left );
						q.push( node->right );
					}
				}
			}
			else
			{
				for ( int i = 0; i<cur_sz; ++i )
				{
					auto node = q.front();
					q.pop();

					if ( node )
					{
						level.insert( level.begin(), node->val ); //key: insert before current.
						q.push( node->left );
						q.push( node->right );
					}
				}
			}

			bL2R = !bL2R;

			if ( !level.empty() )
			{
				res.push_back( level );
			}

		}

		return res;
	}
};

//<--> 104. Maximum Depth of Binary Tree
/*
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes
along the longest path from the root node
down to the farthest leaf node.
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxDepth(TreeNode* root) {
        
        if(!root) return 0;
        
        int l = maxDepth(root->left);
        int r = maxDepth(root->right);
        
        return max(l,r)+1;
    }
};

//<--> 105. Construct Binary Tree from Preorder and Inorder Traversal
/*
Given preorder and inorder traversal of a tree,
construct the binary tree.
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder)
    {
        return bt(preorder, 0, preorder.size()-1, inorder, 0, inorder.size()-1);
    }
    
    TreeNode* bt(vector<int>& preorder, int p_left, int p_right, vector<int>& inorder, int i_left, int i_right)
    {
        if(p_left > p_right || i_left > i_right)
        {
            return nullptr;
        }
        
        int i = i_left;
        for(; i<=i_right; ++i)
        {
            if(preorder[p_left]==inorder[i])
            {
                break;
            }
        }
        
        TreeNode* cur = new TreeNode(preorder[p_left]);
        cur->left = bt(preorder, p_left+1, p_left + i - i_left, inorder, i_left, i-1);
        cur->right = bt(preorder, p_left + i - i_left + 1, p_right, inorder, i + 1, i_right);
        
        return cur;
    }
};

//<--> 106. Construct Binary Tree from Inorder and Postorder Traversal
/*
Given inorder and postorder traversal of a tree, construct the binary tree.
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder)
    {    
    }
    
    TreeNode* bt(vector<int>& postorder, int p_left, int p_right, vector<int>& inorder, int i_left, int i_right)
    {
        if(p_left > p_right || i_left > i_right)
        {
            return nullptr;
        }
        
        int i = i_left;
        for(; i<=i_right; ++i)
        {
            if(postorder[p_right]==inorder[i])
            {
                break;
            }
        }
        
        TreeNode* cur = new TreeNode(postorder[p_right]);
        cur->left = bt(postorder, p_left, p_left + i - i_left - 1, inorder, i_left, i-1);
        cur->right = bt(postorder, p_left + i - i_left, p_right-1, inorder, i + 1, i_right);
        
        return cur;
    }
};

//<--> (TODO) 107. Binary Tree Level Order Traversal II
/*
Given a binary tree, return the bottom-up level order traversal

of its nodes' values. (ie, from left to right, level by level from leaf to root).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its bottom-up level order traversal as:
[
  [15,7],
  [9,20],
  [3]
]
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root)
    {    
    }
};


//<--> 108. Convert Sorted Array to Binary Search Tree

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums)
    {
        return bt(nums, 0, nums.size()-1);
    }
    
    TreeNode* bt( vector<int>& nums, int left, int right )
    {
        if(left > right)
        {
            return nullptr;
        }
        
        int mid = (right+left)/2;
        TreeNode* cur = new TreeNode(nums[mid]);
        cur->left = bt(nums, left, mid-1);
        cur->right = bt(nums, mid+1, right);
        
        return cur;
    }
};

//<--> 109. Convert Sorted List to Binary Search Tree
/*
Given a singly linked list where elements are
sorted in ascending order,
convert it to a height balanced BST.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedListToBST(ListNode* head)
    {
        if(!head)
        {
            return nullptr;
        }
        
        if(!head->next)
        {
            return new TreeNode(head->val);
        }
        
        auto slow = head;
        auto fast = head;
        auto prev = head;
        
        while(fast->next && fast->next->next)
        {
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
        }
        
        fast = slow->next;
        prev->next = nullptr;
        
        TreeNode* cur = new TreeNode(slow->val);
        if(head!=slow)
        {
            cur->left = sortedListToBST(head);
        }
        
        cur->right = sortedListToBST(fast);
        
        return cur;
    }
};

//<--> 110. Balanced Binary Tree
/*
Given a binary tree, determine if it is height-balanced.
For this problem, a height-balanced binary tree is defined as a binary tree
in which the depth of the two subtrees of every node never differ by more than 1.
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
// efficient way
class Solution {
public:
    bool isBalanced(TreeNode* root)
    {
        return check_depth(root) != -1;
    }
    
    int check_depth(TreeNode* root)
    {
        if(!root)
        {
            return 0;
        }
        
        int left = check_depth(root->left);
        if(left == -1)
        {
            return -1;
        }
        
        int right = check_depth(root->right);
        if(right == -1)
        {
            return -1;
        }
        
        int diff = abs(left-right);
        if(diff >1)
        {
            return -1;
        }
        
        return 1+max(left, right);
    }
};

//<--> 111. Minimum Depth of Binary Tree
/*
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along

the shortest path from the root node down to the nearest leaf node.
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int minDepth(TreeNode* root)
    {
        if(!root)
        {
            return 0;
        }
        
        if(!root->left && root->right)
        {
            return 1+minDepth(root->right);
        }
        
        if(!root->right && root->left)
        {
            return 1+minDepth(root->left);
        }
        
        return 1+min(minDepth(root->left), minDepth(root->right));
    }
};

//<--> 112. Path Sum
/*
Given a binary tree and a sum,
determine if the tree has a root-to-leaf path such
that adding up all the values along the path equals the given sum.

For example:
Given the below binary tree and sum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum)
    {
        if(!root)
        {
            return false;
        }
        
        if(!root->left && !root->right && root->val == sum)
        {
            return true;
        }
        
        return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->right);
    }
};

//<--> 113. Path Sum II
/*
Given a binary tree and a sum,
find all root-to-leaf paths where each path's sum equals the given sum.

For example:
Given the below binary tree and sum = 22,
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
return
[
   [5,4,11,2],
   [5,8,4,5]
]
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum)
    {
        vector<vector<int>> res;
        vector<int> out;
        
        dfs(res, out, root, sum);
        
        return res;
    }
    
    void dfs(vector<vector<int>>& res, vector<int>& out, TreeNode* root, int sum)
    {
        if(!root)
        {
            return;
        }
        
        out.push_back(root->val);
        if(root->val == sum && !root->left && !root->right)
        {
            res.push_back(out);
            //return; --> must not call return here, otherwise, pop_back will not be called.
        }
        
        dfs(res, out, root->left, sum - root->val);
        dfs(res, out, root->right, sum - root->val);
        
        out.pop_back();
    }
};

//<--> 114. Flatten Binary Tree to Linked List
/*
Given a binary tree, flatten it to a linked list in-place.

For example,
Given

         1
        / \
       2   5
      / \   \
     3   4   6
     
The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //recursive method
    void flatten(TreeNode* root)
    {
        if(!root)
        {
            return;
        }
        
        if(root->left)
        {
            flatten(root->left);
        }
        
        if(root->right)
        {
            flatten(root->right);
        }
        
        auto tmp = root->right;
        root->right = root->left;
        root->left = nullptr;
        
        while(root->right)
        {
            root = root->right;
        }
        
        root->right = tmp;
    }
    
    //iterative: starting from root
    void flatten(TreeNode* root)
    {
        auto cur = root;
        while(cur)
        {
            if(cur->left)
            {
                auto p = cur->left;
                while(p->right)
                {
                    p = p->right;
                }
                
                p->right = cur->right;
                cur->right = cur->left;
                cur->left = nullptr;
            }
            
            cur = cur->right;
        }
    }
};

//<--> 115. Distinct Subsequences
/*
Given a string S and a string T, count the number of distinct subsequences of T in S.

A subsequence of a string is a new string which is
formed from the original string by deleting some (can be none)
of the characters without disturbing the relative positions of
the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).

Here is an example:
S = "rabbbit", T = "rabbit"

Return 3.
*/
class Solution {
public:
    int numDistinct(string s, string t)
    {
        vector<vector<int>> dp(s.size()+1, vector<int>(t.size()+1));
        int len_s = s.size();
        int len_t = t.size();
        
        for(int i = 0; i<=len_s; ++i)
        {
            dp[i][0] = 1;
        }
        
        for(int i = 1;i<=len_s;++i)
        {
            for(int j=1; j<=len_t; ++j)
            {
                dp[i][j]= dp[i-1][j];
                if(s[i-1]=t[j-1])
                {
                    dp[i][j] += dp[i-1][j-1];
                }
            }
        }
    }
};

//<--> 116. Populating Next Right Pointers in Each Node
/*
Given a binary tree

    struct TreeLinkNode {
      TreeLinkNode *left;
      TreeLinkNode *right;
      TreeLinkNode *next;
    }
Populate each next pointer to point to its next right node.

If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

Note:

You may only use constant extra space.
You may assume that it is a perfect binary tree
(ie, all leaves are at the same level, and every parent has two children).
For example,
Given the following perfect binary tree,
         1
       /  \
      2    3
     / \  / \
    4  5  6  7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \  / \
    4->5->6->7 -> NULL
*/
/**
 * Definition for binary tree with next pointer.
 * struct TreeLinkNode {
 *  int val;
 *  TreeLinkNode *left, *right, *next;
 *  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
 * };
 */
class Solution {
public:
    //method 1: recursive
    void connect(TreeLinkNode *root)
    {
        if(!root)
        {
            return;
        }
        
        if(root->left) root->left->next = root->right;
        if(root->right) root->right->next = root->next ? root->next->left : nullptr;
        
        connect(root->left);
        connect(root->right);
    }
    
    //method2: iterative (using queue)
    void connect(TreeLinkNode *root)
    {
        if(!root)
        {
            return;
        }
        
        queue<TreeLinkNode*> q;
        q.push(root);
        
        while(!q.empty())
        {
            int len = q.size();
            
            for(int i = 0; i<len; ++i)
            {
                auto t = q.front();
                q.pop();
                
                if(i<len-1)
                {
                    t->next = q.front();
                }
                
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
            }
        }
    }
    
    //method 3: O(1) memory usage
    void connect(TreeLinkNode *root)
    {
        if(!root) return;
        auto start = root;
        TreeLinkNode* cur = nullptr;
        
        while(start->left)
        {
            cur = start;
            while(cur)
            {
                cur->left->next = cur->right;
                if(cur->next)
                {
                    cur->right->next = cur->next->left;
                }
                
                cur = cur->next;
            }
            
            start = start->left;
        }
    }
};

//<--> 117. Populating Next Right Pointers in Each Node II
/*
Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree?

Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,
         1
       /  \
      2    3
     / \    \
    4   5    7
After calling your function, the tree should look like:
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL
*/
/**
 * Definition for binary tree with next pointer.
 * struct TreeLinkNode {
 *  int val;
 *  TreeLinkNode *left, *right, *next;
 *  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
 * };
 */
class Solution {
public:
    //Method1: recursive 
    void connect(TreeLinkNode *root)
    {
        if(!root)
        {
            return;
        }
        
        auto p = root->next;
        
        while(p)
        {
            if(p->left)
            {
                p = p->left;
                break;
            }
            
            if(p->right)
            {
                p = p->right;
                break;
            }
            
            p = p->next;
        }
        
        if(root->right)
        {
            root->right->next = p;
        }
        
        if(root->left)
        {
            root->left->next = root->right ? root->right : p;
        }
        
        connect(root->right);
        connect(root->left);
    }
    //method 2: iterative using queue : same as 116's method2
    //method 3: O(1) memory
    void connect(TreeLinkNode* root)
    {
        if(!root)
        {
            return;
        }
        
        auto start = root;
        TreeLinkNode* cur = nullptr;
        
        while(start)
        {
            auto p = start;
            while(p&&!p->left&&p->right)
            {
                p = p->next;
            }
            
            if(!p) return;
            
            start = p->left ? p->left : p->right;
            auto cur = start;
            
            while(p)
            {
                if(cur == p->left)
                {
                    if(p->right)
                    {
                        cur->next = p->right;
                        cur = cur->next;
                    }
                    
                    p = p->next;
                }
                else if(cur == p->right)
                {
                    p = p->next;
                }
                else
                {
                    if(!p->left && !p->right)
                    {
                        p = p->next;
                        continue;
                    }
                    
                    cur->next = p->left ? p->left : p->right;
                    cur = cur->next;
                }
            }
        }
    }
};

//<--> 118. Pascal's Triangle
/*
Given numRows, generate the first numRows of Pascal's triangle.

For example, given numRows = 5,
Return

[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
*/
class Solution {
public:
    vector<vector<int>> generate(int numRows)
    {
        vector<vector<int>> res;
        if(numRows==0)
        {
            return res;
        }
        
        res.push_back(vector<int>(1,1));
        
        for(int i = 1; i<numRows; ++i)
        {
            vector<int> row(i+1, 0);
            row[0] = 1;
            
            for(int j = 1; j<i; ++j)
            {
                row[j] = res[i-1][j] + res[i-1][j-1];
            }
            
            row[i] = 1;
            
            res.push_back(row);
        }
        
        return res;
    }
};

//<--> 119. Pascal's Triangle II
/*
Given an index k, return the kth row of the Pascal's triangle.

For example, given k = 3,
Return [1,3,3,1].

Note:
Could you optimize your algorithm to use only O(k) extra space?
*/
class Solution {
public:
    vector<int> getRow(int rowIndex)
    {
        if(rowIndex < 0)
        {
            return {};
        }
        
        vector<int> out(rowIndex+1,0);
        out[0] = 1;
        
        for(int i = 1; i<=rowIndex; ++i)
        {
            for(int j = rowIndex; j>=1; --j)
            {
                out[j] = out[j] + out[j-1];
            }
        }
        
        return out;
    }
};

//<--> 120. Triangle
/*
Given a triangle, find the minimum path sum from top to bottom.

Each step you may move to adjacent numbers on the row below.

For example, given the following triangle

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

Note:
Bonus point if you are able to do this using only O(n) extra space,

where n is the total number of rows in the triangle.
*/
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle)
    {
        auto&t = triangle;
        
        vector<int> dp(t.back());
        
        int levels = t.size();
        
        //starting from level -2 to up.
        for(int i = levels-2; i>=0; --i)
        {
            for(int j = 0; j<=i; ++j)
            {
                dp[j] = min(dp[i], dp[j+1]) + t[i][j];
            }
        }
        
        return dp[0];
    }
};

//<--> 121. Best Time to Buy and Sell Stock
/*
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction

(ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
Example 2:
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.
*/
class Solution {
public:
    int maxProfit(vector<int>& prices)
    {
        if(prices.empty())
        {
            return 0;
        }
     
        auto min_val = prices[0];
        
        int len = prices.size();
        
        int max_p = 0;
        
        for(int i = 1; i<len; ++i)
        {
            int cur_p = prices[i] - min_val;
            if( cur_p > max_p )
            {
                max_p = cur_p;
            }
            
            min_val = min(prices[i], min_val);
        }
        
        return max_p;
    }
};

//<--> 122. Best Time to Buy and Sell Stock II
/*
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in
multiple transactions at the same time (ie, you must sell the stock before you buy again).
*/

class Solution {
public:
    int maxProfit(vector<int>& prices)
    {    
        if(prices.empty())
        {
            return 0;
        }
        
        int len = prices.size();
        int max_p = 0;
        
        for(int i = 1; i<len; ++i)
        {
            if(prices[i] > prices[i-1])
            {
                max_p += (prices[i] - prices[i-1]);
            }
        }
        
        return max_p;
        
    }
};

//<--> 123. Best Time to Buy and Sell Stock III
/*
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note:
You may not engage in multiple transactions at the same time
(ie, you must sell the stock before you buy again).
*/
class Solution {
public:
    int maxProfit(vector<int>& prices)
    {
        auto& p = prices; //simpily coding
        
        int len  = p.size();
        
        vector<int> profits(len, 0);
        
        int max_prc_so_far = p.back();
        
        for(int i = len-2; i>=0; --i)
        {
            max_prc_so_far = max(max_prc_so_far, p[i]);
            profits[i] = max(profits[i+1], max_prc_so_far - p[i]);
        }
        
        int min_prc_so_far = p[0];
        
        for(int i = 1; i<len; ++i)
        {
            min_prc_so_far = min(min_prc_so_far, p[i]);
            profits[i] = max(profits[i-1], p[i] - min_prc_so_far + profits[i]);
        }
        
        return profits.back();
    }
};

//<--> 124. Binary Tree Maximum Path Sum
/*
Given a binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes
from some starting node to any node in the tree along the parent-child connections.
The path must contain at least one node and does not need to go through the root.

For example:
Given the below binary tree,

       1
      / \
     2   3
Return 6.
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxPathSum(TreeNode* root)
    {
        if(!root)
        {
            return 0;
        }
        
        int max_sum = INT_MIN;
        get_max_sum(root, max_sum);
        
        return max_sum;
    }
    
    int get_max_sum(TreeNode* root, int& mx)
    {
        if(!root)
        {
            return 0;
        }
        
        int l = get_max_sum(root->left, mx);
        int r = get_max_sum(root->right, mx);
        
        int max_single_sum = max(root->val, max(l,r)+root->val);
        int cur_max_sum = max(root->val+l+r, max_single_sum);
        mx = max(cur_max_sum, mx);
        
        return max_single_sum;
    }
};

//<--> 125. Valid Palindrome
/*
Given a string, determine if it is a palindrome,
considering only alphanumeric characters and ignoring cases.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.

Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.

For the purpose of this problem, we define empty string as valid palindrome.
*/
class Solution {
public:
    bool isPalindrome(string s)
    {
        //while(left<right)
        //{
        //    if(s[left] is not alpha)
        //    {
        //        ++left;
        //        continue;
        //    }
        //    
        //    if(s[right] is not alpha)
        //    {
        //        --right;
        //        continue;
        //    }
        //    
        //    if(s[left] is not equal to s[right] without case sensitive)
        //    {
        //        return false;
        //    }
        //    
        //    ++left;
        //    --right;
        //}
    }
};

//<--> 127. Word Ladder
/*
Given two words (beginWord and endWord), and a dictionary's word list,
find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
For example,

Given:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]
As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
*/
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList)
    {
        queue<string> q;   
        q.push(beginWord);
        
        unordered_set<string> s(wordList.begin(), wordList.end());
        unordered_map<string, int> m;
        
        m[beginWord] = 1;

        int word_len = beginWord.size();
                
        //bfs，不用循环。
        while(!q.empty())
        {
            auto word = q.front();
            q.pop();
            
            if(word==endWord)
            {
                return m[word];
            }
            
            int ladders = m[word];
            
            for(int i = 0; i<word_len; ++i)
            {
                auto ch = word[i]; //key: save current character;
                
                for(char c = 'a'; c<='z'; ++c)
                {
                    word[i] = c;
                    
                    if(s.count(word) == 1 && m.find(word) == m.end())
                    {
                        q.push(word);
                        m[word] = ladders + 1;
                    }
                }
                
                word[i] = ch; //key: restore current string
            }
        }
        
        return 0;
    }
};

//<--> 128. Longest Consecutive Sequence
/*
Given an unsorted array of integers,
find the length of the longest consecutive elements sequence.

For example,
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

Your algorithm should run in O(n) complexity.
*/
class Solution {
public:
    int longestConsecutive(vector<int>& nums)
    {
        unordered_map<int, int> m;
        
        int n = nums.size();
        
        int max_len = 0;
        
        for( auto d : nums)
        {
            if(m.find(d)!=m.end())
            {
                continue;
            }
            
            int left = m.find(d-1) == m.end() ? 0 : m[d-1];
            int right = m.find(d+1) == m.end() ? 0 : m[d+1];
            
            int len = left + right + 1;
            m[d] = len;
            m[d-left] = len;
            m[d+right] = len;
            
            max_len = max(max_len, len);
        }
        
        return max_len;
        
    }
};

//<--> 129. Sum Root to Leaf Numbers
/*
Given a binary tree containing digits from 0-9 only,
each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

For example,

    1
   / \
  2   3
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.

Return the sum = 12 + 13 = 25.
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int sumNumbers(TreeNode* root)
    {
        return get_sum(root, 0);
    }
    
    int get_sum(TreeNode* root, int sum)
    {
        if(!root)
        {
            return 0;
        }
        
        sum = sum *10 +root->val;
        
        if(!root->left && !root->right)
        {
            return sum;
        }
        
        return get_sum(root->left, sum) + get_sum(root->right, sum);
    }
};

//<--> 130. Surrounded Regions
/*
Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

For example,
X X X X
X O O X
X X O X
X O X X
After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X
*/
class Solution {
public:
    void solve(vector<vector<char>>& board)
    {    
    }
};

//131. Palindrome Partitioning
/*
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

For example, given s = "aab",
Return

[
  ["aa","b"],
  ["a","a","b"]
]
*/
class Solution {
public:
    vector<vector<string>> partition(string s)
    {
        
    }
};


//<--> 132. Palindrome Partitioning II
/*
Given a string s, partition s such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of s.

For example, given s = "aab",

Return 1 since the palindrome partitioning ["aa","b"]

could be produced using 1 cut.
*/
class Solution {
public:
    int minCut(string s)
    {
    		if ( s.empty() )
			{
				return 0;
			}

			int len = s.size();

			vector<vector<int>> p( len, vector<int>( len, 0 ) );

			for ( int i = 0; i < len; ++i )
			{
				p[i][i] = 1;
			}

			for ( int i = 0; i < len - 1; ++i )
			{
				p[i][i + 1] = ( s[i] == s[i + 1] ) ? 1 : 0;
			}

			for ( int l = 3; l <= len; ++l )  //key: length=1,2 already visited. start with 3
			{
				for ( int i = 0; i < len - l + 1; ++i )
				{
					int j = i + l - 1;

					if ( s[i] == s[j] )
					{
						p[i][j] = p[i + 1][j - 1];
					}
				}
			}


			vector<int> c( len, 0 );

			for ( int i = 1; i < len; ++i )
			{
				if ( p[0][i] == 1 )
				{
					c[i] = 0;
				}
				else
				{
					c[i] = len;

					for ( int j = 0; j < i; ++j )
					{
						if ( p[j + 1][i] == 1 )
						{
							c[i] = min( c[i], 1 + c[j] );
						}
					}
				}
			}

			return c[len - 1];    
    }
};

//<--> 133. Clone Graph
/*
Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.


OJ's undirected graph serialization:

Nodes are labeled uniquely.

We use # as a separator for each node, and ,
as a separator for node label and each neighbor of the node.
As an example, consider the serialized graph {0,1,2#1,2#2,2}.

The graph has a total of three nodes, and therefore contains three parts as separated by #.

First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
Second node is labeled as 1. Connect node 1 to node 2.
Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
Visually, the graph looks like the following:

       1
      / \
     /   \
    0 --- 2
         / \
         \_/
*/
/**
 * Definition for undirected graph.
 * struct UndirectedGraphNode {
 *     int label;
 *     vector<UndirectedGraphNode *> neighbors;
 *     UndirectedGraphNode(int x) : label(x) {};
 * };
 */
class Solution {
public:
    UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node)
    {
    }
};

//<--> 134. Gas Station
/*
There are N gas stations along a circular route, where the amount of gas at station i is gas[i].

You have a car with an unlimited gas tank and

it costs cost[i] of gas to travel from station i to its next station (i+1).

You begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.

Note:
The solution is guaranteed to be unique.
*/
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        
    }
};

//<--> 135. Candy
/*
There are N children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?
*/
class Solution {
public:
    int candy(vector<int>& ratings)
    {      
    }
};

//<--> 136. Single Number
/*
Given an array of integers, every element appears twice except for one. Find that single one.

Note:
Your algorithm should have a linear runtime complexity.
Could you implement it without using extra memory?
*/
class Solution {
public:
    int singleNumber(vector<int>& nums)
    {    
    }
};

//<--> 137. Single Number II
/*
Given an array of integers, every element appears three times except for one,
which appears exactly once. Find that single one.

Note:
Your algorithm should have a linear runtime complexity.
Could you implement it without using extra memory?
*/
class Solution {
public:
    int singleNumber(vector<int>& nums)
    {    
    }
};

//<--> 138. Copy List with Random Pointer
/*
A linked list is given such that

each node contains an additional random pointer

which could point to any node in the list or null.

Return a deep copy of the list.
*/
/**
 * Definition for singly-linked list with a random pointer.
 * struct RandomListNode {
 *     int label;
 *     RandomListNode *next, *random;
 *     RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
 * };
 */
class Solution {
public:
	//method 1: using map 
    RandomListNode *copyRandomList(RandomListNode *head)
    {
		if(!head)
		{
			return nullptr;
		}
		
		RandomListNode* clone_head = new RandomListNode(head->label);
		auto cur = head->next;
		auto clone = clone_head;
		
		unordered_map<RandomListNode*, RandomListNode*> m;
		m[head] = clone_head;
		
		
		while(cur)
		{
			auto tmp = new RandomListNode(cur->label);
			clone->next = tmp;
			m[cur] = tmp;
			
			cur = cur->next;
			clone = clone->next;
		}
		
		cur = head;
		clone = clone_head;
		
		while(cur)
		{
			clone->random = m[cur->random];
			cur = cur->next;
			clone = clone->next;
		}
		
		
		return clone_head;
	}
	
	//method 2: add the clone next to current
	RandomListNode *copyRandomList(RandomListNode *head)
	{
		if(!head)
		{
			return nullptr;
		}
		
		auto cur = head;
		
		while(cur)
		{
			auto tmp = new RandomListNode(cur->label);
			tmp->next = cur->next;
			cur->next = tmp;
			
			cur = tmp->next;
		}
		
		cur = head;
		
		while(cur)
		{
			if(cur->random)
			{
				cur->next->random = cur->random->next;
			}
			
			cur = cur->next->next;
		}
		
		cur = head;
		auto new_head = cur->next;
		auto clone = new_head;
		
		while(clone)
		{
			auto tmp = cur->next;
			cur->next = tmp->next;
			if(tmp->next)
			{
				tmp->next = tmp->next->next;
			}
			cur = cur->next;
		}
		
		return new_head;
	}
};

//<--> 139. Word Break
/*
Given a non-empty string s and a dictionary

wordDict containing a list of non-empty words,

determine if s can be segmented into a space-separated sequence of one or more dictionary words.

You may assume the dictionary does not contain duplicate words.

For example, given
s = "leetcode",
dict = ["leet", "code"].

Return true because "leetcode" can be segmented as "leet code".
*/
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict)
    {
		if ( s.empty() )
		{
			return false;
		}

		int len = s.size();

		unordered_set<string> word_s( wordDict.begin(), end( wordDict ) );

		vector<int> dp( len + 1, 0 );
		dp[0] = 1;

		for ( int i = 1; i <= len; ++i )
		{
			for ( int j = 0; j < i; ++j )
			{
				if ( dp[j] == 1 )
				{
					if ( word_s.find( s.substr( j, i - j ) ) != word_s.end() )
					{
						dp[i] = 1;
						break;
					}
				}
			}
		}

		return dp.back() == 1;
    }
};

//<--> 140. Word Break II
/*
Given a non-empty string s and a dictionary wordDict 
containing a list of non-empty words, 
add spaces in s to construct a sentence 
where each word is a valid dictionary word. 
You may assume the dictionary does not contain duplicate words.

Return all such possible sentences.

For example, given
s = "catsanddog",
dict = ["cat", "cats", "and", "sand", "dog"].

A solution is ["cats and dog", "cat sand dog"].
*/
class Solution {
public:
    vector<string> wordBreak(string s, vector<string>& wordDict) 
	{
		vector<string> res;
		string out;
		
		vector<int> possible(s.size()+1, 1);
		
		unordered_set<string> ws(wordDict.begin(), wordDict.end());
		
		dfs(s,ws,possible,out,res,0);
		
		return res;
	}
	
	void dfs(const string &s, const unordered_set<string>& ws, vector<int>& possible, string& out, vector<string>& res, int start)
	{
		if(start == s.size())
		{
			res.emplace_back(out.substr(0, out.size()-1).c_str()); //key: remove the final blank from out.
			return;
		}
		
		for(int i = start; i<s.size(); ++i)
		{
			auto word = s.substr(start, i-start+1);
			if( ( ws.find(word)!=ws.end() ) && ( possible[i+1] == 1 ) )
			{
				out += word;
				out += " ";
				
				int old_size = res.size();
				
				dfs(s,ws,possible,out,res,i+1);
				
				if(old_size == res.size())
				{
					possible[i+1] = 0;
				}
				
				out.resize(out.size() - word.size() - 1);
			}			
		}
	}
};

//<--> 141. Linked List Cycle
/*
Given a linked list, determine if it has a cycle in it.
https://codesays.com/2014/solution-to-fish-by-codility/
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
 
class Solution {
public:
    bool hasCycle(ListNode *head) {
        
    }
};

//<--> 142. Linked List Cycle II
/*
Given a linked list, return the node where the cycle begins. 

If there is no cycle, return null.

Note: Do not modify the linked list.

Follow up:
Can you solve it without using extra space?
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *detectCycle(ListNode *head) 
	{
		if(!head || !head->next)
		{
			return nullptr;
		}
		
		auto slow = head;
		auto fast = head;
		
		while( fast && fast->next )
		{
			slow = slow->next;
			fast = fast->next->next;
			
			if(slow==fast)
			{
				break;
			}
		}
		
		if(!fast || !fast->next)
		{
			return nullptr;
		}
		
		start = head;
		
		while(slow!=fast)
		{
			slow = slow->next;
			fast = fast->next;
		}
		
		return fast;
    }
};

//<--> 143. Reorder List
/*
Given a singly linked list L: L_0→L_1→…→L_{n-1}→L_n,
reorder it to: L_0→L_n→L_1→L_{n-1}→L_2→L_{n-2}→…

You must do this in-place without altering the nodes' values.

For example,
Given {1,2,3,4}, reorder it to {1,4,2,3}.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void reorderList(ListNode* head) 
	{    
    }
};

//<--> 144. Binary Tree Preorder Traversal
/*
Given a binary tree, return the preorder traversal of its nodes' values.

For example:
Given binary tree {1,#,2,3},
   1
    \
     2
    /
   3
return [1,2,3].

Note: Recursive solution is trivial, could you do it iteratively?
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) 
	{    
    }
};

//<--> 145. Binary Tree Postorder Traversal
/*
Given a binary tree, return the postorder traversal of its nodes' values.

For example:
Given binary tree {1,#,2,3},
   1
    \
     2
    /
   3
return [3,2,1].

Note: Recursive solution is trivial, could you do it iteratively?
*/
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) 
	{
		vector<int> res;
		post_order(root,res);
    }
	
	void post_order(TreeNode* root, vector<int>& out)
	{
		if(!root)
		{
			return;
		}
		
		stack<TreeNode*> s;
		s.push(root);
		
		while(!s.empty())
		{
			auto t = s.top();
			s.pop();
			
			out.insert(out.begin(), t->val);
			
			if(t->left)
			{
				s.push(t->left);
			}
			
			if(t->right)
			{
				s.push(t->right);
			}
		}
	}
};

//<--> 146. LRU Cache
/*
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in O(1) time complexity?

Example:

LRUCache cache = new LRUCache( 2 //  );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
*/
class LRUCache {
public:
    LRUCache(int capacity) {
        
		max_c = capacity;
    }
    
    int get(int key) 
	{
		auto it = m.find(key);
		
		if(it==m.end())
		{
			return -1;
		}
		
		cache.splice(cache.begin(), cache, it->second); //move the current iterator into the beginning of the list.
		return it->second->second; //key: it->second return the iterator in the list, and it->second->second returns the pair's second
    }
    
    void put(int key, int value) 
	{
		auto it = m.find(key);
		if(it!=m.end())
		{
			cache.erase(it); //key: no need to remove from map since we will update m[key] later
		}
		
		cache.emplace_front(key,value);
		
		m[key] = cache.begin(); //key: we need to put new element into the first positon of the list;
		
		if(m.size()>max_c)
		{
			int remove_key = cache.back().first;
			m.erase(remove_key);
			cache.pop_back();
		}
    }
	
	private:
		
		int max_c;
		
		list<pair<int,int>> cache;
		
		unordered_map<int, list<pair<int,int>>::iterator> m;
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */

//<--> 147. Insertion Sort List
/*
Sort a linked list using insertion sort.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* insertionSortList(ListNode* head) 
	{
		auto dummy = new ListNode(-1);
		
		auto cur = head;
		
		while(cur)
		{
			auto next = cur->next;
			auto new_cur = dummy;
			
			while(new_cur->next && new_cur->next->val < cur->val)
			{
				new_cur = new_cur->next;
			}
			
			cur->next = new_cur->next;
			new_cur->next = cur;
			cur = next;
		}
		
		return dummy->next;
    }
};

//<--> 148. Sort List
/*
Sort a linked list in O(n log n) time using constant space complexity.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        
		if(!head || !head->next)
		{
			return head;
		}
		
		auto slow = head, fast = head;
		ListNode* pre = nullptr;
		
		while(fast && fast->next)
		{
			pre = slow;
			slow = slow->next;
			fast = fast->next->next;
		}
		
		pre->next = nullptr;
		
		return merge(sortList(head), sortList(slow));
		
    }
	
	ListNode* merge(ListNode* l1, ListNode* l2)
	{
		auto dummy = new ListNode(-1);
		auto cur = dummy;
		
		while(l1&&l2)
		{
			if(l1->val < l2->val)
			{
				cur->next = l1;
				l1 = l1->next;
			}
			else
			{
				cur->next = l2;
				l2 = l2->next;
			}
			
			cur = cur->next;
		}
		
		if(l1) cur->next = l1;
		if(l2) cur->next = l2;
		
		return dummy->next;
	}
	
	//recursive merge
	ListNode* merge(ListNode* l1, ListNode* l2)
	{
		if(!l1) return l2;
		if(!l2) return l2;
		
		if(l1->val < l2->val)
		{
			l1->next = merge(l1->next, l2);
			return l1;
		}
		else
		{
			l2->next = merge(l1, l2->next);
			return l2;
		}
		
	}
};
 
//<--> 151. Reverse Words in a String
/*
Given an input string, reverse the string word by word.

For example,
Given s = "the sky is blue",
return "blue is sky the"
*/
class Solution {
public:
    void reverseWords(string &s)
    {
        if(s.empty())
        {
            return;
        }
        
        reverse(s.begin(), s.end());
        
        int store_pos = 0;
        
        for(size_t i = 0; i< s.size(); ++i)
        {
            if(s[i]!=' ')
            {
                if(store_pos!=0)
                {
                    s[store_pos++] = ' ';
                }
                
                size_t j = i;
                while( j < s.size() && s[j] != ' ' )
                {
                    s[store_pos++] = s[j++];
                }
                
                reverse(s.begin()+store_pos-(j-i), s.begin()+store_pos); //key: start position = store_pos - (j-i)
                
                i = j;
            }
        }
        
        s.resize(store_pos);
    }
};

//<--> 152. Maximum Product Subarray
/*
Find the contiguous subarray within an array
(containing at least one number) which has the largest product.

For example, given the array [2,3,-2,4],
the contiguous subarray [2,3] has the largest product = 6.
*/
class Solution {
public:
    int maxProduct(vector<int>& nums)
    {
        if(nums.size() ==  1)
        {
            return nums[0];
        }
        
        int max_so_far = 1, min_so_far = 1; //important: need to set both to 1
        int old_max_so_far = -1;
        
        int res = 0;
        
        for(size_t i = 0; i<nums.size(); ++i)
        {
            old_max_so_far = max(max_so_far, 1);
            
            if(nums[i] > 0)
            {
                max_so_far = old_max_so_far * nums[i];
                min_so_far = min_so_far * nums[i];
            }
            else //covery 0 and negative
            {
                max_so_far = min_so_far * nums[i];
                min_so_far = old_max_so_far * nums[i];
            }
            
            res = max(res, max_so_far);
        }
        
        return res;
    }
};

//<--> 153. Find Minimum in Rotated Sorted Array
/*
Suppose an array sorted in ascending order is rotated

at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.

You may assume no duplicate exists in the array.
*/

/*
 (1) A[mid] < A[end]： A[mid : end] sorted => min不在A[mid+1 : end]中
搜索A[start : mid]

(2) A[mid] > A[end]：A[start : mid] sorted且又因为该情况下A[end]<A[start] => min不在A[start : mid]中
搜索A[mid+1 : end]
 */
class Solution {
public:
    int findMin(vector<int>& nums)
    {
        int left = 0, right = nums.size() - 1;
        
        while(left < right) // key: using "<" not "<=": if there is only one element, left will be set to 1
        {
            if(nums[mid]<nums[right]) // right half is sorted, min element cannot be in [mid+1, right]
            {
                right = mid; 
            }
            else
            {
                left = mid + 1; //left half is sorted, min element cannot be in [left, mid]
            }
        }
        
        return nums[left];
    }
};

//<--> 154. Find Minimum in Rotated Sorted Array II
/*
Follow up for "Find Minimum in Rotated Sorted Array":
What if duplicates are allowed?

Would this affect the run-time complexity? How and why?

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.

The array may contain duplicates.
*/
class Solution {
public:
    int findMin(vector<int>& nums)
    {
        int left = 0, right = nums.size() - 1;
        
        while(left < right) // key: using "<" not "<=": if there is only one element, left will be set to 1
        {
            int mid = left + (right-left)/2;
            
            if(nums[mid]<nums[right]) // right half is sorted, min element cannot be in [mid+1, right]
            {
                right = mid; 
            }
            else if(nums[mid]>nums[right]) //left half is sorted, min element cannot be in [left, mid]
            {
                left = mid + 1; 
            }
            else
            {
                --right;
            }
        }
        
        return nums[left];
    }
};

//<--> 155. Min Stack
/*
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
Example:
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
*/

lass MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
        min_elem = INT_MAX;
    }
    
    void push(int x) {
        //important: we must use x<=min_elem not x<min_elem
        //because if we use x<min_elem, when x=min_elem, we only push once, but pop will be taken twice.
        if(x <= min_elem)
        {
            s.push(min_elem);           
            min_elem = x;
        }
        
        s.push(x);
    }
    
    void pop()
    {
        auto t = s.top();
        s.pop();
        
        if(t == min_elem)
        {
            //important: since for each new minimal element, we push two elements into the stack
            //therefore, we need to pop twice.
            min_elem = s.top();
            s.pop();
        }
    }
    
    int top() {
        
        return s.top();
    }
    
    int getMin() {
        
        return min_elem;
    }
    
    private:
        int min_elem;
        stack<int> s;
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */

//<--> 156. Binary Tree Upside Down
/*
Given a binary tree where all the right nodes are either leaf nodes with a sibling
(a left node that shares the same parent node) or empty,
flip it upside down and turn it into a tree where the original right nodes
turned into left leaf nodes. Return the new root.

For example:

Given a binary tree {1,2,3,4,5},

    1

   / \

  2   3

 / \

4   5

return the root of the binary tree [4,5,2,#,#,3,1].

   4

  / \

 5   2

    / \

   3   1 
*/

class Solution {
public:
    //method 1: recursive
    TreeNode *upsideDownBinaryTree(TreeNode *root)
    {
        if(!root || !root->left)
        {
            return root;
        }
        
        auto l = root->left;
        auto r = root->right;
        
        auto res = upsideDownBinaryTree(l);
        
        l->left = r;
        l->right = root;
        
        root->left = nullptr;
        root->right = nullptr;
        
        return res;
    }
    
    //method 1: iterative
    TreeNode *upsideDownBinaryTree(TreeNode *root)
    {
        auto cur = root;
        TreeNode* pre = nullptr;
        TreeNode* next = nullptr;
        TreeNode* tmp = nullptr;
        
        while(cur)
        {
            next = cur->left;
            cur->left = tmp;
            
            tmp = cur->right;
            cur->right = pre;
            
            pre = cur;
            cur = next;
        }
        
        return pre;
    }
};

//<--> 157. Read N Characters Given Read4
/*
The API: int read4(char *buf) reads 4 characters at a time from a file.
The return value is the actual number of characters read.

For example, it returns 3 if there is only 3 characters left in the file.

By using the read4 API, implement the function

int read(char *buf, int n)

that reads n characters from the file.

Note:
The read function will only be called once for each test case. 
*/

// Forward declaration of the read4 API.
int read4(char *buf);

class Solution {
public:
    
    //iterative method:
    int read(char *buf, int n)
    {
        int total_chars = 0;
        
        for(int i = 0; i<=n/4; ++i)
        {
            auto num_chars = read4(buf+total_chars);
            if(num_chars==0)
            {
                break;
            }
            
            total_chars += num_chars;
        }
        
        return min(total_chars, n);
    }
};

//<--> 158. Read N Characters Given Read4 II - Call multiple times
/*
The API: int read4(char *buf) reads 4 characters at a time from a file.
The return value is the actual number of characters read.
For example, it returns 3 if there is only 3 characters left in the file.

By using the read4 API,
implement the function

int read(char *buf, int n)

that reads n characters from the file.

Note:
The read function may be called multiple times.
*/
class Solution {
public:
    int read(char *buf, int n)
    {
        int read_pos = 0, write_pos = 0;
        char buffer[4];
        
        for(int i = 0; i<n; ++i)
        {
            if(read_pos == write_pos)
            {
                write_pos = read4(buffer);
                read_pos = 0;
                if(write_pos==0)
                {
                    return i;
                }
            }
            buf[i++] = buffer[read_pos++];
        }
        
        return n;
    }
};


//<--> 159. Longest Substring with At Most Two Distinct Characters
/*
Given a string S, find the length of the longest substring T
that contains at most two distinct characters.
For example,
Given S = “eceba”,
T is “ece” which its length is 3.
*/
class Solution {
public:
    //method 1: using count as map value
    int lengthOfLongestSubstringTwoDistinct(string s)
    {
        unordered_map<char,int> m;
        int left = 0; max_len = 0;
        
        int len = s.size();
        
        for(int i = 0; i<len; ++i)
        {
            if(m.find(s[i]) == m.end())
            {
                m[s[i]] = 1;
            }
            else
            {
                ++m[s[i]];
            }
            
            while(m.size()>2)
            {
                auto& v = m[s[left]];
                --v;
                if(v == 0)
                {
                    m.erase(s[left]);
                }
                ++left;
            }
            
            max_len = max(max_len, i - left + 1);
        }
        
        return max_len;
    }
    
    //method 2: using pos as map value
    int lengthOfLongestSubstringTwoDistinct(string s)
    {
        unordered_map<char,int> m;
        int left = 0; max_len = 0;
        
        int len = s.size();
        
        for(int i = 0; i<len; ++i)
        {
            m[s[i]] = i;
            
            while(m.size()>2)
            {
                if( m[s[left]] == left )
                {
                    m.erase(s[left]);
                }
                ++left;
            }
            
            max_len = max(max_len, i - left + 1);
        }
        
        return max_len;
    }
    
    //method 3: using O(1) memory but only for 2 duplicate characters.
    int lengthOfLongestSubstringTwoDistinct(string s)
    {
        int left = 0; max_len = 0;
        int len = s.size();
        int right = -1;
        
        for(int i = 1; i<len; ++i)
        {
            if(s[i]==s[i-1])
            {
                ++i;
            }
            
            if(right >=0 && s[right] != s[i])
            {
                max_len = max(max_len, i - left);
                left = right + 1;
            }
            
            right = i - 1;
        }
        
        return max(max_len, len - left);
    }
};

//<--> 160. Intersection of Two Linked Lists
/*
Write a program to find the node at which the intersection of two singly linked lists begins.


For example, the following two linked lists:

A:          a1 → a2 
                   
                     c1 → c2 → c3
                               
B:     b1 → b2 → b3
begin to intersect at node c1.


Notes:

If the two linked lists have no intersection at all, return null.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in O(n) time and use only O(1) memory.
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB)
    {
        if(!headA || !headB)
        {
            return nullptr;
        }
        
        auto a = headA;
        auto b = headB;
        
        while(a!=b)
        {
            a = a ? a->next : headB;
            b = b ? b->next : headA;
        }
        
        return a;
    }
};

// <--> 161. One Edit Distance
/*
Given two strings S and T, determine if they are both one edit distance apart.
*/
class Solution {
public:
    bool isOneEditDistance(string s, string t)
    {
        if(s.size()< t.size())
        {
            swap(s,t);
        }
        
        int diff = s.size() - t.size();
        
        if(diff>1)
        {
            return false;
        }
        
        if(diff==1)
        {
            for(size_t i = 0; i < t.size(); ++i)
            {
                if(s[i]!=t[i])
                {
                    return s.substr(i+1) == t.substr(i)
                }
            }
            
            return true;
        }
        
        int diff_count = 0;
        
        for(size_t i = 0; i<t.size(); ++i)
        {
            if(s[i]!=t[i])
            {
                ++diff_count;
            }
        }
        
        return diff_count == 1;
    }
};

//<--> 162. Find Peak Element
/*
A peak element is an element that is greater than its neighbors.

Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that num[-1] = num[n] = -INF.

For example, in array [1, 2, 3, 1], 3 is a peak element
and your function should return the index number 2.

Note:
Your solution should be in logarithmic complexity.
*/

class Solution {
public:
    int findPeakElement(vector<int>& nums)
    {
        int left = 0, right = nums.size() - 1;
        
        while(left < right)
        {
            int mid = left + (right - left)/2;
            if(nums[mid] < nums[mid+1]) //key: peak must be in the right hand side
            {
                left = mid+1;
            }
            else  //key: peak must be in the left hand side
            {
                right = mid;
            }
        }
        
        return right;
    }
};

//<--> 163. Missing Ranges
/*
Given a sorted integer array where the range of elements are [0, 99]
inclusive, return its missing ranges.
For example, given [0, 1, 3, 50, 75],
return [“2”, “4->49”, “51->74”, “76->99”]
*/

class Solution {
public:
    vector<string> findMissingRanges(vector<int>& nums, int lower, int upper)
    {
        int len = nums.size();
        int l = lower, r = 0;
        
        vector<string> res;
        
        for(int i = 0; i<=len; ++i)
        {
            r = (i<len && nums[i]<=upper) ? nums[i] : upper+1;
            
            if(l==r)
            {
                ++l;
            }
            else if(r > l)
            {
                if( r- l == 1)
                {
                    res.push_back(to_string(l));
                }
                else
                {
                    res.push_back(to_string(l) + "->" + to_string(r-1));
                    l = r+1;
                }
            }
        }
        
        return res;
    }
};

//<--> 164. Maximum Gap
/*
Given an unsorted array, find the maximum difference

between the successive elements in its sorted form.

Try to solve it in linear time/space.

Return 0 if the array contains less than 2 elements.

You may assume all elements in the array are non-negative integers

and fit in the 32-bit signed integer range.
*/
class Solution {
public:
    int maximumGap(vector<int>& A)
    {
        if(A.empty())
        {
            return 0;
        }
        
        int len = A.size();
        
        auto iter_pair = minmax_element(A.begin(), A.end());
        
        auto max_v = *(iter_pair.second);
        auto min_v = *(iter_pair.first);
        
        int bck_size = (max_v - min_v) / len + 1;
        int num_bcks = (max_v - min_v) / bck_size + 1;
        
        vector<int> bck_min(num_bcks, INT_MAX);
        vector<int> bck_max(num_bcks, INT_MIN);
        
        unordered_set<int> s;
        
        for( int d: A )
        {
            int bck_idx = (d-min_v) / bck_size;
            
            bck_min[bck_idx] = min(bck_min[bck_idx], d);
            bck_max[bck_idx] = max(bck_max[bck_idx], d);
            
            s.insert(bck_idx);
        }
        
        int pre = 0, max_gap = 0;
        
        for(int i = 1; i<n; ++i)
        {
            if(s.count(i) == 0)
            {
                continue;
            }
            
            max_gap = max(max_gap, ( bck_min[i] - bck_max[pre] ));
        }
        
        return max_gap;
    }
};

//<--> 165. Compare Version Numbers
/*
Compare two version numbers version1 and version2.

If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.
The . character does not represent a decimal point and is used to separate number sequences.
For instance, 2.5 is not "two and a half" or "half way to version three",
it is the fifth second-level revision of the second first-level revision.

Here is an example of version numbers ordering:

0.1 < 1.1 < 1.2 < 13.37
*/
class Solution {
public:
    int compareVersion( string version1, string version2 )
    {
        istringstream ssv1(version1 +".");
        istringstream ssv2(version2 +".");
        
        int d1 = 0, d2 = 0;
        char dot = '.';
        
        while(ssv1.good() || ssv2.good())
        {
            if(ssv1.good())
            {
                ssv1 >> d1 >> dot;
            }
            
            if(ssv2.good())
            {
                ssv2>>d2>>dot;
            }
            
            if(d1 > d2) return 1;
            if(d1 < d2) return -1;
            
            d1 = 0;
            d2 = 0;
        }
        
        return 0;
    }
};


//<--> 166. Fraction to Recurring Decimal
/*
Given two integers representing the numerator

and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

For example,

Given numerator = 1, denominator = 2, return "0.5".
Given numerator = 2, denominator = 1, return "2".
Given numerator = 2, denominator = 3, return "0.(6)".
*/
class Solution {
public:
    string fractionToDecimal(int numerator, int denominator)
    {
        if ( denominator == 0 )
        {
            return "";
        }

        if ( numerator == 0 )
        {
            return "0";
        }

        int sign = numerator > 0 ? 1 : -1;
        sign = denominator > 0 ? sign : sign * -1;

        long long num = abs( (long long)numerator );
        long long den = abs( (long long)denominator );

        

        auto q = num / den;
        auto r = num - q * den;

        string prefix;

        if ( sign == -1 )
        {
            prefix += "-";
        }

        prefix += to_string( q );
        if ( r == 0 )
        {
            return prefix;
        }

        prefix += ".";

        unordered_map<long long, int> m;

        int pos = 0;

        string postfix;

        while ( r != 0 )
        {
            if ( m.find( r ) != m.end() )
            {
                postfix.insert( m[r], "(" );
                postfix += ")";

                return prefix + postfix;
            }

            m[r] = pos;

            q = (r * 10) / den;
            r = r * 10 - q*den;

            postfix += to_string( q );
            ++pos;
        }

        return prefix + postfix;
    }
};

//<--> /*so easy*/ 167. Two Sum II - Input array is sorted
/*
Given an array of integers that is already sorted
in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers
such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2
*/


//<--> 168. Excel Sheet Column Title
/*
Given a positive integer,

return its corresponding column title as appear in an Excel sheet.

For example:

    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
*/

class Solution {
public:
    string convertToTitle(int n)
    {
       string res;
       while(n != 0)
       {
         res += (--n) % 26 + 'A';
         n /= 26;
       }
       
       return string(r.rbegin(), r.rend());
    }
};

// 169. Majority Element
/*
Given an array of size n, find the majority element.
The majority element is the element that appears more than n/2 times.

You may assume that the array is non-empty and the majority element

always exist in the array.
*/
class Solution {
public:
    //method 1: Moore Voting method
    int majorityElement(vector<int>& nums)
    {
        int elem = 0;
        int count = 0;
        
        for(int d: nums)
        {
            if(count == 0)
            {
                elem = d;
                ++count;
            }
            else
            {
                if(elem == d)
                {
                    ++count;
                }
                else
                {
                    --count;
                }
            }
        }
        
        return elem;
    }
    
    //method 2: bit operation
    int majorityElement(vector<int>& nums)
    {
        int ones = 0;
        int zeros = 0;
        int elem = 0;
        
        int mask = 0;
        
        for(int i = 0; i<32; ++i)
        {
            mask = 1<<i;
            
            ones = 0;
            zeros = 0;
            
            for(int d : nums)
            {
                if(d & mask)
                {
                    ++ones;
                }
                else
                {
                    ++zeros;
                }
            }
            
            if(ones > zeros) elem |= mask;
        }
        
        return elem;
    }
};


//<--> 170. Two Sum III - Data structure design
/*
Design and implement a TwoSum class.
It should support the following operations:add and find.

add - Add the number to an internal data structure.
find - Find if there exists any pair of numbers which sum is equal to the value.

For example,
add(1); add(3); add(5);
find(4) -> true
find(7) -> false
*/

class TwoSum
{
    public:
        void add(int number)
        {
            if(m.find(number) == m.end())
            {
                m[number] = 1;
            }
            else
            {
               ++m[number];
            }
        }
        
        bool find(int value)
        {
            for( const auto &elem : m)
            {
                int diff = value - elem.first;
                
                if( ( elem.first != diff ) && (m.find(diff) != m.end()) )
                {
                    return true;
                }
                
                if( ( elem.first == diff ) && ( elem.second > 1 ) )
                {
                    return true;
                }
            }
            
            return false;
        }
        
    private:
        unordered_map<int, int> m;
}
