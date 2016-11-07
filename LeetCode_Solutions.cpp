Problem:

[LeetCode] Convert a Number to Hexadecimal

Given an integer, write an algorithm to convert it to hexadecimal. For negative integer, two’s complement method is used.

Note:

All letters in hexadecimal (a-f) must be in lowercase.
The hexadecimal string must not contain extra leading 0s. If the number is zero, it is represented by a single zero character '0'; otherwise, the first character in the hexadecimal string will not be the zero character.
The given number is guaranteed to fit within the range of a 32-bit signed integer.
You must not use any method provided by the library which converts/formats the number to hex directly.
 

Example 1:

Input:
26

Output:
"1a"
 

Example 2:

Input:
-1

Output:
"ffffffff"

Solution: 

我们采取位操作的思路，每次取出最右边四位，
如果其大于等于10，找到对应的字母加入结果
，反之则将对应的数字加入结果，然后num像右平移四位，
循环停止的条件是num为0，或者是已经循环了7次，参见代码如下：

class Solution {
public:
    string toHex(int num) {
        string res = "";
        for (int i = 0; num && i < 8; ++i) {
            int t = num & 0xf;
            if (t >= 10) res = char('a' + t - 10) + res;
            else res = char('0' + t) + res;
            num >>= 4;
        }
        return res.empty() ? "0" : res;
    }
};

My Solution:

class Solution {
public:
    string toHex(int num) {
  
		//Define a char arry to map the number to hex;
		
        char H[] = "0123456789abcdefgh";
        
        string res;
        
		//the loop will stop if num==0 or i == 8 (most 7 loops)
        for( int i = 0; (num) && (i<8); ++i )
        {
			//get the lowest 4 bits
            int t = num & 0xF;
			//Prepend the value into the result string.
            res = H[t] + res;
			//right shift 4 bits;
            num >>= 4;
        }
        
        return res.size() ? res : "0";
    }
};



MySolution:

class Solution {
public:
    int myAtoi(string str) {
        
        const char* p = str.c_str();
        const auto len = str.size();
        
        size_t start = 0;
        
        while( p[start] == ' ' )
        {
            ++start;
            if (start == len)
            {
                return 0;
            }
        }
        
        if( ( p[start] <'0' || p[start] > '9' ) && (p[start] != '-') && (p[start] != '+') )
        {
            return 0;
        }
        
        
        int m = 1;
        int limit = std::numeric_limits<int>::max();
        if( p[start] == '-' )
        {
            m = -1;
            ++start;
            limit = std::numeric_limits<int>::min();
        }
        else if(p[start]=='+')
        {
            ++start;
        }
        
        int result = 0;
        int count = 0;
        
        while( start < len )
        {
            if( p[start] <'0' || p[start] > '9'  )
            {
                break;
            }
            
            ++count;
            if( count > 10 )
            {
                return limit;
            }
            
            result *= 10;
            result += (p[start] - '0');
            
            if( (result>>31) & 0x1)
            {
                return limit;
            }
            
            ++start;
        }
        
        return result*m;

    }
};

others:

class Solution {
public:
    int myAtoi(string str) {
        if (str.empty()) return 0;
        int sign = 1, base = 0, i = 0, n = str.size();
        while (i < n && str[i] == ' ') ++i;
        if (str[i] == '+' || str[i] == '-') {
            sign = (str[i++] == '+') ? 1 : -1;
        }
        while (i < n && str[i] >= '0' && str[i] <= '9') {
            if (base > INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0' > 7)) {
                return (sign == 1) ? INT_MAX : INT_MIN;
            }
            base = 10 * base + (str[i++] - '0');
        }
        return base * sign;
    }
};


====================
[LeetCode] Palindrome Number 验证回文数字
 

Determine whether an integer is a palindrome. Do this without extra space.

click to show spoilers.

Some hints:
Could negative integers be palindromes? (ie, -1)

If you are thinking of converting the integer to string, note the restriction of using extra space.

You could also try reversing an integer. However, if you have solved the problem "Reverse Integer", you know that the reversed integer might overflow. How would you handle such case?

There is a more generic way of solving this problem.

 这道验证回文数字的题不能使用额外空间，意味着不能把整数变成字符，然后来验证回文字符串。而是直接对整数进行操作，我们可以利用取整和取余来获得我们想要的数字，比如 1221 这个数字，如果 计算 1221 / 1000， 则可得首位1， 如果 1221 % 10， 则可得到末尾1，进行比较，然后把中间的22取出继续比较。代码如下：

class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        int div = 1;
        while (x / div >= 10) div *= 10;
        while (x > 0) {
            int left = x / div;
            int right = x % 10;
            if (left != right) return false;
            x = (x % div) / 10;
            div /= 100;
        }
        return true;
    }
};

MyCode

class Solution {
public:
    bool isPalindrome(int x) {
        
        if( x < 0 )
        return false;
        
        if( x < 10 )
        {
            return true;
        }
        
        int q = 0;
        int r = 0;
        int result = 0;
        int k = x;
        
        int threshold = std::numeric_limits<int>::max()/10;
        
        
        while( k > 0 )
        {
            q = k/10;
            r = k - q*10;
            
			//This will determine if the result is outside of numeric limits.
            if( result > threshold || (result == threshold && (r > 7) ))
            {
                return false;
            }
            
            result = result*10+r;
            k = q;
        }

        return result == x;        

    }
};


=================================
417. Pacific Atlantic Water Flow

Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.


Note:
The order of returned grid coordinates does not matter.
Both m and n are less than 150.
Example:

Given the following 5x5 matrix:

  Pacific ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic

Return:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).

test case:
[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]

class Solution {
public:
    vector<pair<int, int>> pacificAtlantic(vector<vector<int>>& matrix) {
        
    }
};

=============================================
404. Sum of Left Leaves

   3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.

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
    int sumOfLeftLeaves(TreeNode* root) {
        
    }
};
[3,9,20,null,null,15,7]

MySolution:

class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        
        int sum = 0;

        if( root )
        {
            if( isLeave(root->left) )
            {
                sum += root->left->val;
            }
            else
            {
                sum += sumOfLeftLeaves( root->left );
            }
            
            sum += sumOfLeftLeaves( root->right );
        }
        
        return sum;
    }
    
    bool isLeave(TreeNode* node)
    {
        return (node)&&(!node->left)&&(!node->right);
    }
};

Other solutions:

// use queue
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (!root || (!root->left && !root->right)) return 0;
        int res = 0;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            TreeNode *t = q.front(); q.pop();
            if (t->left && !t->left->left && !t->left->right) res += t->left->val;
            if (t->left) q.push(t->left);
            if (t->right) q.push(t->right);
        }
        return res;
    }
};

//use stack
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        if (!root || (!root->left && !root->right)) return 0;
        int res = 0;
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode *t = s.top(); s.pop();
            if (t->left && !t->left->left && !t->left->right) res += t->left->val;
            if (t->left) s.push(t->left);
            if (t->right) s.push(t->right);
        }
        return res;
    }
};

=============================================
// LeetCode 10: Regular Expression Matching
/*
Implement regular expression matching with support for '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "a*") → true
isMatch("aa", ".*") → true
isMatch("ab", ".*") → true
isMatch("aab", "c*a*b") → true
*/
class Solution {
public:
    bool isMatch(string s, string p) {
        
        const auto len_s = s.size();
        const auto len_p = p.size();
        
		//note:
		//if s is empty and p is ".*", should return true;
		//we add a auxilary helper character into s and p, so that the 
		//lables will have (len_s+1)*(len_p+1)
        
        vector<vector<unsigned char>> l(len_s + 1, vector<unsigned char>(len_p + 1));

		//at start, the helper character is equal to the helper character
        l[0][0] = 1; 
        
		//have to deal with something like a*b*c* 
        for( std::size_t i = 1; i<=len_p; ++i )
        {
            if( p[ i - 1 ] == '*')
            {
                l[0][i] = l[0][i - 2];
            }
        }

        for( std::size_t i = 1; i<= len_s; ++i )
        {
            for( std::size_t j = 1; j<= len_p; ++j )
            {
				//if current character in s is equal to current character in p or equal to '.'
                if( s[i-1] == p[j-1] || p[j-1] == '.' )
                {
                   l[i][j] = l[i-1][j-1];
                }
                else if( p[j-1]=='*' )
                {
					//match 0 
                    l[i][j] = l[i][j-2];
                    if( s[i-1] == p[j-2] || p[j-2] == '.' )
                    {
						//match 1
						//note: we need to still consider l[i][j-2] since l[i][j] is determined by both
						//l[i][j-2] and l[i-1][j]
                        l[i][j] = ( l[i][j] + l[i-1][j] ) > 0 ? 1 : 0;
                    }
                }
            }
        }

        return l[len_s][len_p] == 1;
        
    }
};

/*
421. Maximum XOR of Two Numbers in an Array
Given a non-empty array of numbers, a[0], a[1,] a[2], … , a[n-1], where 0 ≤ a[i] < 2^31.
Find the maximum result of a[i] XOR a[j], where 0 <= i, j < n.
Could you do this in O(n) runtime?

Example:

Input: [3, 10, 5, 25, 2, 8]

Output: 28

Explanation: The maximum result is 5 ^ 25 = 28.

class Solution {
public:
    int findMaximumXOR(vector<int>& nums) {
    }
};
*/

/*
12. Integer to Roman (1~3999)
*/

/*My Code*/
class Solution {
    
public:

    Solution()
    {
        RMap[1] = "I";
        RMap[5] = "V";
        RMap[10] = "X";
        RMap[50] = "L";
        RMap[100] = "C";
        RMap[500] = "D";
        RMap[1000] = "M";

        RMap[4] = "IV";
        RMap[9] = "IX";
        RMap[40] = "XL";
        RMap[90] = "XC";
        RMap[400] = "CD";
        RMap[900] = "CM";
    }

    string intToRoman(int num) {
     
        int n = num;
        
        std::vector<int> e{1000,100,10,1};
        
        std::size_t idx = 0;
        
        std::string r = "";
        
        while( n > 0 && idx < 4 )
        {
            if( n < e[idx] )
            {
                ++idx;
                continue;
            }
            
            int c = e[idx];
            int q = n / c;
            
			/*4 and 9 have special codes*/
            if( q == 4 || q == 9 )
            {
                r += RMap[q*c];
                n = n - q*c;
                continue;
            }
            
			/* process 500, 50 and 5 */
            if( q > 4 )
            {
                r += RMap[5*c];
                n = n - 5*c;
                
                continue;
            }
            
            for( int l = 0; l < q; ++l )
            {
                r += RMap[c];
            }
            
            n = n - q*c;
        }


        return r;

    }
    
private:

    std::map<int,std::string> RMap;
};

/*Other Solutions*/
class Solution {
public:
    string intToRoman(int num) {
        string res = "";
        vector<int> val{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        vector<string> str{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        for (int i = 0; i < val.size(); ++i) {
            while (num >= val[i]) {
                num -= val[i];
                res += str[i];
            }
        }
        return res;
    }
};

class Solution {
public:
    string intToRoman(int num) {
        string res = "";
        vector<string> v1{"", "M", "MM", "MMM"};
        vector<string> v2{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        vector<string> v3{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        vector<string> v4{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        return v1[num / 1000] + v2[(num % 1000) / 100] + v3[(num % 100) / 10] + v4[num % 10];
    }
};


/*
14. Longest Common Prefix
Write a function to find the longest common prefix string amongst an array of strings.
*/

class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        
        const auto num = strs.size();
        
        if( num ==  0 )
        {
            return "";
        }
        
        if( num ==  1 )
        {
            return strs[0];
        }
        // we get first string's length
        const auto len = strs[0].size();
        
        std::string result = "";
        
		// 1. iterate through each character in the first string.
		
        for( std::size_t cp = 0; cp < len; ++cp )
        {
            const auto c = strs[0][cp];
            
            bool bDiff = false;
            bool bSearchEnd = false;
        
			// 2. for current character, iterate through remaining strings in the vector.		
            for( std::size_t pos = 1; pos < num; ++pos )
            {
				// 3. get current string length 
                const auto lenComp = strs[pos].size();
                
				// 4. if the position of current character in the first string is larger than the length of current string.
				//  then mark the search will be end after the loop is ended. 
                if( cp >= lenComp )
                {
                    bSearchEnd = true;
                    continue;
                }
                
				// 5 if the character at the same position in the current string is not same as current character.
				// the loop can be ended immediately since we cannot get any longer common prefix.
                if( strs[pos][cp] != c )
                {
                    bDiff = true;
                    break;
                }
            }
            
            if( bDiff || bSearchEnd )
            {
                break;
            }
            
			//6. if the all strings have same character at same position, added this character into the final result.
			
            result += c; 
        }
        
        return result;
        
    }
};
 
/* 15  3Sum
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero

Note: The solution set must not contain duplicate triplets.

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]

*/

class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
         const auto len = nums.size();
        
        vector<vector<int>> result;
        
        if( len < 3 )
        {
            return result;
        }
        
        sort( std::begin(nums), std::end(nums));
        
        for( std::size_t i = 0; i < len - 2 ; ++i )
        {
			//note: 必须和前一个数字比较看是否相同，如果是和后一个数字比较，会错过诸如-1,-1,2这样的结果哦。
            if( i > 0 && nums[i-1] == nums[i] )
            {
                continue;
            }
            
            int k = i + 1;
            int m = len - 1;
            
            while( k < m )
            {
                if( k > (i+1) && nums[k-1] == nums[k] )
                {
                    ++k;
                    continue;
                }
				//Interesting thing: if I added following, the speed of execution is slow.
				/*
                if( m <(len-1) && nums[m+1] == nums[m] )
                {
                    --m;
                    continue;
                }
				*/
                
                int sum = nums[k]+nums[m];
            
                if( sum + nums[i] == 0 )
                {
                    result.emplace_back( std::initializer_list<int>{nums[i],nums[k],nums[m]} );
                    ++k;
                    --m;
                    continue;
                }
                
				//如果结果大于0，由于数组是排序过的，所以，我们要从尾部往回走
                if( sum +nums[i] > 0 )
                {
                    --m;
                    continue;
                }
                
				//如果结果小于0，因为数组是排序过的，要找一个更大的数，所以，从头部往前走。
                ++k;
            }
            
        }
        
        return result;
    }
};

/*
 第一步最好是对原数组进行排序。
*/

//most effient solution
//Reference website: http://www.sigmainfy.com/
/*
3Sum Problem Analysis: Handling Duplicates Without Using Set

Overview

We discuss the 3sum problem here and give a sorting approach with time complexity O(N^2) to generate all the triplets without using set to remove the repeated triplets at the end. Approach based on hash would be touched a bit too and an open question about using hash to handle duplicates in the input would be raised (copyright @sigmainfy) which for now I haven’t found a good solution to resolve. Hope the readers enjoy and any comments are highly appreciated especially on the questions that are raised here.

3Sum Problem Definition

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:

Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
The solution set must not contain duplicate triplets.
For example, given array S = {-1 0 1 2 -1 -4}, A solution set is: (-1, 0, 1) (-1, -1, 2). Note the triplets consists of real values in the input rather than the index of the input numbers

3Sum Solution Analysis

As I have mentioned a bit in previous two sum problem analysis “Two Sum Problem Analysis 3: Handling Duplicates Input“: We need to distinguish VALUE pairs and INDEX pair (To see more on two sum problems, also check these two posts: post 1, post 2). It is a bit different context here in 3sum, we need to return triplets consisting of the input values rather than the index as the two sum problem requires previously. Of course, you can do the same thing in two sum problem to find value pairs too, but let’s just try something in a different context, and you need to keep this difference in mind, otherwise, you might get confused.

The general idea would be to convert 3 sum into 2 sum problem: pick one value A[i] from the input array, the next would be find the two two sum -A[i] in the rest of the array. And the detailed steps are:

1. Sort the whole input array by increasing order: O(NlogN)
2. Linear scan the sorted array, say the current element to process is A[i], check if A[i] is the same with the previous one, only when they are different we go the next steps.
3. Treat A[i] as the first value in the potential triplet, and solve a two sum problem in the rest of the input starting from i+1 with the target as -A[i], similar trick as in step 2 should be performed to avoid repeated triplets.

Remarks: Several noteworthy things from the above steps are:

1. The total time complexity would be O(N^2) rather than O(N^2logN) because we sort the array for only one time at the beginning, many people will interpret it in a wrong way by converting 3 sum into 2 sum problems, that is, they convert it in a simple way, and do a “complete” 2 sum every time for each element in the array.
2. Everytime we pick A[i] in step three, we only need to do 2 sum in the rest of the input starting from i + 1, think about why?
We avoid repeated triplets before generating any of them by comparing the current value A[i] with the previous one A[i-1], this is the issue which bothered me for quite a while (I raised this in my earlier post in Chinese). Now we indeed managed to remove actually avoid duplicates (copyright @sigmainfy) in the final results without using the set data structure.
3. As noted in the problem description, it says the solution set must not contain duplicate triplets, a naive and straightforward way to do this is to use set to do a post-filtering after all the triplets are generated. We managed to avoid this in this post.
*/

/*
16. 3Sum Closest

Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

For example, given array S = {-1 2 1 -4}, and target = 1.

    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

*/

/*
采用类似 15  3Sum 的解法，不过这时候我们要比较和target的差，确定哪三个可以得到最小的差，另外，如果得到的和刚好和target相等就直接返回target.
*/

class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        
        const auto len = nums.size();
        
        if( len <= 3 )
        {
            return accumulate( begin(nums), end(nums), 0 ); 
        }
        
        sort( begin(nums), end(nums) );
        
        auto min_diff = numeric_limits<int>::max();
        int result = 0;
        
        for( std::size_t i = 0; i < len - 2; ++i )
        {
            if( i > 0 && nums[i-1] == nums[i] )
            {
                continue;
            }
            
            int k = i+1;
            int m = len - 1;

            while( k < m )
            {
                if( k > i+1 && nums[k-1] == nums[k] )
                {
                    ++k;
                    continue;
                }
                
                if( m < len - 1 && nums[m+1] == nums[m] )
                {
                    --m;
                    continue;
                }
                
                int sum = nums[k] + nums[m] + nums[i];
                
                if( sum ==  target )
                {
                    return target;
                }
                
				// if sum is larger than target
				// we need to search back from right
                if( sum > target )
                {
                    auto diff = sum - target;
                    if( min_diff > diff )
                    {
                        min_diff = diff;
                        result = sum;
                    }
                    --m;
                    continue;
                }
                
				// if sum is less than target
				// going forward from left.
                if( sum < target )
                {
                    auto diff = target - sum;
                    if( min_diff > diff )
                    {
                        min_diff = diff;
                        result = sum;
                    }
                    ++k;
                }
            }
                
        }
        
        return result;
    }
};

/*
17. Letter Combinations of a Phone Number
Given a digit string, return all possible letter combinations that the number could represent.
A mapping of digit to letters (just like on the telephone buttons) is given below.

Input:Digit string "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
*/

class Solution {
public:
    vector<string> letterCombinations(string digits) {
        
        return {""};
    }
};