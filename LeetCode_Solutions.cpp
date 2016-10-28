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
