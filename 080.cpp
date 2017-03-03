#include <vector>
#include <list>
#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <functional>
#include <array>
#include <map>
using namespace std;

//<--> 10. Regular Expression Matching
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
	bool isMatch(string s, string p) 
	{
		vector<vector<int>> dp(s.size() + 1, vector<int>(p.size() + 1, 0));

		dp[0][0] = 1;

		for (size_t i = 2; i <= p.size(); ++i)
		{
			if (p[i - 1] == '*')
			{
				dp[0][i] = dp[0][i - 2];
			}
		}

		for (size_t i = 1; i <= s.size(); ++i)
		{
			for (size_t j = 1; j <= p.size(); ++j)
			{
				if ( ( s[i - 1] == p[j - 1] ) || ( p[j - 1] = '.') )
				{
					dp[i][j] = dp[i - 1][j - 1];
				}
				else if (p[j - 1] == '*')
				{
					dp[i][j] = dp[i][j - 2]; // * match zero

					if ((s[i - 1] = p[j - 2]) || p[j - 2] == '.')
					{
						dp[i][j] |= dp[i - 1][j]; //* match at least 1
					}
				}
			}
		}


		return dp.back().back() == 1;
	}
};

//28. Implement strStr() 
/*
Implement strStr().
Returns the index of the first occurrence of needle in haystack, 
or -1 if needle is not part of haystack. 
*/

class Solution {
public:
    int strStr(string haystack, string needle) 
	{
		if(needle.empty()||haystack.empty()) return -1;
		auto m = haystack.size();
		auto n = needle.size();
		
		if( m< n )
		{
			return -1;
		}
		
		for( size_t i = 0; i<=m-n; ++i )
		{
			size_t j = 0;
			for( j = 0; j<n; ++j )
			{
				if( haystack[i+j] != needle[j] )
				{
					break;
				}
			}
			
			if(j==n) return i;
		}
		
		return -1;
		
    }
};

//38. Count and Say
/*
The count-and-say sequence is the sequence of integers beginning as follows:
1, 11, 21, 1211, 111221, ...

1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.

Given an integer n, generate the nth sequence.

Note: The sequence of integers will be represented as a string. 
*/

class Solution {
public:
    string countAndSay(int n) 
	{
		string res = "1";
		for(int i = 1; i<n; ++i)
		{
			string tmp;
			res.push_back('$');
			
			int count = 1;
			
			for(size_t j = 1; j < res.size(); ++j )
			{
				if(res[j-1]==res[j])
				{
					++count;
				}
				else
				{
					tmp.push_back(count+'0');
					tmp.push_back(res[j-1]);
					count = 1;
				}
			}
			
			res = tmp;
		}
		
		return res;
    }
};
//39. Combination Sum
/*
 Given a set of candidate numbers (C) (without duplicates) and a target number (T), 
 find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:

    All numbers (including target) will be positive integers.
    The solution set must not contain duplicate combinations.

For example, given candidate set [2, 3, 6, 7] and target 7,
A solution set is:

[
  [7],
  [2, 2, 3]
]
*/
class Solution {
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) 
	{
    }
	
	void dfs( vector<int>& nums, vector<int>& out, vector<vector<int>> &res, int target, int start )
	{
		if(target < 0) 
		{
			return;
		}
		if(target == 0)
		{
			res.push_back(out);
			return;
		}
		
		for(int i = start; i<nums.size();++i)
		{
			out.push_back(nums[i]);
			dfs(nums,out,res,target-nums[i], i); //start is set to i not i+1 because "The same repeated number may be chosen"
			out.pop_back();
		}
	}
};
//40. Combination Sum II
/*
Given a collection of candidate numbers (C) and a target number (T), 
find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8, 
A solution set is: 
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
*/

class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) 
	{
		
	}
	
	void dfs( vector<int>& nums, vector<int>& out, vector<vector<int>> &res, int target, int start )
	{
		if(target < 0) 
		{
			return;
		}
		if(target == 0)
		{
			res.push_back(out);
			return;
		}
		
		for(int i = start; i<nums.size();++i)
		{
			if(i>start && nums[i]==nums[i-1]) continue;
			out.push_back(nums[i]);
			dfs(nums,out,res,target-nums[i], i+1);
			out.pop_back();
		}
	}
};

//41. First Missing Positive
/*
Given an unsorted integer array, find the first missing positive integer.

For example,
Given [1,2,0] return 3,
and [3,4,-1,1] return 2.

Your algorithm should run in O(n) time and uses constant space.
*/
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) 
	{
		size_t i = 0;
		
		auto n = nums.size();
		
		while( i < n )
		{
			if(nums[i]!=i+1&&nums[i]>0&&nums[i]<=n&&nums[i]!=nums[nums[i]-1])
			{
				swap(nums[i],nums[nums[i]-1]);
			}
			else
			{
				++i;
			}
		}
		
		for( i = 0; i<n;++i)
		{
			if(nums[i]!=i+1) return i+1;
		}
		
		return n+1;
	}
};

//<--> 42. Trapping rain water
//Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining. 
//Given [0,1,0,2,1,0, 1, 3, 2, 1, 2, 1], return 6

class Solution {
public:
    int trap(vector<int>& height) 
	{
		auto n = height.size();
		
		vector<int> dp(n, 0);
		
		int max_so_far = 0;
		
		for( size_t i = 0; i < n; ++i )
		{
			dp[i] = max_so_far;
			max_so_far = max(max_so_far, height[i]);
		}
		
		max_so_far = 0;
		
		int res = 0;
		
		for( size_t i = 0; i<n; ++i )
		{
			auto idx = n-1-i;
			dp[idx] = min(max_so_far, dp[idx]);
			max_so_far = max(max_so_far, height[idx]);
			if(dp[idx] > height[idx])
			{
				res += (dp[idx]-height[idx]);
			}
		}
		
		return res;
	}
	
		//second solution
		// int trap(vector<int>& height) 
		// {
			// size_t l = 0, r = height.size() - 1;
			
			// int res = 0;
			
			// while(l<r)
			// {
				// auto min_h = min(height[l],height[r]);
				// if(height[l]==min_h)
				// {
					// ++l;
					// while(l<r && height[l]<min_h)
					// {
						// res += min_h - height[l];
						// ++l;
					// }
				// }
				// else
				// {
					// --r;
					// while(l<r && height[r]<min_h)
					// {
						// res += min_h - height[r];
						// --r;
					// }
				// }
			// }
		// }
};

//<--> 43. Multiply Strings
/*
Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2.

Note:

    The length of both num1 and num2 is less than 110.
    Both num1 and num2 contains only digits 0-9.
    Both num1 and num2 does not contain any leading zero.
    You must not use any built-in BigInteger library or convert the inputs to integer directly.

*/
class Solution {
public:
    string multiply(string num1, string num2) 
	{
		auto n1 = num1.size();
		auto n2 = num2.size();
		
		auto k = n1+n2-2;
		
		vector<int> v(n1+n2,0);
		
		for( size_t i = 0; i< n1; ++i )
		{
			for( size_t j = 0; j<n2; ++j)
			{
				v[k-i-j] += (num1[i]-'0')*(num2[i]-'0');
			}
		}
		
		int carry = 0;
		
		for(size_t i = 0; i<n1+n2; --i)
		{
			v[i] += carry;
			carry = v[i]/10;
			v[i] -= carry * 10;
		}
		
		size_t i = n1+n2-1;
		
		while(v[i]==0) --i;
		
		if(i<0) return "0";
		
		string res;
		
		while(i>=0)
		{
			res.push_back(v[i]+'0');
			--i;
		}
		
		return res;
    }
};

//44. Wildcard Matching 
/*
Implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).

The matching should cover the entire input string (not partial).

The function prototype should be:
bool isMatch(const char *s, const char *p)

Some examples:
isMatch("aa","a") → false
isMatch("aa","aa") → true
isMatch("aaa","aa") → false
isMatch("aa", "*") → true
isMatch("aa", "a*") → true
isMatch("ab", "?*") → true
isMatch("aab", "c*a*b") → false
*/

/*
difference between regular matching and wildcard matching
1. in wildcard matching, '*' is a standalong pattern character
2. in regular maching, '*' is dependent on its previous character.
*/
class Solution {
public:
    bool isMatch(string s, string p) 
	{
		vector<vector<int>> dp(s.size() + 1, vector<int>(p.size() + 1));// key: add empty string

		dp[0][0] = 1; //s is empty and p is empty, so they matched

		//fill first row: s is empty and p is not empty
		//so s="", and if p="***", they still can be matched
		for (size_t i = 1; i <= p.size(); ++i)
		{
			if (p[i - 1] == '*')
			{
				dp[i - 1] = dp[i];
			}
		}

		for (size_t i = 1; i <= s.size(); ++i)
		{
			for (size_t j = 1; j <= p.size(); ++j)
			{
				if (p[j - 1] == '*')
				{
					dp[i][j] = dp[i][j - 1] | dp[i - 1][j]; //dp[i][j-1] means '*' match zero character while dp[i-1][j] means '*' match s[i-1]
				}
				else if ((s[i - 1] == p[j - 1]) || (p[j - 1] == '?'))
				{
					dp[i][j] = dp[i - 1][j - 1];
				}
			}
		}

		return dp[s.size()][p.size()] == 1;
    }
};


//45. Jump Game II
/*
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

For example:
Given array A = [2,3,1,1,4]

The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.) 
*/

class Solution {
public:
    int jump(vector<int>& nums) 
	{
		auto n = nums.size();
		
		size_t cur = 0;
		size_t i = 0;
		
		int res = 0;
		
		while( cur < n-1 )
		{
			size_t pre = cur;
			
			while( i <= pre )
			{
				cur = max(cur, i + nums[i]);
				++i;
			}
			++res;
			
			if(pre==cur) return -1;
		}
		
		return res;
    }
};

//46. Permutations
/*
 Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:

[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

*/

class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) 
	{
		vector<vector<int>> res;
		
		return res;
    }
	
	void dfs( vector<int>& nums, vector<vector<int>>& res, size_t start  )
	{
		if( start==nums.size() )
		{
			res.push_back(nums);
		}
		else
		{
			for( size_t i = start; i< nums.size(); ++i )
			{
				swap(nums[start],nums[i]);
				dfs(nums, res, start+1);
				swap(nums[start],nums[i]);
			}
		}
	}
	
};

//47. Permutations II 
/*
 Given a collection of numbers that might contain duplicates, return all possible unique permutations.

For example,
[1,1,2] have the following unique permutations:

[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

*/

class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) 
	{
    }
	
};

//48. Rotate Image
/*
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Follow up:
Could you do this in-place?
*/
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) 
	{
		auto n = matrix.size();
		
		for( size_t i = 0; i<n; ++i )
		{
			for( size_t j = i+1; j<n; ++j )
			{
				swap(matrix[i][j], matrix[j][i]);
			}
			
			reverse(matrix[i].begin(), matrix[i].end());
		}
	}
};


//49. Group Anagrams
/*
Given an array of strings, group anagrams together.

For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
Return:

[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]
*/
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs)
	{
    }
};

//50. Pow(x, n) 
class Solution {
public:
    double myPow(double x, int n)
	{
		double res = 1.0;
		for(int i = n; i!=0; i/=2 )
		{
			if( i&1 != 0 ) res *= x;
			x* = x;
		}
	}
	
	//recursion
	double myPow(double x, int n)
	{
		if( n<0 ) return 1.0 / Pow( x, -n );
		return Pow(x,n);
	}
	
	double Pow(double x, int n)
	{
		if(n==0) return 1;
		auto half = Pow(x, n/2);
		if(n%1 == 0) return half*half;
	}
	
};


//////////////////////////////////

// DP Initialization:

// // both text and pattern are null
// T[0][0] = true; 

// // pattern is null
// T[i][0] = false; 

// // text is null
// T[0][j] = T[0][j - 1] if pattern[j – 1] is '*'  

// // If current characters match, result is same as 
// // result for lengths minus one. Characters match
// // in two cases:
// // a) If pattern character is '?' then it matches  
// //    with any character of text. 
// // b) If current characters in both match
// if ( pattern[j – 1] == ‘?’) || 
     // (pattern[j – 1] == text[i - 1])
    // T[i][j] = T[i-1][j-1]   
 
// If we encounter ‘*’, two choices are possible-
// a) We ignore ‘*’ character and move to next 
//    character in the pattern, i.e., ‘*’ 
//    indicates an empty sequence.
// b) '*' character matches with ith character in
//     input 
// else if (pattern[j – 1] == ‘*’)
    // T[i][j] = T[i][j-1] || T[i-1][j]  

// else // if (pattern[j – 1] != text[i - 1])
    // T[i][j]  = false 

//53. Maximum Subarray
/*
Find the contiguous subarray within an array
(containing at least one number) which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
*/

class Solution {
public:
    int maxSubArray(vector<int>& nums)
    {
        int res = nums[0];
        int tmp = res;
        
        for(size_t i = 1 ;i<nums.size(); ++i)
        {
            tmp = max(nums[i]+tmp, nums[i]);
            res = max(tmp, res);
        }
        
        return res;
    }
    
    //using divide and conquer
    
    int getMaxSubArray(vector<int>& nums, int left, int right)
    {
        if(left>=right) return nums[left];
        
        int mid = left+(right-left)/2;
        int lmax = getMaxSubArray(nums, left, mid-1);
        int rmax = getMaxSubArray(nums, mid+1, right);
        
        int cur_max = nums[mid], tmp = nums[mid];
        
        for(int i = mid-1; i>=left; --i)
        {
            tmp += nums[i];
            cur_max = max(tmp, cur_max);
        }
        
        tmp = cur_max;
        
        for(int i = mid+1; i<=right; ++i)
        {
            tmp += nums[i];
            cur_max = max(tmp, cur_max);
        }
        
        return max(cur_max, max(lmax, rmax));
    }
};

//54. Spiral Matrix
/*
Given a matrix of m x n elements (m rows, n columns),
return all elements of the matrix in spiral order.

For example,
Given the following matrix:

[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
You should return [1,2,3,6,9,8,7,4,5].
*/

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix)
    {
        vector<int> res;
        if(matrix.size()==0||matrix[0].size()==0) return res;
        auto m = matrix.size(), n = matrix[0].size();
        
        auto c = min(m,n)+1;
        c /= 2;
        
        size_t col = 0, row = 0;
        
        auto p = m, q = n;
        
        for( size_t i = 0; i<c; ++i, p-=2, q-=2 )
        {
            for( col = i; col < i+p; ++col )
            {
                res.push_back(matrix[i][col]);
            }
            
            for( row = i+1; row<i+q; ++row )
            {
                res.push_back(matrix[row][i+q-1]);
            }
            
            if( p == 1 ||  q == 1 ) break;
            
            for( col = i+q -2; col >= i; --col )
            {
                res.push_back(matrix[i+p-1][col]);
            }
            
            for( row = i+p-2; row > i; --row )
            {
                res.push_back(matrix[row][i]);
            }
        }
        return res;
    }
};

//55. Jump Game
/*
Given an array of non-negative integers,

you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

For example:
A = [2,3,1,1,4], return true.

A = [3,2,1,0,4], return false.
*/

class Solution {
public:
    bool canJump(vector<int>& nums)
    {
        int max_so_far = 0;
        
        for( int i = 0; i<nums.size(); ++i )
        {
            if(i>max_so_far||max_so_far >= nums.size()-1) break;
            max_so_far = max(max_so_far, i+nums[i]);
        }
        
        return max_so_far >= nums.size()-1;
    }
};

//56. Merge Intervals
/*
Given a collection of intervals,
merge all overlapping intervals.

For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18].
*/
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution
{
public:
    vector<Interval> merge(vector<Interval>& intervals)
    {
        vector<Interval> res;
        for(size_t i = 0;i<intervals.size();++i)
        {
            insert(res, intervals[i]);
        }
        
        return res;
    }
    
    void insert(vector<interval>& res, interval& newInterval)
    {
        auto iter = res.begin();
        int overlap = 0;
        
        while(iter!=res.end())
        {
            if(newInterval.start>iter->end)
            {
                ++iter;
                continue;
            }
            
            if(newInterval.end < iter->start)
            {
                break;
            }
            
            newInterval.start = min(newInterval.start, iter->start);
            newInterval.end = max(newInterval.end, iter->end);
            ++overlap;
        }
        
        if(overlap!=0)
        {
            iter = res.erase(iter - overlap, iter);
        }
        
        res.insert(iter, newInterval);
    }
};

//<--> 57. Insert Interval
/*
Given a set of non-overlapping intervals,
insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16],

insert and merge [4,9] in as [1,2],[3,10],[12,16].

This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
*/
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
    vector<Interval> insert(vector<Interval>& intervals, Interval newInterval)
    {
        auto iter = intervals.begin();
		
		int overlap = 0; // number of intervals that can be merged with newInterval
		while(iter != intervals.end())
		{
			if(newInterval.start > iter->end)
			{
				++iter;
				continue;
			}
			
			if(newInterval.end < iter->start) // key: found the first interval that is right of newInterval
			{
				break;
			}
			
			newInterval.start = min(newInterval.start, iter->start);
			newInterval.end = max(newInterval.end, iter->end);
			++overlap;
			++iter;
		}
		
		auto pos = intervals.erase(iter - overlap, iter); //important: remove intervals that overlapped with newInterval, and return the insert position.
		intervals.insert(pos, newInterval);
		
		return intervals;
    }
};

//58. Length of Last Word
/*
Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

If the last word does not exist, return 0.

Note: A word is defined as a character sequence consists of non-space characters only.

For example, 
Given s = "Hello World",
return 5.
*/

class Solution {
public:
    int lengthOfLastWord(string s)
    {
        
    }
};

//59. Spiral Matrix II
/*
Given an integer n,
'generate a square matrix filled
with elements from 1 to n^2 in spiral order.

For example,
Given n = 3,

You should return the following matrix:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
*/

class Solution {
public:
    vector<vector<int>> generateMatrix(int n)
    {
        vector<vector<int>> m(n, vector<int>(n,0));
        int p = n;
        int c = n /2;
        
        int col = 0, row = 0;
        int num = 1;
        for(int i = 0; i<c; ++i, p-=2)
        {
            for(col = i; col < i+p; ++col)
            {
                m[i][col] = num++;
            }
            
            for(row = i + 1; row< i+p; ++row)
            {
                m[row][i+p-1] = num++;
            }
            
            
            for(col=i+p-2; col >= i; --col)
            {
                m[i+p-1][col] = num++;
            }
            
            for(row=i+p-2; row > i;--row)
            {
                m[row][i] = num++;
            }
        }
        
        if( n & 1 )
        {
            m[n/2][n/2] = num;
        }
        
        return m;
    }
};

//<--> 60. Permutation Sequence
/*
The set [1,2,3,…,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order,
We get the following sequence (ie, for n = 3):

"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.

Note: Given n will be between 1 and 9 inclusive.
*/
class Solution {
public:
    string getPermutation(int n, int k)
    {
        string res;
        string num = "123456789";
        
        vector<int> f(n,1);
        
        for(int i = 1;i<n;++i)
        {
            f[i] = i*f[i-1];
        }
        
        --k;
        
        for(int i = n; i>=1; --i)
        {
            int j = k/f[i-1];
            k -= j*f[i-1];
            
            res.push_back(num[j]);
            num.erase(j,1);
        }
        
        return res;
    }
};

// 61. Rotate List 
/*
Given a list, rotate the list to the right by k places, where k is non-negative.

For example:
Given 1->2->3->4->5->NULL and k = 2,
return 4->5->1->2->3->NULL.
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
    ListNode* rotateRight(ListNode* head, int k) 
	{
		if(!head) return nullptr;
		
		int n = 1;
		auto cur = head;
		while(cur->next)
		{
			++n;
			cur = cur->next;
		}
		
		cur->next = head;
		int m = n - k%n;
		
		for(int i = 0; i<m; ++i)
		{
			cur = cur->next;
		}
		
		auto new_head = cur->next;
		cur->next = nullptr;
		
		return new_head;
    }
};

//62. Unique Paths 
/*
A robot is located at the top-left corner of a m x n grid 
(marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. 
The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
*/
class Solution {
public:
    int uniquePaths(int m, int n) 
	{
		vector<vector<int>> dp( m,  vector<int>(n, 1) );
		for(int i = 1; i< m; ++i)
		{
			for( int j = 1; j<n; ++j)
			{
				dp[i][j] = dp[i-1][j] + dp[i][j-1];
			}
		}
		
		return dp[m-1][n-1];
    }
};

//63. Unique Paths II
/*
Follow up for "Unique Paths":

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

For example,

There is one obstacle in the middle of a 3x3 grid as illustrated below.

[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]

The total number of unique paths is 2.

Note: m and n will be at most 100.
*/
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) 
	{
		    if ( obstacleGrid.empty() || obstacleGrid[0].empty() || obstacleGrid[0][0] == 1 )
			{
				return 0;
			}

			auto m = obstacleGrid.size();
			auto n = obstacleGrid[0].size();

			vector<vector<int>> dp( m, vector<int>( n, 0 ) );
			dp[0][0] = 1;

			for ( size_t i = 1; i < m; ++i )
			{
				if ( obstacleGrid[i][0] == 0 )
				{
					dp[i][0] = dp[i - 1][0];
				}
			}

			for ( size_t i = 1; i < n; ++i )
			{
				if ( obstacleGrid[0][i] == 0 )
				{
					dp[0][i] = dp[0][i-1];
				}
			}

			for ( size_t i = 1; i < m; ++i )
			{
				for ( size_t j = 1; j < n; ++j )
				{
					if ( obstacleGrid[i][j] == 1 )
					{
						dp[i][j] = 0;
					}
					else
					{
						dp[i][j] = dp[i][j - 1] + dp[i - 1][j];
					}
				}
			}

			return dp.back().back();
    }
};

//<--> 64. Minimum Path Sum
/*
Given a m x n grid filled with non-negative numbers, 
find a path from top left to bottom right which minimizes the sum of all numbers along its path.
*/

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) 
	{
    }
};

//65. Valid Number
/*
Validate if a given string is numeric.

Some examples:
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
Note: It is intended for the problem statement to be ambiguous.
 You should gather all requirements up front before implementing one.
*/
class Solution {
public:
    bool isNumber(string s) 
	{
		
    }
};

//66. Plus One
/*
Given a non-negative integer represented as a non-empty array of digits, 
plus one to the integer.

You may assume the integer do not contain any leading zero, 
except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.
*/
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) 
	{
		auto n = digits.size();
		for( size_t i = n-1; i>=1; --i)
		{
			if( digits[i] == 9 ) digits[i] = 0;
			else
			{
				digits[i] += 1;
				return digits;
			}
		}
		
		if(digits.front() == 0)
		{
			digits.insert(digits.begin(),1);
		}
		
		return digits;
    }
};

//67. Add Binary
/*
Given two binary strings, return their sum (also a binary string).

For example,
a = "11"
b = "1"
Return "100".
*/
class Solution {
public:
    string addBinary(string a, string b)
	{
		string res = "";
		if(a.empty()||b.empty())
		{
			return res;
		}
		
		int m = a.size()-1;
		int n = b.size()-1;
		int carry = 0;
		int sum = 0;
		
		while(m>=0 || n>=0)
		{
			auto p = m>=0 ? a[m--] - '0' : 0;
			auto q = n>=0 ? b[n--] - '0' : 0;
			
			sum = p+q+carry;
			
			carry = sum>>1; //  sum / 2;
			sum -= ( carry<<1 ); //carry * 2
			
			res.push_back(sum+'0');
		}
		
		if(carry==1)
		{
			res.push_back('1');
		}
		
		reverse(res.begin(), res.end());
		
		return res;
    }
};

//appendix:

int number_of_ones(unsigned int x) {
  int total_ones = 0;
  while (x != 0) {
    x = x & (x-1);
    total_ones++;
  }
  return total_ones;
}

int isPowerOfTwo (unsigned int x)
{
  return ((x != 0) && !(x & (x - 1)));
}

//68. Text Justification
/*
Given an array of words and a length L, 
format the text such that each line has exactly L characters and is fully (left and right) justified.

You should pack your words in a greedy approach; 
that is, pack as many words as you can in each line. 
Pad extra spaces ' ' when necessary so that each line has exactly L characters.

Extra spaces between words should be distributed as evenly as possible. 
If the number of spaces on a line do not divide evenly between words, 
the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left justified and no extra space is inserted between words.

For example,
words: ["This", "is", "an", "example", "of", "text", "justification."]
L: 16.

Return the formatted lines as:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
Note: Each word is guaranteed not to exceed L in length.
*/

/*

比较麻烦的字符串细节实现题。需要解决以下几个问题：

1. 首先要能判断多少个word组成一行：
这里统计读入的所有words的总长curLen，并需要计算空格的长度。
假如已经读入words[0:i]。当curLen + i <=L 且加curLen + 1 + word[i+1].size() > L时，一行结束。

2. 知道一行的所有n个words，以及总长curLen之后要决定空格分配：
平均空格数：k = (L - curLen) / (n-1)
前m组每组有空格数k+1：m = (L - curLen) % (n-1)

例子：L = 21，curLen = 14，n = 4
k = (21 - 14) / (4-1) = 2
m = (21 - 14) % (4-1)  = 1
A---B--C--D

3. 特殊情况：
(a) 最后一行：当读入到第i = words.size()-1 个word时为最后一行。该行k = 1，m = 0
(b) 一行只有一个word：此时n-1 = 0，计算(L - curLen)/(n-1)会出错。该行k = L-curLen, m = 0
(c) 当word[i].size() == L时。

*/
class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) 
    {
        vector<string> res;
		
		if(words.empty() || maxWidth == 0)
		{
		    res.push_back("");
		    return res;
		}
		
		
        int num_words = static_cast<int>(words.size());
		int start = 0, end = -1, total_word_len = 0;
		
		int i = 0; 

		
		while(i<num_words)
		{
		    int cur_word_len = words[i].size();
		    if(cur_word_len > maxWidth) return res;
		    int new_len = total_word_len + (end-start+1) + cur_word_len;
		    if(new_len <= maxWidth)
		    {
		        total_word_len += cur_word_len;
		        end = i;
		        ++i;
		    }
		    else
		    {
		        create_line(words,res,start,end,total_word_len,maxWidth,false);
		        start=i;
		        end = i-1;
		        total_word_len = 0;
		    }
		}
		
		create_line(words,res,start,end,total_word_len,maxWidth,true);
		return res;

    }
    
    void create_line( vector<string>& words, vector<string>& res, int start, int end, int word_len, int W, bool bLast )
	{
	    int n = words.size();
	    if(start<0||end>=n||start>end) return;
	    
	    n = end - start + 1;
	    string line = words[start];
	    
	    if(n==1||bLast)
	    {
	        for(int i = start+1; i<=end; ++i)
	        {
	            line += " ";
	            line += words[i];
	        }
	        
	        int nspace = W - word_len - (n-1);
	        line.append(nspace, ' ');
	        res.push_back(line);
	        return;
	    }
	    
	    int k = (W-word_len)/(n-1);
	    int m = (W-word_len) - k*(n-1);
	    
	    for(int i = start+1; i<=end; ++i)
	    {
	        int nspace = i-start<=m ? k+1:k;
	        line.append(nspace, ' ');
	        line.append(words[i]);
	     
	    }
	    
	    res.push_back(line);
	}
};


//<--> 69. Sqrt(x)
/*Implement int sqrt(int x).*/
class Solution {
public:
    int mySqrt(int x) 
	{
		
    }
};

//<--> 71. Simplify Path
/*
Given an absolute path for a file (Unix-style), 
simplify it.

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"
*/

class Solution {
public:
    string simplifyPath(string path)
	{
		string t;
		stringstream ss(path);
		vector<string> v;
		while(getline(path, t, '/'))
		{
			if( t.empty() || t == "." )
			{
				continue;
			}
			
			if(t==".."&&!v.empty())
			{
				v.pop_back();
			}
			else if( t != ".." )
			{
				v.push_back(t);
			}
		}
		
		string result;
		for( const auto& s : v )
		{
			result += "/";
			result += s;
		}
		
		return result.empty() ? "/" : result;
    }
};

//<--> 72. Edit Distance
/*
 Given two words word1 and word2, 
 find the minimum number of steps required to convert word1 to word2. 
 (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

a) Insert a character
b) Delete a character
c) Replace a character
*/
class Solution {
public:
    int minDistance(string word1, string word2) 
	{
		auto n1 = word1.size();
		auto n2 = word2.size();
		
		vector<vector<int>> dp( n1+1, vector<int>(n2+1,0) );
		for(size_t i = 0; i<=n2; ++i)
		{
			dp[0][i] = i;
		}
		for(size_t i = 0; i<=n1; ++i)
		{
			dp[i][0] = i;
		}
		
		for(size_t i = 1; i<=n1; ++i)
		{
			for(size_t j = 1;j<=n2; ++j)
			{
				if(words1[i-1]==words2[j-1])
				{
					dp[i][j] = dp[i-1][j-1];
				}
				else
				{
					dp[i][j] = min(dp[i][j-1], dp[i-1][j]);
					dp[i][j] = min(dp[i][j],dp[i-1][j-1]);
					dp[i][j] += 1;
				}
			}
		}
		
		return dp[n1][n2];
    }
};

//<--> 73. Set Matrix Zeroes
/*
Given a m x n matrix, if an element is 0, 
set its entire row and column to 0. Do it in place.

Follow up:

Did you use extra space?
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
*/

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix)
	{
		if(matrix.empty() || matrix[0].empty())
		{
			return;
		}
		
		int r = matrix.size();
		int c = matrix[0].size();
		
		bool row_zero = false;
		bool col_zero = false;
		
		for( int i = 0; i< r; ++i )
		{
			if( matrix[i][0] == 0 )
			{
				col_zero = true;
				break;
			}
		}
		
		for( int j = 0; j < c; ++j)
		{
			if(matrix[0][j] == 0)
			{
				row_zero = true;
				break;
			}
		}
		
		for( int i = 1; i< r; ++i )
		{
			for(int j = 1; j<c; ++j)
			{
				if( matrix[i][j] == 0 )
				{
					matrix[0][j] = 0;
					matrix[i][0] = 0;
				}
			}
		}
		
		for( int i = 1; i< r; ++i )
		{
			for(int j = 1; j<c; ++j)
			{
				if(matrix[0][j] == 0 || matrix[i][0] == 0)
				{
					matrix[i][j] = 0;
				}
			}
		}
		
		if(row_zero)
		{
			for(int i = 0; i<c; ++i)
			{
				matrix[0][i] = 0;
			}
		}
		
		if(col_zero)
		{
			for(int j = 0; j<r; ++j)
			{
				matrix[j][0] = 0;
			}
		}
    }
};

//<--> 74. Search a 2D Matrix 
/*
Write an efficient algorithm that searches for a value in an m x n matrix. 
This matrix has the following properties:

    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.

For example,

Consider the following matrix:

[
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]

Given target = 3, return true.
*/

class Solution {
public:
	//using two binary search
    bool searchMatrix(vector<vector<int>>& matrix, int target)
	{
		if(matrix.empty()||matrix[0].empty())
		{
			return false;
		}
		
		if(target > matrix.back().back() || target < matrix[0] ) return false;
		
		int m = matrix.size();
		int n = matrix[0].size();
		int left = 0, right = m-1;
		while(left<=right)
		{
			int mid = (left+right)/2;
			if(matrix[mid][0]==target)
			{
				return true;
			}
			
			if(matrix[mid][0]<target)
			{
				left = mid + 1;
			}
			else
			{
				right = mid-1;
			}
		}
		
		int row = right;
		left = 0;
		right = n-1;
		
		while(left<=right)
		{
			int mid = (left+right)/2;
			if(matrix[row][mid]==target)
			{
				return true;
			}
			
			if(matrix[row][mid]<target)
			{
				left = mid + 1;
			}
			else
			{
				right = mid-1;
			}
		}
		
		return false;
    }
	
	//using one binary search
	bool searchMatrix(vector<vector<int>>& matrix, int target)
	{
		if(matrix.empty()||matrix[0].empty())
		{
			return false;
		}
		
		if(target > matrix.back().back() || target < matrix[0] ) return false;
		
		int m = matrix.size();
		int n = matrix[0].size();
		
		int left = 0, right = m*n-1;
		while(left<=right)
		{
			int mid = (left+right)/2;
			if(matrix[mid/n][mid%n] == target) return true;
			else if(matrix[mid/n][mid%n]<target) left = mid + 1;
			else if(matrix[mid/n][mid%n]>target) right = mid - 1;
		}
		
		return false;
    }
};

//<-->75. Sort Colors
/*
Given an array with n objects colored red, white or blue, 
sort them so that objects of the same color are adjacent, 
with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent 
the color red, white, and blue respectively.

Note:
You are not suppose to use the library's sort function for this problem

Follow up:
A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, 
then overwrite array with total number of 0's, then 1's and followed by 2's.

Could you come up with an one-pass algorithm using only constant space?
*/
class Solution {
public:
    void sortColors(vector<int>& nums) 
	{
		if(nums.empty()) return;
		
		int red = 0;
		int blue = nums.size() - 1;
		
		for( int i = 0; i<=blue;++i)
		{
			if(nums[i]==0)
			{
				swap(A[i], A[red++]);
			}
			else if(nums[i]==2)
			{
				swap(A[i--], A[blue--]);
			}
		}
    }
	
	// another way:
	/*
	 假设已经完成到如下所示的状态：

0......0   1......1  x1 x2 .... xm   2.....2
          |           |         |
          left        cur     right

(1) A[cur] = 1：已经就位，cur++即可
(2) A[cur] = 0：交换A[cur]和A[left]。由于A[left]=1或left=cur，所以交换以后A[cur]已经就位，cur++，left++
(3) A[cur] = 2：交换A[cur]和A[right]，right--。由于xm的值未知，cur不能增加，继续判断xm。
cur > right扫描结束。
	*/
	void sortColors(vector<int>& nums) 
	{
		int left = 0, right = nums.size()-1;
		int cur = 0;
		while(cur<=right)
		{
			if(nums[cur]==0)
			{
				swap(nums[cur++], nums[left++]);
			}
			else if(nums[cur]==1)
			{
				++cur;
			}
			else if(nums[cur]==2)
			{
				swap(nums[cur],nums[right--]);
			}
		}
	}
	
};

//<-->76. Minimum Window Substring
/*
Given a string S and a string T, 
find the minimum window in S which will contain all the characters in T in complexity O(n).

For example,
S = "ADOBECODEBANC"
T = "ABC"

Minimum window is "BANC".

Note:
If there is no such window in S that covers all characters in T, return the empty string "".

If there are multiple such windows, you are guaranteed that there will always be only 
one unique minimum window in S. 
*/

/*
1. 如果S[i:j]是min window，那么S[i], S[j]必然也在T中。
2. 对于任意S[i]要检查是否也在T中，需要将T的所有字符建立一个hash table。T中可能有重复字符，所以key = T[i], val = freq of (T[i])。
3. 先找到第一个window包含T
4. 扫描S[i]
若S[i]不在T中，跳过。
若S[i]在T中，freq of (S[i]) - 1，并且match的长度+1

*/
class Solution {
public:
    string minWindow(string s, string t)
	{
      if(s.empty()||t.empty()) return "";
        
        int ss = s.size();
        int ts = t.size();
        
        if(ss<ts) return "";
        
        vector<int> m(256, 0);
        vector<int> n(256, 0);
        
        for(int i = 0;i<ts; ++i)
        {
            ++m[t[i]];
            n[t[i]] = 1;
        }
        
        int left = 0, right = 0, count = 0, min_len = ss+1;
        string res;
        for(;right<ss;++right)
        {
            if(n[s[right]] == 1)
            {
                --m[s[right]];
                if(m[s[right]] >= 0)
                {
                    ++count;
                }
                
                while(count==ts)
                {
                    int cur_len = right-left+1;
                    if(cur_len<min_len)
                    {
                        min_len = cur_len;
                        res = s.substr(left,min_len);
                    }
                    
                    if(n[s[left]]==1)
                    {
                        ++m[s[left]];
                        if(m[s[left]]>0)
                        {
                            --count;
                        }
                    }
                    
                    ++left;
                }
            }
        }
        
        return res;
	}
};

//<--> 77. Combinations
/*
 Given two integers n and k, 
 return all possible combinations of k numbers out of 1 ... n.

For example,
If n = 4 and k = 2, a solution is:

[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
*/
class Solution {
public:
    vector<vector<int>> combine(int n, int k) 
	{
		vector<vector<int>> res;
		vector<int> out;
		dfs(res,out,n,k,0,0);
		return res;
    }
	
	void dfs(vector<vector<int>>& res, vector<int>& out, int n, int k, int start, int count)
	{
		if(count==k)
		{
			res.push_back(out);
			return;
		}
		
		for(int i = start; i<=n; ++i)
		{
			out.push_back(i);
			dfs(res,out,n,k,i+1,count+1);
			out.pop_bacK();
		}
	}
};

//<--> 78. Subsets
/*
 Given a set of distinct integers, nums, return all possible subsets.

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,3], a solution is:

[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
*/
class Solution {
public:
	//first method: iterative
    vector<vector<int>> subsets(vector<int>& nums) 
	{
		vector<vector<int>> res(1);
		if(nums.empty()) return res;
		
		int n = nums.size();
		
		for( int i = 0; i<n; ++i )
		{
			int s = res.size();
			for(int j = 0;j<s;++j)
			{
				res.push_back(res[j]);
				res.back().push_back(nums[i]);
			}
		}
		
		return res;
    }
	
	//second method: recursive
	vector<vector<int>> subsets(vector<int>& nums)
	{
		vector<vector<int>> res;
		vector<int> out;
		dfs(nums, out, res, 0);
		return res;
	}
	
	void dfs(vector<int>& nums, vector<int>& out, vector<vector<int>> &res, int start)
	{
		res.push_back(out);
		int n = nums.size();
		for(int i = start; i<n;++i)
		{
			out.push_back(i);
			dfs(nums,out, res,i+1);
			out.pop_back();
		}
	}
};
//<--> 79. Word Search
/*
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell,
where "adjacent" cells are those horizontally or vertically neighboring.
The same letter cell may not be used more than once.

For example,
Given board =

[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.
*/
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        
        if(board.empty()||board[0].empty())
        {
            return false;
        }
        
        int r = board.size(), c = board[0].size();
        
        vector<vector<int>> visited(r, vector<int>(c,0));
        
        for(int i = 0; i<r; ++i)
        {
            for(int j = 0;j <c; ++j)
            {
                if(board[i][j] == word[0])
                {
                    if(search(board,visited,word,0,i,j))
                    {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    bool search(vector<vector<char>>& b, vector<vector<int>>& v, string& word, int start, int i, int j)
    {
        if(start==word.size())
        {
            return true;
        }
        
        int r = b.size(), c = b[0].size();
        
        if(i<0||i>=r||j<0||j>=c||v[i][j]==1||b[i][j]!=word[start])
        {
            return false;
        }
        
        v[i][j] = 1;
        if(search(b,v,word,start+1, i-1, j)) return true;
        if(search(b,v,word,start+1, i+1, j)) return true;
        if(search(b,v,word,start+1, i, j-1)) return true;
        if(search(b,v,word,start+1, i, j+1)) return true;
        
        v[i][j] = 0;
        return false;
    }
};

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
            for(int j = cur_sz - 1; j>=0; --j) //key: this is mirror: from end to start
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
                    tn->right = right[k]; //important: 对产生的right子树集合的每一个sub tree，接到节点的 right
                    
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


//<--> 172. Factorial Trailing Zeroes
/*
Given an integer n, return the number of trailing zeroes in n!.

Note: Your solution should be in logarithmic time complexity.
*/
class Solution {
public:
    int trailingZeroes(int n)
    {
        int res = 0;
        while(n!=0)
        {
            res += n/5;
            n /= 5;
        }
        
        return res;
    }
};


//<--> 173. Binary Search Tree Iterator
/*
Implement an iterator over a binary search tree (BST).

Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory,

where h is the height of the tree.
*/

/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class BSTIterator {
public:
    BSTIterator(TreeNode *root)
    {
        while(root)
        {
            s.push(root);
            root = root->left;
        }
    }

    /** @return whether we have a next smallest number */
    bool hasNext()
    {
        return !s.empty();
    }

    /** @return the next smallest number */
    int next()
    {
        auto p = s.top();
        s.pop();
        
        int res = p->val;
        
        if(p->right)
        {
            p = p->right;
            while(p)
            {
                s.push(p);
                p = p->left;
            }
        }
        
        return res;
    }
    
    private:
        stack<TreeNode*> s;
};

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = BSTIterator(root);
 * while (i.hasNext()) cout << i.next();
 */

//<--> 174. Dungeon Game
/*
The demons had captured the princess (P) and imprisoned her
in the bottom-right corner of a dungeon.

The dungeon consists of M x N rooms laid out in a 2D grid.

Our valiant knight (K) was initially positioned in the top-left room
and must fight his way through the dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer.
If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons, so the knight loses health
(negative integers) upon entering these rooms;
other rooms are either empty (0's)
or contain magic orbs that increase the knight's health (positive integers).

In order to reach the princess as quickly as possible,
the knight decides to move only rightward or downward in each step.


Write a function to determine the knight's minimum initial health
so that he is able to rescue the princess.

Notes:

The knight's health has no upper bound.

Any room can contain threats or power-ups,
even the first room the knight enters and the bottom-right room
where the princess is imprisoned.
*/
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon)
    {
        auto& d = dungeon;
        int rows = d.size();
        int cols = d[0].size();
        
        vector<vector<int>> dp(rows, vector<int>(cols, 0));
        
        dp[rows-1][cols-1] = max(1, 1 - d[rows-1][cols-1]);
        
        for(int i = rows-2; i>=0; --i)
        {
            dp[i][cols-1] = max(1, dp[i+1][cols-1] - d[i][cols-1]);
        }
        
        for(int j=cols - 2; j>=0; --j)
        {
            dp[rows-1][j] = max(1, dp[rows-1][j+1] - d[rows-1][j]);
        }
        
        for(int i = rows-2; i>=0; --i)
        {
            for(int j = cols-2; j>=0; --j)
            {
                dp[i][j] = max(1, min(dp[i][j+1], dp[i+1][j]) - d[i][j] );
            }
        }
        
        return dp[0][0];
    }
};

//<--> 179. Largest Number
/*
Given a list of non negative integers,

arrange them such that they form the largest number.

For example, given [3, 30, 34, 5, 9], t
Given an input string, reverse the string word by word. A word is defined as a sequence of non-space characters.
The input string does not contain leading or trailing spaces and the words are always separated by a single space.
For example,
Given s = "the sky is blue",
return "blue is sky the".
Could you do it in-place without allocating extra space?
he largest formed number is 9534330.

Note: The result may be very large,

so you need to return a string instead of an integer.
*/
class Solution {
public:
    string largestNumber(vector<int>& nums)
    {
        if(nums.empty())
        {
            return "";
        }
        
        sort(nums.begin(), nums.end(), [](int a, int b){
           
           return to_string(a) + to_string(b) > to_string(b) + to_string(a);
            
        });
        
        string res;
        
        for( int d : nums )
        {
            res += to_string(d);
        }
        
        return res[0] == '0' ? 0 : res;
    }
};

//<--> 186. Reverse Words in a String II
/*
Given an input string, reverse the string word by word.
A word is defined as a sequence of non-space characters.
The input string does not contain leading or trailing spaces
and the words are always separated by a single space.

For example,
Given s = "the sky is blue",
return "blue is sky the".
Could you do it in-place without allocating extra space?
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
        int len = s.size();
        
        for(int i = 0; i<len; ++i)
        {
            if(s[i] == ' ')
            {
                if(store_pos > 0)
                {
                    s[store_pos++] = ' ';
                }
            }
            else
            {
                int j = i;
                while(j<len && s[j] != ' ')
                {
                    s[store_pos++] = s[j++];
                }
                
                reverse(s.begin()+store_pos - (j-i), s.begin() + store_pos);
                
                i = j;
            }
        }
        
        s.resize(store_pos);
    }
};

//<--> 187. Repeated DNA Sequences
/*
All DNA is composed of a series of nucleotides abbreviated

as A, C, G, and T, for example: "ACGAATTCCG".

When studying DNA,

it is sometimes useful to identify repeated sequences within the DNA.

Write a function to find all the 10-letter-long sequences

(substrings) that occur more than once in a DNA molecule.

For example,

Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",

Return:
["AAAAACCCCC", "CCCCCAAAAA"].
*/

class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s)
    {
        string res;
        if(s.size()<=10)
        {
            return {};
        }
        
        int cur = 0;
        int i = 0;
        
        int mask = 0x7ffffff;
        
        while(i<9)
        {
            cur = (cur <<  3) | (s[i++] & 7);
        }
        
        int len = s.size();
        
        unordered_set m<int, int> m;
        
        while( i < len )
        {
            cur = ((cur & mask) << 3) | (s[i++] & 7);
            
            if(m.find[cur]!=m.end())
            {
                if(m[cur] == 1)
                {
                    res.push_back(s.substr(i-10, 10));
                }
                
                ++m[cur];
            }
            else
            {
                m[cur] = 1;
            }
        }
        
        return res;
    }
};

//<--> 188. Best Time to Buy and Sell Stock IV
/*
Say you have an array for which the ith element

is the price of a given stock on day i.

Design an algorithm to find the maximum profit.

You may complete at most k transactions.

Note:
You may not engage in multiple transactions at the same time

(ie, you must sell the stock before you buy again).
*/

/*
 *In this post,
 *we are only allowed to make at max k transactions.
 *The problem can be solve by using dynamic programming.

Let profit[t][i] represent maximum profit using at most t transactions up to day i
(including day i). Then the relation is:

profit[t][i] = max(profit[t][i-1], max(price[i] – price[j] + profit[t-1][j]))
          for all j in range [0, i-1]

profit[t][i] will be maximum of –

1. profit[t][i-1] which represents not doing any transaction on the ith day.
2. Maximum profit gained by selling on ith day.
In order to sell shares on ith day,
we need to purchase it on any one of [0, i – 1] days.
If we buy shares on jth day and sell it on ith day,
max profit will be price[i] – price[j] + profit[t-1][j]
where j varies from 0 to i-1.
Here profit[t-1][j] is best
we could have done with one less transaction till jth day.
 */

class Solution {
public:
    int maxProfit(int k, vector<int>& prices)
    {
		if ( k == 0 || prices.empty() )
		{
			return 0;
		}

		int len = prices.size();

		vector<vector<int>> dp( k + 1, vector<int>( len, 0 ) );

		if ( k > price.size() )
		{
			//since k is large than the prices length, we can sell as long as
			//the price is higher than last day

			int result = 0;

			for ( int i = 0; i < len; ++i )
			{
				if ( prices[i] > prices[i - 1] )
				{
					result += prices[i] - prices[i - 1];
				}
			}

			return result;
		}

		for ( int i = 1; i <= k; ++i )
		{
			for ( int j = 1; j < len; ++j )
			{
				int max_so_far = INT_MIN;

				for ( int m = 0; m < j; ++m )
				{
					max_so_far = max(max_so_far,prices[j] - prices[m] + dp[i - 1][m]);
				}

				dp[i][j] = max( dp[i][j - 1], max_so_far );
			}
		}

		return dp.back().back();
    }
    
    // more efficient
    int maxProfit(int k, vector<int>& prices)
    {
        if(k == 0||prices.empty())
        {
            return 0;
        }
        
        int len = prices.size();
           
        vector<vector<int>> dp(k+1, vector<int>(len , 0));
        
        if( k > price.size() )
        {
            //since k is large than the prices length, we can sell as long as
            //the price is higher than last day
            
            int result = 0;
            
            for(int i = 0; i< len; ++i)
            {
                if( prices[i] > prices[i-1] )
                {
                    result += prices[i] - prices[i-1];
                }
            }
            
            return result;
        }
        
        for(int i = 1; i<=k; ++i)
        {
            int prev_diff = INT_MIN;
            
            for(int j = 1; j<len; ++j)
            {
                prev_diff = max(prev_diff, dp[i-1][j-1] - prices[j-1]);               
                dp[i][j] = max(dp[i][j-1], prev_diff + prices[j]);
            }
        }
        
        return dp.back().back();
    }    
};

//<--> 189. Rotate Array
/*
Rotate an array of n elements to the right by k steps.

For example, with n = 7 and k = 3,
the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

Note:
Try to come up as many solutions as you can,
there are at least 3 different ways to solve this problem.
*/
class Solution {
public:
	// method 1: 3 reverse
    void rotate(vector<int>& nums, int k)
    {
		if ( nums.empty() )
		{
			return;
		}

		int n = num.size();
		k %= n;

		if ( k == 0 )
		{
			return;
		}

		reverse( nums.begin(), nums.begin() + n - k );
		reverse( nums.begin() + n - k, nums.end() );
		reverse( nums.begin(), nums.end() );
    }
	
	//method 2: vector push_back and erase
	void rotate( vector<int>& nums, int k )
	{
		if ( nums.empty() )
		{
			return;
		}

		int n = num.size();
		k %= n;

		if ( k == 0 )
		{
			return;
		}

		for ( int i = 0; i < n - k; ++i )
		{
			nums.push_back( nums[i] );
		}

		nums.erase( nums.begin(), nums.begin() + n - k );
	}

	//method 3: using swap
	void rotate( vector<int>& nums, int k )
	{
		if ( nums.empty() )
		{
			return;
		}

		int n = num.size();

		int start = 0;

		while ( n != 0 )
		{
			k %= n;

			if ( k == 0 )
			{
				break;
			}

			for ( int i = 0; i < k; ++i )
			{
				swap( nums[i + start], nums[n - k + i + start] );
			}

			n -= k;
			start += k;
		}
	}
};

//<--> 190. Reverse Bits
/*
Reverse bits of a given 32 bits unsigned integer.

For example, given input 43261596
(represented in binary as 00000010100101000001111010011100),
return 964176192 (represented in binary as 00111001011110000010100101000000).

Follow up:
If this function is called many times, how would you optimize it?
*/

class Solution {
public:
	uint32_t reverseBits( uint32_t n )
	{
		uint32_t res = 0;

		for ( int i = 0; i < 32; ++i )
		{
			res |= ((n >> i) & 1) << (31 - i);
		}
	}
};


//<--> 198. House Robber
/*
You are a professional robber planning to rob houses along a street.

Each house has a certain amount of money stashed,

the only constraint stopping you from robbing each of them is that

adjacent houses have security system connected

and it will automatically contact the police

if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing
the amount of money of each house,
determine the maximum amount of money you can rob tonight
without alerting the police.
*/
class Solution {
public:
	int rob( vector<int>& nums )
	{
		if ( nums.empty() )
		{
			return 0;
		}

		if ( nums.size() == 1 )
		{
			return nums[0];
		}

		vector<int> dp( nums.size(), 0 );
		dp[0] = nums[0];

		dp[1] = max( nums[0], nums[1] );

		for ( size_t i = 2; i < nums.size(); ++i )
		{
			dp[i] = max( dp[i - 2] + nums[i], dp[i - 1] );
		}


		return dp.back();
	}
};

//<--> 199. Binary Tree Right Side View
/*
Given a binary tree,

imagine yourself standing on the right side of it,

return the values of the nodes you can see ordered from top to bottom.

For example:
Given the following binary tree,
   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---
You should return [1, 3, 4].
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
    vector<int> rightSideView(TreeNode* root)
    {
        
    }
};

//<--> 200. Number of Islands
/*
Given a 2d grid map of '1's (land) and '0's (water),

count the number of islands.

An island is surrounded by water

and is formed by connecting adjacent lands horizontally or vertically.

You may assume all four edges of the grid are all surrounded by water.

Example 1:

11110
11010
11000
00000
Answer: 1

Example 2:

11000
11000
00100
00011
Answer: 3
*/
class Solution {
public:
    int numIslands(vector<vector<char>>& grid)
    {
		vector<vector<int>> visit( grid.size(), vector<int>( grid[0].size(), 0 ) );

		int rows = grid.size();
		int cols = grid[0].size();

		int nums = 0;

		for ( int i = 0; i < rows; ++i )
		{
			for ( int j = 0; j < cols; ++j )
			{
				if ( grid[i][j] == 1 && visit[i][j] == 0 )
				{
					dfs( grid, visit, i, j );
					++nums;
				}
			}
		}

		return nums;
    }

	//DFS method to mark all islands
	void dfs( vector<vector<char>>& grid, vector<vector<int>>& v, int r, int c )
	{
		int dx[] = { 0, 0, -1, 1 };
		int dy[] = { 1, -1, 0, 0 };

		int rows = grid.size();
		int cols = grid[0].size();

		v[r][c] = 1;

		for ( int i = 0; i < 4; ++i )
		{
			int x = r + dx[i];
			int y = c + dy[i];

			if ( x < 0 || y < 0 || x >= rows || y >= cols || grid[x][y] == 0 || v[x][y] == 1 )
			{
				continue;
			}

			dfs( grid, v, x, y );
		}
	}
};

//<--> 201. Bitwise AND of Numbers Range
/*
Given a range [m, n] where 0 <= m <= n <= 2147483647,

return the bitwise AND of all numbers in this range, inclusive.

For example, given the range [5, 7], you should return 4.
*/
class Solution {
public:
    int rangeBitwiseAnd(int m, int n)
	{

	}
};

//<--> 202. Happy Number
/*
Write an algorithm to determine if a number is "happy".

A happy number is a number
defined by the following process:
Starting with any positive integer,
replace the number by the sum of the squares of its digits,
and repeat the process
until the number equals 1 (where it will stay),
or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy numbers.

Example: 19 is a happy number

1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
*/
class Solution {
public:
    bool isHappy(int n)
    {    
    }
};

//<--> 203. Remove Linked List Elements
/*
Remove all elements from a linked list of integers that have value val.

Example
Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
Return: 1 --> 2 --> 3 --> 4 --> 5
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
    ListNode* removeElements(ListNode* head, int val)
    {
        
    }
};

//<--> 204. Count Primes
/*
Count the number of prime numbers
less than a non-negative number, n.
*/
class Solution {
public:
    int countPrimes(int n)
    {
        
    }
};

//<--> 205. Isomorphic Strings
/*
Given two strings s and t, determine if they are isomorphic.

Two strings are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character

must be replaced with another character

while preserving the order of characters.
No two characters may map to the same character
but a character may map to itself.

For example,
Given "egg", "add", return true.

Given "foo", "bar", return false.

Given "paper", "title", return true.

Note:
You may assume both s and t have the same length.
*/
class Solution {
public:
    bool isIsomorphic(string s, string t) {
    }
};


//<--> 206. Reverse Linked List
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
    ListNode* reverseList(ListNode* head)
    {
        
    }
};

//<--> 207. Course Schedule
/*
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites,

for example to take course 0 you have to first take course 1,

which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs,

is it possible for you to finish all courses?

For example:

2, [[1,0]]

There are a total of 2 courses to take.

To take course 1 you should have finished course 0. So it is possible.


2, [[1,0],[0,1]]
There are a total of 2 courses to take.

To take course 1 you should have finished course 0,

and to take course 0 you should also have finished course 1.

So it is impossible.

Note:
The input prerequisites is a graph represented by a list of edges,
not adjacency matrices. Read more about how a graph is represented.

You may assume that there are no duplicate edges in the input prerequisites.

Hints:
1. This problem is equivalent to finding
if a cycle exists in a directed graph.
If a cycle exists, no topological ordering exists
and therefore it will be impossible to take all courses.
2. Topological Sort via DFS - A great video tutorial (21 minutes) on Coursera
explaining the basic concepts of Topological Sort.
3. Topological sort could also be done via BFS.
*/
class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites)
    {
        
    }
};

//<--> 208. Implement Trie (Prefix Tree)
/*
Implement a trie with insert, search, and startsWith methods.

Note:
You may assume that all inputs are consist of lowercase letters a-z.
*/
class Trie {
public:
    /** Initialize your data structure here. */
    Trie() {
        
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * bool param_2 = obj.search(word);
 * bool param_3 = obj.startsWith(prefix);
 */

//<--> 209. Minimum Size Subarray Sum
/*
Given an array of n positive integers

and a positive integer s, find the minimal length
of a contiguous subarray of which the sum >= s.

If there isn't one, return 0 instead.

For example, given the array [2,3,1,2,4,3] and s = 7,
the subarray [4,3] has the minimal length under the problem constraint.

More practice:
If you have figured out the O(n) solution,

try coding another solution of which the time complexity is O(n log n).
*/

class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums)
    {
        
    }
};

//<--> 210. Course Schedule II
/*
There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites,
for example to take course 0 you have to first take course 1,
\which is expressed as a pair: [0,1]

Given the total number of courses
and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

For example:

2, [[1,0]]
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]

4, [[1,0],[2,0],[3,1],[3,2]]
There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].

Note:
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
*/

//<--> 211. Add and Search Word - Data structure design
/*
Design a data structure that supports the following two operations:

void addWord(word)
bool search(word)

search(word) can search a literal word or a regular expression string containing only letters a-z or .. A

. means it can represent any one letter.

For example:

addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
Note:
You may assume that all words are consist of lowercase letters a-z.
*/

class WordDictionary {
public:
    /** Initialize your data structure here. */
    WordDictionary() {
        
    }
    
    /** Adds a word into the data structure. */
    void addWord(string word) {
        
    }
    
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    bool search(string word) {
        
    }
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * bool param_2 = obj.search(word);
 */

//<--> 212. Word Search II
/*
Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell,
where "adjacent" cells are those horizontally or vertically neighboring.
The same letter cell may not be used more than once in a word.

For example,
Given words = ["oath","pea","eat","rain"] and board =

[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

Return ["eat","oath"].
Note:
You may assume that all inputs are consist of lowercase letters a-z.

You would need to optimize your backtracking to pass the larger test.

Could you stop backtracking earlier?

If the current candidate does not exist in all words' prefix,

you could stop backtracking immediately.

What kind of data structure could answer such query efficiently?

Does a hash table work? Why or why not? How about a Trie?

If you would like to learn how to implement a basic trie,

please work on this problem: Implement Trie (Prefix Tree) first.
*/

class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words)
    {    
    }
};

//<--> 213. House Robber II (Too easy) 
/*
After robbing those houses on that street,

the thief has found himself a new place for his thievery

so that he will not get too much attention.

This time, all houses at this place are arranged in a circle.

That means the first house is the neighbor of the last one.

Meanwhile, the security system for these houses remain

the same as for those in the previous street.

Given a list of non-negative integers representing

the amount of money of each house,

determine the maximum amount of money

you can rob tonight without alerting the police.
*/
class Solution {
public:
    int rob(vector<int>& nums)
    {
    }
};

//<--> 214. Shortest Palindrome
/*
Given a string S, you are allowed to convert it to

a palindrome by adding characters in front of it.

Find and return the shortest palindrome you can find by performing this transformation.

For example:

Given "aacecaaa", return "aaacecaaa".

Given "abcd", return "dcbabcd".
*/

class Solution {
public:
    string shortestPalindrome(string s)
    {    
    }
};

//<--> 215. Kth Largest Element in an Array
/*
Find the kth largest element in an unsorted array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

Note: 
You may assume k is always valid, 1 <=  k <= array's length.
*/

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k)
    {    
    }
};


//<--> 216. Combination Sum III
/*
Find all possible combinations of k numbers
that add up to a number n,
given that only numbers from 1 to 9 can be used and
each combination should be a unique set of numbers.

Example 1:

Input: k = 3, n = 7

Output:

[[1,2,4]]

Example 2:

Input: k = 3, n = 9

Output:

[[1,2,6], [1,3,5], [2,3,4]]
*/
class Solution {
public:
    vector<vector<int>> combinationSum3(int k, int n)
    {
			vector<vector<int>> res;
			vector<int> out;
			
			dfs(k, n, 1, out, res);
			
			return res;
    }
		
		void dfs(int k, int sum, int start, vector<int>& out, vector<vector<int>>& res)
		{
			if(sum==0 && out.size() == k)
			{
				res.push_back(out);
				return;
			}
			
			//error: for(int i = start; i<=sum; ++i) since sum can be larger than 9
			for(int i = start; i<=9; ++i) 
			{
				if(i<=sum) // we only apply number that less than current sum
				{
					out.push_back(i);
					dfs(k,sum-i,i+1,out,res);
					out.pop_back();
				}
			}
		}
};


//<--> 217. Contains Duplicate <-->
/*
Given an array of integers,
find if the array contains any duplicates.
Your function should return true
if any value appears at least twice in the array,
and it should return false if every element is distinct.
*/

class Solution {
public:
    bool containsDuplicate(vector<int>& nums)
    {    
    }
};

//<--> 218. The Skyline Problem
/*
A city's skyline is the outer contour of the silhouette
formed by all the buildings in that city when viewed from a distance. Now suppose you are given the locations and height of all the buildings as shown on a cityscape photo (Figure A),
write a program to output the skyline formed
by these buildings collectively

The geometric information of each building

is represented by a triplet of integers [Li, Ri, Hi],

where Li and Ri are the x coordinates of

the left and right edge of the ith building, respectively,

and Hi is its height.

It is guaranteed that 0 <= Li, Ri <= INT_MAX, 0 < Hi <= INT_MAX,
and Ri - Li > 0. You may assume all buildings are perfect rectangles
grounded on an absolutely flat surface at height 0.

For instance, the dimensions of all buildings in Figure A

are recorded as: [ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] .

The output is a list of "key points" (red dots in Figure B)

in the format of [ [x1,y1], [x2, y2], [x3, y3], ... ]

that uniquely defines a skyline.

A key point is the left endpoint of a horizontal line segment.

Note that the last key point,

where the rightmost building ends,

is merely used to mark the termination of the skyline,

and always has zero height.

Also, the ground in between any two adjacent buildings

should be considered part of the skyline contour.

For instance, the skyline in Figure B should be represented as:

[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ]

Notes:

1. The number of buildings in any input list
is guaranteed to be in the range [0, 10000].
2. The input list is already sorted in ascending order
by the left x position Li.
3. The output list must be sorted by the x position.
4. There must be no consecutive horizontal lines
of equal height in the output skyline.
For instance, [...[2 3], [4 5], [7 5], [11 5], [12 7]...]
is not acceptable; the three lines of height 5 should be merged
into one in the final output as such: [...[2 3], [4 5], [12 7], ...]
*/

class Solution {
public:
    vector<pair<int, int>> getSkyline(vector<vector<int>>& buildings)
    {    
    }
};

//<--> 219. Contains Duplicate II
/*
Given an array of integers and an integer k,

find out whether there are two distinct indices i and j

in the array such that nums[i] = nums[j]

and the absolute difference between i and j is at most k.
*/
class Solution {
public:
    bool containsNearbyDuplicate(vector<int>& nums, int k)
    {
        
    }
};

//<--> 220. Contains Duplicate III
/*
Given an array of integers,
find out whether there are two distinct indices i and j in the array
such that the absolute difference between nums[i] and nums[j]
is at most t and the absolute difference between i and j is at most k.
*/

class Solution {
public:
/*
平衡树的方法。复杂度O(nlogk)

题意有：-t <= x- nums[i] <= t

左边有 nums[i]  – t <= x 因此把符合条件的数构建成一颗平衡树，然后查找一个最小的x使得x>= nums[i] – t

如果该x还满足 x – nums[i] <= t就是我们要的答案啦
*/
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t)
    {
		map<long long,int> m;
        
        int N = nums.size();
        int j = 0;
        
		long long lt = static_cast<long long>(t);
        
        for(int i = 0; i<N; ++i)
        {
            long long n = static_cast<long long>(nums[j]);
            
            if( ( i -j > k ) && ( m.find(n)!=m.end() ) )
            {
                m.erase(n);
                ++j;
            }
            
            n = static_cast<long long>(nums[i]);
            
            auto it = m.lower_bound(n - t);
            
            if( it!=m.end() && (it->first - n) <= t ) //key: using long long to avoid overflow.
            {
                return true;
            }
            
            m[n] = i;
        }
        
        return false;
    }
};

//<--> 221. Maximal Square
/*
Given a 2D binary matrix filled with 0's and 1's, 
find the largest square containing only 1's and return its area.

For example, given the following matrix:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Return 4.
*/

/*
当我们判断以某个点为正方形右下角时最大的正方形时，
那它的上方，左方和左上方三个点也一定是某个正方形的右下角，
否则该点为右下角的正方形最大就是它自己了。这是定性的判断，
那具体的最大正方形边长呢？我们知道，该点为右下角的正方形的最大边长，
最多比它的上方，左方和左上方为右下角的正方形的边长多1，
最好的情况是是它的上方，左方和左上方为右下角的正方形的大小都一样的，
这样加上该点就可以构成一个更大的正方形。但如果它的上方，左方和左上方为右下角的正方形的大小不一样，
合起来就会缺了某个角落，这时候只能取那三个正方形中最小的正方形的边长加1了。
假设dp[i][j]表示以i,j为右下角的正方形的最大边长，则有

dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
当然，如果这个点在原矩阵中本身就是0的话，那dpi肯定就是0了。
*/

class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) 
	{
		if(!matrix.empty() || !matrix[0].empty())
		{
			return 0;
		}
		
		auto & M = matrix;
		
		int rows = M.size();
		int cols = M[0].size();
		
		vector<vector<int>> dp(rows, vector<int>(cols, 0));
		
		int max_sq = 0;
		
		for(int i = 0; i< rows; ++i)
		{
			dp[i][0] = M[i][0] - '0';
			max_sq = max(max_sq, dp[i][0]);
		}
		
		for(int j = 0; j< cols; ++j)
		{
			dp[0][j] = M[0][j] - '0';
			max_sq = max(max_sq, dp[0][j]);
		}
		
		for(int i = 1; i<rows; ++i)
		{
			for(int j = 1; j<cols; ++j)
			{
				if(M[i][j] == '1')
				{
					dp[i][j] = min( dp[i-1][j-1], min( dp[i][j-1], dp[i-1][j] ) ) + 1;  
				}
				
				max_sq = max( dp[i][j], max_sq );
			}
		}
		
		return max_sq * max_sq; //key: max_sq is the side
    }
};

//<--> 222. Count Complete Tree Nodes
/*
Given a complete binary tree, count the number of nodes.

In a complete binary tree every level, 

except possibly the last, is completely filled, 

and all nodes in the last level are as far left as possible.

 It can have between 1 and 2h nodes inclusive at the last level h.
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
    int countNodes(TreeNode* root)
    {
        auto l = root;
        auto r = root;

        int l_height = 0;

        while(l)
        {
            ++l_height;   // key: compute the height of left tree
            l = l->left;
        }

        int r_height = 0;

        while(r)
        {
            ++r_height;   // key: compute the height of right tree.
            r = r->right;
        }

        if(l_height == r_height)
        {
            return 1<<l_height - 1;
        }

        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};

//<--> 223. Rectangle Area
/*
Find the total area covered by two rectilinear rectangles in a 2D plane.

Each rectangle is defined by its bottom left corner and top right corner

Assume that the total area is never beyond the maximum possible value of int.
*/

/*
 *先找出所有的不相交的情况，只有四种，一个矩形在另一个的上下左右四个位置不重叠，
 *这四种情况下返回两个矩形面积之和。其他所有情况下两个矩形是有交集的，
 *这时候我们只要算出长和宽，即可求出交集区域的大小，
 *然后从两个巨型面积之和中减去交集面积就是最终答案。
 *求交集区域的长和宽也不难，由于交集都是在中间，
 *所以横边的左端点是两个矩形左顶点横坐标的较大值，右端点是两个矩形右顶点的较小值，
 *同理，竖边的下端点是两个矩形下顶点纵坐标的较大值，上端点是两个矩形上顶点纵坐标的较小值
 */

class Solution {
public:
    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H)
    {
        int sum = (C-A)*(D-B) + (G-E)*(H-F);
        
        if(A>=G||B>=H||E>=C||F>=D) // key: non-overlap: min > max (A is the min left, G is the max left of another rectangle)
        {
            return sum;
        }
        
        return sum - (min(G,C) - max(A,E))*(min(H,D) - max(B,F)); // min(max, max) - max (min, min);
    }
};

//<--> 224. Basic Calculator
/*
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ),

the plus + or minus sign -, non-negative integers and empty spaces .

You may assume that the given expression is always valid.

Some examples:
"1 + 1" = 2
" 2-1 + 2 " = 3
"(1+(4+5+2)-3)+(6+8)" = 23
*/
class Solution {
public:
    int calculate(string s)
    {
        int sum = 0;
        
        vector<int> sign(2,1); // key: this is a vector for '+' and '-': reason for 2 is that add a additional one to make the arry is not empty
        
        int len = s.size();
        
        for(int i = 0; i<len; ++i)
        {
            auto c = s[i];
            
            if(c>='0')
            {
                int num = 0;
                
                while(i<len && s[i] >= '0')
                {
                    num = 10*num + s[i] - '0';
                    ++i;
                }
                
                sum += sign.back()*num;
                sign.pop_back();
                
                --i; // without --i, we will skip a item since the loop will take ++i after this block.
            }
            else if(c==')')
            {
                //this means the block with parenthesis is completed
                sign.pop_back();
            }
            else if(c!=' ')
            {
                sign.push_back(sign.back() * (c=='-'?=1:1));
            }
        }
        
        return sum;
    }
};

//<--> 225. Implement Stack using Queues
/*
Implement the following operations of a stack using queues.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
empty() -- Return whether the stack is empty.
Notes:
You must use only standard operations of a queue --
which means only push to back, peek/pop from front, size, and is empty operations are valid.
Depending on your language, queue may not be supported natively.
You may simulate a queue by using a list or deque (double-ended queue),
as long as you use only standard operations of a queue.
You may assume that all operations are valid
(for example, no pop or top operations will be called on an empty stack).*
*/
//总共需要两个队列，其中一个队列用来放最后加进来的数，
//模拟栈顶元素。剩下所有的数都按顺序放入另一个队列中。
//当push操作时，将新数字先加入模拟栈顶元素的队列中，
//如果此时队列中有数字，则将原本有的数字放入另一个队中，让
//新数字在这队中，用来模拟栈顶元素。
//当top操作时，如果模拟栈顶的队中有数字则直接返回，
//如果没有则到另一个队列中通过平移数字取出最后一个数字加入模拟栈顶的队列中。
//当pop操作时，先执行下top()操作，保证模拟栈顶的队列中有数字，
//然后再将该数字移除即可。当empty操作时，当两个队列都为空时，栈为空。

//这道题还有另一种解法，可比较好记，只要实现对了push函数，
//后面三个直接调用队列的函数即可。
//这种方法的原理就是每次把新加入的数插到前头，
//这样队列保存的顺序和栈的顺序是相反的，
//它们的取出方式也是反的，那么反反得正，就是我们需要的顺序了。
//我们需要一个辅助队列tmp，把s的元素也逆着顺序存入tmp中，此
//时加入新元素x，再把tmp中的元素存回来，这样就是我们要的顺序了，
//其他三个操作也就直接调用队列的操作即可

class MyStack {
public:
    /** Initialize your data structure here. */
    MyStack() {
        
    }
    
    /** Push element x onto stack. */
    void push(int x) {
        
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        
    }
    
    /** Get the top element. */
    int top() {
        
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
        
    }
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * bool param_4 = obj.empty();
 */

// <--> 226. Invert Binary Tree

/*
 Invert a binary tree.

     4
   /   \
  2     7
 / \   / \
1   3 6   9

to
     4
   /   \
  7     2
 / \   / \
9   6 3   1

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
    TreeNode* invertTree(TreeNode* root)
    {
        if(!root)
        {
            return root;
        }
        
        swap(root->left, root->right);
        
        invertTree(root->left);
        invertTree(root->right);
        
        return root;
    }
};

//<--> 227. Basic Calculator II
/*
Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers,

+, -, *, / operators and empty spaces . The integer division should truncate toward zero.

You may assume that the given expression is always valid.

Some examples:
"3+2*2" = 7
" 3/2 " = 1
" 3+5 / 2 " = 5
*/
class Solution {
public:
    int calculate(string s)
    {    
    }
};

//<--> 228. Summary Ranges
/*
Given a sorted integer array without duplicates,

return the summary of its ranges.

For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
*/

class Solution {
public:
    vector<string> summaryRanges(vector<int>& nums)
    {
        int i = 0;
        int n = nums.size();
        
        vector<string> res;
        
        while(i<n)
        {
            int j = 1;
            
            while(i+j<n && nums[i+j] - nums[i] == j)
            {
                ++j;
            }
            
            if(j == 1)
            {
                res.push_back(to_string(nums[i]));
            }
            else
            {
                res.push_back(to_string(nums[i]) + "->" + to_string(nums[i+j-1]));
            }
            
            i += j;
        }
        
        return res;
    }
};

//<--> 229. Majority Element II
/*
Given an integer array of size n,

find all elements that appear more than ⌊ n/3 ⌋ times.

The algorithm should run in linear time and in O(1) space.
*/

//任意一个数组出现次数大于n/3的众数最多有两个，
//那么有了这个信息，我们使用投票法的核心是找出两个候选众数进行投票，
//需要两遍遍历，第一遍历找出两个候选众数，
//第二遍遍历重新投票验证这两个候选众数是否为众数即可，
//这道题却没有这种限定，即满足要求的众数可能不存在，所以要有验证

class Solution {
public:
    vector<int> majorityElement(vector<int>& nums)
    {
        vector<int> res;
        
        int m1 = 0, m2 = 0, count_m1 = 0, count_m2 = 0;
        
        for( auto n : nums )
        {
            if(n == m1)
            {
                ++count_m1;
            }
            else if(n == m2)
            {
                ++count_m2;
            }
            else if(count_m1==0)
            {
                m1 = n;
                count_m1 = 1;
            }
            else if(count_m2==0)
            {
                m2 = n;
                count_m2 = 1;
            }
            else
            {
                --count_m1;
                --count_m2;
            }
        }
        
        count_m1 = 0;
        count_m2 = 0;
        
        for(auto n : nums)
        {
            if(n==m1)
            {
                ++count_m1;
            }
            else if(n==m2)
            {
                ++count_m2;
            }
        }
        
        int len = nums.size();
        
        if(count_m1 > len / 3)
        {
            res.push_back(m1);
        }
        
        if( count_m2 > len / 3 )
        {
            res.push_back(m2);
        }
        
        return res;
    }
};

//<--> 230. Kth Smallest Element in a BST
/*
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Follow up:
What if the BST is modified (insert/delete operations)
often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?
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
    
    // non recursive: inorder traverse will give the sorted result
    int kthSmallest(TreeNode* root, int k)
    {
        stack<TreeNode*> s;
        auto p = root;
        
        int count = 0;
        
        while(p || s.empty())
        {
            while(p)
            {
                s.push(p);
                p = p->left;
            }
            
            p = s.top();
            s.pop();
            
            ++count;
            
            if(count == k)
            {
                return p->val;
            }
            
            p = p->right;
        }
    }
    
    //recursive
    
    int kthSmallest(TreeNode* root, int k)
    {
        int count = 0;
        return inorder(root, k, count);
    }
    
    int inorder(TreeNode* root, int k, int& count)
    {
        if(!root)
        {
            return -1;
        }
        
       
        int val = inorder(root->left, k, count );
        
        if(k == count) // key: this could happened in left tree.
        {
            return val;
        }
        
        ++count; //visit root 
        
        if(count == k)
        {
            return root->val;
        }

        return inorder(root->right, k, count);
        
    }
};


//<--> 231. Power of Two
/*
Given an integer, write a function to determine if it is a power of two.
*/

class Solution {
public:
    bool isPowerOfTwo(int n) {
        
        return (n > 0 ) && ( (n & ( n - 1 ) ) == 0 ); //key: n& (n-1) will remove a bit 1 from n;
         
    }
};

//<--> 232. Implement Queue using Stacks
/*
Implement the following operations of a queue using stacks.

push(x) -- Push element x to the back of queue.
pop() -- Removes the element from in front of queue.
peek() -- Get the front element.
empty() -- Return whether the queue is empty.

Notes:
1. You must use only standard operations of a stack --
which means only push to top, peek/pop from top, size, and is empty operations are valid.
2. Depending on your language, stack may not be supported natively.
You may simulate a stack by using a list or deque (double-ended queue),
as long as you use only standard operations of a stack.
3. You may assume that all operations are valid
(for example, no pop or peek operations will be called on an empty queue).

// Method 1 (By making push operation costly)

class MyQueue {
public:
    /** Initialize your data structure here. */
    MyQueue() {
        
        
    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        
        stack<int> tmp;
        while(!s.empty())
        {
            tmp.push(s.top());
            s.pop();
        }
        
        s.push(x);
        while(!tmp.empty())
        {
            s.push(tmp.top());
            tmp.pop();
        }
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        
        s.pop();
    }
    
    /** Get the front element. */
    int peek() {
        
        return s.top();
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        
        return !s.empty();
    }
    
    private:
        stack<int> s;
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * bool param_4 = obj.empty();
 */


// <--> 233. Number of Digit One
/*
Given an integer n, count the total number of digit 1 appearing

in all non-negative integers less than or equal to n.

For example:
Given n = 13,
Return 6, because digit 1 occurred in the following numbers: 1, 10, 11, 12, 13.
*/
class Solution {
public:
    int countDigitOne(int n) {
        int res = 0;
        int a = 1;
        int b = 1;
        
        while(n > 0)
        {
            int q = n / 10;
            int r = n - q * 10;
            
            res += (n+8)/10;
            res += (r==1) ? b : 0;
            
            b += (r) * a;
            a *= 10;
            
            n = q;
        }
    }
};

//<--> 234. Palindrome Linked List
/*
Given a singly linked list, determine if it is a palindrome.

Follow up:
Could you do it in O(n) time and O(1) space?
*/
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
//using fast->next && fast->next->next, the slow will be at the position = length/2
//using fast && fast->next, the slow will be at the position = (length+1)/2
class Solution {
public:
    bool isPalindrome(ListNode* head)
    {
        if(!head || !head->next)
        {
            return true;
        }
        
        auto slow = head;
        auto fast = head;
        
        while(fast->next && fast->next->next)
        {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        //key: reverse from slow->next to end;
        auto last = slow->next;
        while(last->next)
        {
            auto tmp = last->next;
            last->next = tmp->next;
            tmp->next = slow->next;
            slow->next = tmp;
        }
        
        auto pre = head;
        while(slow->next)
        {
            slow = slow->next;
            if(head->val != slow->val)
            {
                return false;
            }
            
            pre = pre->next;
        }
        
    }
};

//<--> 235. Lowest Common Ancestor of a Binary Search Tree
/*
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia:

“The lowest common ancestor is defined between two nodes v and w
as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

        _______6______
       /              \
    ___2__          ___8__
   /      \        /      \
   0      _4       7       9
         /  \
         3   5
For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6.
Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
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
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
    {
        if(!root)
        {
            return root;
        }
        
        if(root->val > max(p->val, q->val))
        {
            return lowestCommonAncestor(root->left, p, q);
        }
        
        if(root->val < min(p->val, q->val))
        {
            return lowestCommonAncestor(root->right, p, q);
        }
        
        return root;
};

//<--> 236. Lowest Common Ancestor of a Binary Tree
/*
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia:

“The lowest common ancestor is defined between

two nodes v and w as the lowest node in T that has both v and w as descendants

(where we allow a node to be a descendant of itself).”

        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3.

Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
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
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
    {
        if( !root ||  root == p | root== q )
        {
            return root;
        }
        
        auto left = lowestCommonAncestor(root->left, p, q);
        if(left && ( left != p && left != q ))
        {
            return left;
        }
        
        auto right = lowestCommonAncestor(root->right, p, q);
        if(right && ( right != p && right != q ))
        {
            return right;
        }
        
        if(left && right)
        {
            return root;
        }
        
        return left ? left : right;
    }
};

//<--> 237. Delete Node in a Linked List
/*
Write a function to delete a node (except the tail)
in a singly linked list, given only access to that node.

Supposed the linked list is 1 -> 2 -> 3 -> 4 and
you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.
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
    void deleteNode(ListNode* node)
    {
        auto next = node->next;
        
        if(next)
        {
            node->val = next->val;
        }
        
        node->next = next->next;
        next->next = nullptr;
        
        delete next;
    }
};

//<--> 238. Product of Array Except Self
/*
Given an array of n integers where n > 1, nums,

return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Solve it without division and in O(n).

For example, given [1,2,3,4], return [24,12,8,6].

Follow up:
Could you solve it with constant space complexity?

(Note: The output array does not count as extra space for the purpose of space complexity analysis.)
*/
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums)
    {
        int len = nums.size();
        
        vector<int> result( len , 1 );
        
        auto& A = nums;
        
        for(int i = 1; i < len; ++i)
        {
            result[i] = A[i-1] * result[i-1];
        }
        
        int right_product = 1;
        for(int i = len - 2; i>=0; --i)
        {
            right_product *= A[i+1];
            result[i] *= right_product;
        }
        
        return result;
    }
};

//<--> 239. Sliding Window Maximum
/*
Given an array nums, there is a sliding window of size k
which is moving from the very left of the array to the very right.
You can only see the k numbers in the window. Each time the sliding window moves right by one position.

For example,
Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
Therefore, return the max sliding window as [3,3,5,5,6,7].

Note: 
You may assume k is always valid, ie: 1 ≤ k ≤ input array's size for non-empty array.

Follow up:
Could you solve it in linear time?

Hint:

1. How about using a data structure such as deque (double-ended queue)?
2. The queue size need not be the same as the window’s size.
3. Remove redundant elements and the queue should store only elements that need to be considered.
*/
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k)
    {
        deque<int> q;
        
        vector<int> res;
        
        int len = nums.size();
        int i = 0;
        
        for(i = 0; i< k; ++i)
        {
            q.push_back(i);
        }
        
        while( i< len )
        {
            if(!q.empty() && nums[q.back()] < nums[i])
            {
                
            }
        }
        
        for(int i = 0; i < len; ++i)
        {
            if(!q.empty() && )
        }
    }
};

//<--> 240. Search a 2D Matrix II
/*
Write an efficient algorithm that searches for a value in an m x n matrix.

This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
For example,

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.
*/
/*
 *如果我们观察题目中给的那个例子，
 *我们可以发现有两个位置的数字很有特点，
 *左下角和右上角的数。左下角的18，
 *往上所有的数变小，往右所有数增加，
 *那么我们就可以和目标数相比较，
 *如果目标数大，就往右搜，
 *如果目标数小，就往top搜。
 *这样就可以判断目标数是否存在。
 *当然我们也可以把起始数放在右上角，
 *往左和下搜，停止条件设置正确就行。
 */
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target)
    {
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        if(target < matrix[0][0] || target > matrix.back().back())
        {
            return false;
        }
        
        int x = rows - 1;
        int y = 0;
        
        while(x >= 0 && y <= cols-1)
        {
            if(matrix[x][y] == target)
            {
                return true;
            }
            
            if(matrix[x][y] > target)
            {
                --x;
            }
            else if(matrix[x][y]<target)
            {
                ++y;
            }
        }
        
        return false;
    }
};

//<--> 241. Different Ways to Add Parentheses
/*
Given a string of numbers and operators,

return all possible results from computing all the different possible ways

to group numbers and operators. The valid operators are +, - and *.


Example 1
Input: "2-1-1".

((2-1)-1) = 0
(2-(1-1)) = 2
Output: [0, 2]


Example 2
Input: "2*3-4*5"

(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10
Output: [-34, -14, -10, -10, 10]
*/
class Solution {
public:
    vector<int> diffWaysToCompute(string input)
    {
        vector<int> res;
        dfs(input, res);
    }
    
    void dfs(string&& input, vector<int>& out)
    {
        for(size_t i = 0; i < input.size(); ++i)
        {
            if( ( input[i]! = '+' ) && ( input[i] != '-' ) && ( input[i] != '*') )
            {
                continue;
            }
            
            vector<int> left;
            vector<int> right;
            
            dfs(input.substr(0,i), left);
            dfs(input.substr(i+1), right);
            
            for(size_t l = 0; l<left.size(); ++l)
            {
                for(size_t r = 0; r < right.size(); ++r)
                {
                    switch(input[i])
                    {
                        case '+':
                            out.push_back(left[l] + right[r]);
                            break;
                        case '-':
                            out.push_back(left[l] - right[r]);
                            break;
                        case '*':
                            out.push_back(left[l] * right[r]);
                            break;
                    } 
                } //end for(r)
            }//end for(l)
        } // end for(i)
        
        if(out.empty())
        {
            out.push_back(stoi(input));
        }
    }
};

//<--> 242. Valid Anagram
/*
Given two strings s and t, write a function
to determine if t is an anagram of s.

For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.

Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters?
How would you adapt your solution to such case?
*/
class Solution {
public:
    bool isAnagram(string s, string t)
    {
    }
};

//<--> 243. Shortest Word Distance
/*
Given a list of words and two words word1 and word2,

return the shortest distance between these two words in the list.

For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Given word1 = “coding”, word2 = “practice”, return 3.
Given word1 = "makes", word2 = "coding", return 1.

Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.
*/
class Solution {
public:
    int shortestDistance(vector<string>& words, string word1, string word2)
    {
        int len = words.size();
        int last_pos = -1;
        int res = len;
        
        for(int i = 0; i < len; ++i)
        {
            auto &w = words[i];
            
            if(w==word1 || w==word2)
            {
                if( ( pos != -1 ) && ( words[pos] != words[i] ) )
                {
                    res = min( res, abs( i - last_pos ) );
                }
                
                last_pos = i;
            }
        }
        
        return res;
    }
};

//<--> 244. Shortest Word Distance II
/*
This is a follow up of Shortest Word Distance.

The only difference is now you are given the list of words

and your method will be called repeatedly
many times with different parameters. How would you optimize it?

Design a class which receives a list of words in the constructor,

and implements a method that takes two words word1 and word2 and

return the shortest distance between these two words in the list.

For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Given word1 = “coding”, word2 = “practice”, return 3.
Given word1 = "makes", word2 = "coding", return 1.

Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.
*/
class WordDistance {
public:
    WordDistance(vector<string>& words)
    {
        int i = 0;
        
        for(auto& w : words)
        {
            if(m.find(w)==m.end())
            {
                m.emplace(w, vector<int>());
            }
            
            m[w].push_back(i);
        }
    }
    
    int shortest(string word1, string word2)
    {
        int i = 0;
        int j = 0;
        
        int len1 = word1.size();
        int len2 = word2.size();
        
        int min_dist = INT_MAX;
        
        while( ( i < len1 ) && ( j < len2 ) )
        {
            int pos1 = m[word1][i];
            int pos2 = m[word2][j];
            
            min_dist = min(min_dist, abs(pos1-pos2));
            pos1 > pos2 ? ++i : ++j;
        }
        
        return min_dist;
    }
    
    private:
        
        unordered_map<string, vector<int>> m;
};

//<--> 245. Shortest Word Distance III
/*
This is a follow up of Shortest Word Distance.

The only difference is now word1 could be the same as word2.

Given a list of words and two words word1 and word2,

return the shortest distance between these two words in the list.

word1 and word2 may be the same and they represent two individual words in the list.

For example,
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Given word1 = “makes”, word2 = “coding”, return 1.
Given word1 = "makes", word2 = "makes", return 3.

Note:
You may assume word1 and word2 are both in the list.
*/
class Solution {
public:
    int shortestWordDistance(vector<string>& words, string word1, string word2)
    {
    }
};

//<--> 246. Strobogrammatic Number
/*
A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Write a function to determine if a number is strobogrammatic. The number is represented as a string.

For example, the numbers "69", "88", and "818" are all strobogrammatic.
*/
class Solution {
public:
    bool isStrobogrammatic(string num)
    {
    }
};

// <--> 247. Strobogrammatic Number II
/*
A strobogrammatic number is a number that

looks the same when rotated 180 degrees (looked at upside down).

Find all strobogrammatic numbers that are of length = n.

For example,
Given n = 2, return ["11","69","88","96"].

Hint:

Try to use recursion and notice that it should recurse with n - 2 instead of n - 1.
*/
class Solution {
public:
    vector<string> findStrobogrammatic(int n)
    {
        vector<string> last;
        vector<string> res;
        
        dfs(n-2, last); //key: since we add two numbers at begin and end of previous one, the length must be 2 less
        
        for(const auto& s : last)
        {
            //key: the final one should not have zero at begin.
            out.emplace_back("1"+elem+"1");
            out.emplace_back("8"+elem+"8");
            out.emplace_back("6"+elem+"9");
            out.emplace_back("9"+elem+"9");
        }
    }
    
    void dfs(int m, vector<string>& out)
    {
        if(m==0)
        {
            out.push_back("");
            return;
        }
        
        if(m==1)
        {
            out.emplace_back("0");
            out.emplace_back("1");
            out.emplace_back("8");
            return;
        }
        
        auto last_level = dfs(m-2);
        
        vector<string> out;
        
        for(const auto& elem : last_level)
        {
            out.emplace_back("0"+elem+"0");
            out.emplace_back("1"+elem+"1");
            out.emplace_back("8"+elem+"8");
            out.emplace_back("6"+elem+"9");
            out.emplace_back("9"+elem+"6");
        }
    }
};

//<--> 248. Strobogrammatic Number III
/*
A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Write a function to count the total strobogrammatic numbers that exist

in the range of low <= num <= high.

For example,
Given low = "50", high = "100", return 3. Because 69, 88, and 96 are three strobogrammatic numbers.

Note:
Because the range might be a large number, the low and high numbers are represented as string
*/
class Solution {
public:
    int strobogrammaticInRange(string low, string high)
    {
        int res = 0;
        
        dfs(low, high, std::move(""), res);
        dfs(low, high, std::move("0"), res);
        dfs(low, high, std::move("1"), res);
        dfs(low, high, std::move("8"), res);
        
        return res;
    }
    
    void dfs(const string& low, const string& high, string&& out, int& count)
    {
        if(out.size()>=low.size() && out.size()<=high.size())
        {
            // key: since this is string comparison, it makes sense when the two strings have same size.
            if( ( out.size()==low.size() && out < low) ) || ( out.size() == high.size() && out > high ) )
            {
                return;
            }
            
            //key: ignore the number with zero at beginning.
            if( out.size() > 1 && out[0] == '0' )
            {
                return;
            }
            
            ++count;
        }
        
        if(out.size()+2 > high.size())
        {
            //since we will add two more characters, check if the added string size is larger than the high.
            return;
        }
        
        dfs(low, high, std::move("1"+out+"1"));
        dfs(low, high, std::move("8"+out+"8"));
        dfs(low, high, std::move("6"+out+"9"));
        dfs(low, high, std::move("9"+out+"6"));
    }
};

//<--> 249. Group Shifted String
/*
Given a string, we can "shift" each of its letter to its successive letter,

for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:

"abc" -> "bcd" -> ... -> "xyz"
Given a list of strings which contains only lowercase alphabets,

group all strings that belong to the same shifting sequence.

For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"], 
Return:

[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

Note: For the return value, each inner list's elements must follow the lexicographic order.
*/
class Solution {
public:
    vector<vector<string>> groupStrings(vector<string>& strings)
    {
    }
};

//<--> 250. Count Univalue Subtrees
/*
Given a binary tree, count the number of uni-value subtrees.

A Uni-value subtree means all nodes of the subtree have the same value.

For example:
Given binary tree,

              5
             / \
            1   5
           / \   \
          5   5   5
 

return 4.
*/
class Solution {
public:
    
    //recursive method
    int countUnivalSubtrees(TreeNode* root)
    {
        int res = 0;
        
        //key: since val in function dfs is the value of the parent of root, so at start,
        //this value can be a arbitrary value.
        dfs(root->val, -1, res); 
        
        return res;
    }
    
    bool dfs(TreeNode* root, int val, int& count)
    {
        if(!root)
        {
            return true;
        }
        
        if( ( !dfs(root->left, root->val, count) ) || ( !dfs(root->right, root->val, count) ) )
        {
            return false;
        }
        
        ++count; //key: this is a univalue tree with root = root;
        
        return root->val == val;
    }
    
    //using postorder
    int countUnivalSubtrees(TreeNode* root)
    {
        if(!root)
        {
            return 0;
        }
        
        set<TreeNode*> trees;
        stack<TreeNode*> s;
        
        s.push(root);
        auto head = root;
        
        while(!s.empty())
        {
            auto t = s.top();
            
            if( ( !t->left && !t->right ) || ( t->left == head ) || ( t->right == head ) )
            {
                //BEGIN: the parts do counting
                helper(t, trees);
                //END: the parts do counting
                s.pop();
                head = t;
            }
            else
            {
                if(t->right)
                {
                    s.push(t->right);
                }
                
                if(t->left)
                {
                    s.push(t->left);
                }
            }
        }
    }
    
    void helper( TreeNode* t, set<TreeNode*>& s )
    {
        if(!t->left && !t->right)
        {
            s.insert(t);
            return;
        }
        
        if(!t->left && s.count(t->right) == 1 && t->right->val == t->val)
        {
            s.insert(t);
            return;
        }
        
        if(!t->right && s.count(t->left) == 1 && t->left->val == t->val)
        {
            s.insert(t);
            return;
        }
        
        if(t->left && t->right)
        {
            if(s.count(t->left) == 1 && s.count(t->right) == 1)
            {
                if(t->left->val == t->val && t->right->val == val)
                {
                    s.insert(t);
                }
            }
        }
    }
    
};

//<--> 251. Flatten 2D Vector
/*
Implement an iterator to flatten a 2d vector.

For example,
Given 2d vector =

[
  [1,2],
  [3],
  [4,5,6]
]
 

By calling next repeatedly until hasNext returns false,
the order of elements returned by next should be: [1,2,3,4,5,6].

Hint:

1. How many variables do you need to keep track?
2. Two variables is all you need. Try with x and y.
3. Beware of empty rows. It could be the first few rows.
4. To write correct code, think about the invariant to maintain. What is it?
The invariant is x and y must always point to a valid point in the 2d vector.
Should you maintain your invariant ahead of time or right when you need it?
Not sure? Think about how you would implement hasNext(). Which is more complex?
5. Common logic in two different places should be refactored into a common method.
Follow up:
As an added challenge, try to code it using only iterators in C++ or iterators in Java.
*/
class Vector2D {
public:
    Vector2D(vector<vector<int>>& vec2d)
    {
    }
    
    int next()
    {
        
    }
    
    bool hasNext()
    {
        
    }
    
    private:
        
};

//<-> 252. Meeting Rooms
/*
Given an array of meeting time intervals
consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),
determine if a person could attend all meetings.
For example,
Given [[0, 30],[5, 10],[15, 20]],
return false.
*/
class Solution {
public:
    bool canAttendMeetings(vector<Interval>& intervals)
    {
    }
};

//<-> 253. Meeting Rooms II
/*
Given an array of meeting time intervals

consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei),

find the minimum number of conference rooms required.

For example,
Given [[0, 30],[5, 10],[15, 20]],
return 2.
*/

class Solution {
public:
	//using map.
    int minMeetingRooms(vector<Interval>& intervals)
    {
		map<int,int> m;
		
		for( const auto& itv : intervals )
		{
			if(m.find(itv.start)!=m.end())
			{
				m.emplace(itv.start, 1);
			}
			else
			{
				++m[itv.start];
			}
			
			if(m.find(itv.end()!=m.end())
			{
				m.emplace(itv.end, -1);
			}
			else
			{
				--m[itv.end];
			}
		}
		
		int count = 0;
		
		int rooms = 0;
		
		for(const auto& p : m)
		{
			rooms += p.second;
			count = max(count, rooms);
		}
		
		return count;
    }
	
	//using min heap
	int minMeetingRooms(vector<Interval>& intervals)
	{
		priority_queue<int, vector<int>, greater<int>> pq;
				
		for( const auto& itv : intervals )
		{
			// key: if the top of pq (which is the minimum end time of previous intervals ) is no larger than current start,
			// these two intervals will using same meeting room because they are not overlapped.
			if( !pq.empty() && pq.top() <= itv.start )
			{
				pq.pop();
			}
			
			pq.push(itv.end);
		}
		
		return pq.size();
	}
};

//<--> 254. Factor Combinations
/*
Numbers can be regarded as product of its factors. For example,

8 = 2 x 2 x 2;
  = 2 x 4.
  
Write a function that takes an integer n and return all possible combinations of its factors.

Note: 

Each combination's factors must be sorted ascending, 

for example: The factors of 2 and 6 is [2, 6], not [6, 2].

You may assume that n is always positive.

Factors should be greater than 1 and less than n.
 

Examples: 
input: 1
output: 

[]
input: 37
output: 

[]
input: 12
output:

[
  [2, 6],
  [2, 2, 3],
  [3, 4]
]

input: 32
output:

[
  [2, 16],
  [2, 2, 8],
  [2, 2, 2, 4],
  [2, 2, 2, 2, 2],
  [2, 4, 4],
  [4, 8]
]
*/
class Solution {
public:

	//method 1: may produce different orders
    vector<vector<int>> getFactors(int n)
	{
		vector<vector<int>> res;
		vector<int> v;
		
		dfs(n, 2, res, v);
		
		return res;
	}
	
	void dfs(int m, int start, vector<vector<int>>& res, vector<int>& out)
	{
		for(int i = start; i*i<=m; ++i)
		{
			if(m%i == 0)
			{
				out.push_back(i);
				dfs(m/i, i, res, out);
				
				res.emplace_back(out);
				res.back().push_back(m/i);
				
				out.pop_back();
				
			}
		}
	}
	
	//method 2: Produce same result as the given in the problem
	vector<vector<int>> getFactors(int n)
	{
		vector<vector<int>> res;
		
		for(int i = 2; i*i<=n; ++i)
		{
			if(n%i==0)
			{
				auto factors = getFactors(n/i);
				
				res.emplace_back(vector<int>{i, n/i}); //key: create a vector contains i and n/i and added to the result.
				
				for( auto& vf : factors )
				{
					if(i <= vf[0])
					{
						vf.insert(begin(vf),i); //key: put i at the beginning.
						res.emplace_back(vf);
					}
				}
			}
		}
		
		return res;
	}
};

//<-->255. Verify Preorder Sequence in Binary Search Tree
/*
Given an array of numbers, 

verify whether it is the correct preorder traversal sequence of a binary search tree.

You may assume each number in the sequence is unique.

Follow up:
Could you do it using only constant space complexity?
*/
class Solution {
public:
	//method 1: using stack
    bool verifyPreorder(vector<int>& preorder)
	{
		stack<int> s;
		
		int low = INT_MIN;
		
		for(auto n : preorder)
		{
			if(n<low)
			{
				return false;
			}
			
			while( !s.empty() && s.top() < n )
			{
				low = s.top();
				s.pop();
			}
			
			s.push(n);
		}
		
		return true;
	}
	
	//method 2: O(1) memory space
	bool verifyPreorder(vector<int>& preorder)
	{
		int low = INT_MIN;
		
		int i = -1;
		
		for(auto n : preorder)
		{
			if(n<low)
			{
				return false;
			}
			
			while( i>=0 && n > preorder[i] )
			{
				low = preorder[i--];
			}
			
			preorder[++i] = n;
		}
		
		return true;
	}
};

//<-->256. Paint House
/*
There are a row of n houses, each house can be painted with one of the three colors:

red, blue or green. The cost of painting each house with a certain color is different.

You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a n x 3 cost matrix.

For example, costs[0][0] is the cost of painting house 0 with color red;

costs[1][2] is the cost of painting house 1 with color green,

and so on... Find the minimum cost to paint all houses.

Note:
All costs are positive integers.
*/
class Solution {
public:
    int minCost(vector<vector<int>>& costs)
    {
        if( costs.empty()||costs[0].empty() )
        {
            return 0;
        }
        
        auto dp = costs;
        
        for(size_t i = 1; i < dp.size(); ++i)
        {
            dp[i][0] += min(dp[i-1][1], dp[i-1][2]);
            dp[i][1] += min(dp[i-1][0], dp[i-1][2]);
            dp[i][2] += min(dp[i-1][0], dp[i-1][1]);
        }
        
        return min( min(dp.back()[0], dp.back()[1]), dp.back()[2] );
    }
};

//<-->257. Binary Tree Paths
/*
Given a binary tree, return all root-to-leaf paths.

For example, given the following binary tree:

   1
 /   \
2     3
 \
  5
All root-to-leaf paths are:

["1->2->5", "1->3"]
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
    //recursive method 1: using a standalone method
    vector<string> binaryTreePaths(TreeNode* root)
    {
        vector<string> res;
        
        if(root)
        {
            dfs(root, "", res);
        }
        
        return res;
    }
    
    void dfs(TreeNode* root, string out, vector<string>& res)
    {
        out += to_string(root->val);
        if(!root->left && !root->right)
        {
            res.emplace_back(out.c_str());
        }
        else
        {
            if(root->left)
            {
                dfs(root->left, out+"->", res);
            }
            
            if(root->right)
            {
                 dfs(root->right, out+"->", res);
            }
        }
    }
};


//<-->259. 3Sum Smaller
/*
Given an array of n integers nums and a target,

find the number of index triplets i, j, k with 0 <= i < j < k < n

that satisfy the condition nums[i] + nums[j] + nums[k] < target.

For example, given nums = [-2, 0, 1, 3], and target = 2.

Return 2. Because there are two triplets which sums are less than 2:

[-2, 0, 1]
[-2, 0, 3]
Follow up:
Could you solve it in O(n^2) runtime?
*/
class Solution {
public:
    int threeSumSmaller(vector<int>& nums, int target)
    {
        if(nums.size()<3)
        {
            return 0;
        }
        
        int res = 0;
        
        sort(begin(nums), end(nums));
        
        for(size_t i = 0; i<nums.size()-2; ++i)
        {
            size_t left = i+1, right = nums.size() - 1;
            
            while(left<right)
            {
                if(nums[i]+nums[left]+nums[right]<target)
                {
                    res += (right - left); // key: the elements between left and right are all satisfied.
                    ++left;
                }
                else
                {
                    --right;
                }
            }
        }
        
        return res;
    }
};

//<-->260. Single Number III
/*
Given an array of numbers nums,
in which exactly two elements appear only once and
all the other elements appear exactly twice.
Find the two elements that appear only once.

For example:

Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].

Note:
1. The order of the result is not important.
So in the above example, [5, 3] is also correct.
2. Your algorithm should run in linear runtime complexity.
Could you implement it using only constant space complexity?
*/
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums)
    {
        int diff = accumulate(begin(nums), end(nums), bit_xor<int>());
        diff &= -diff;
        
        vector<int> res(2,0);
        
        for(auto& a : num)
        {
            if(a& diff)
            {
                res[0] ^= a;
            }
            else
            {
                res[1] ^= a;
            }
        }
        
        return res;
    }
};

//<-->261. [LeetCode] Graph Valid Tree
/*

Given n nodes labeled from 0 to n - 1 and

a list of undirected edges (each edge is a pair of nodes),

write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.

Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.

Hint:

Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]],
what should your return? Is this case a valid tree?
According to the definition of tree on Wikipedia:
“a tree is an undirected graph in which any two vertices are connected
by exactly one path. In other words,
any connected graph without simple cycles is a tree.”

Note: you can assume that no duplicate edges will appear in edges.
Since all edges are undirected,
[0, 1] is the same as [1, 0] and
thus will not appear together in edges.
*/
class Solution {
public:
    
    //method 1: dfs
    bool validTree(int n, vector<pair<int, int>>& edges)
    {
        vector<vector<int>> g(n, vector<int>());
        vector<int> v(n, 0);
        
        for(auto& edge : edges)
        {
            g[edge.first].push_back(edge.second);
            g[edge.second].push_back(edge.first);
        }
        
        if(!dfs(g,v,0,-1))
        {
            return false;
        }
        
        for(auto a : v)
        {
            if( a == 0 )
            {
                return false;
            }
        }
    }
    
    bool dfs(vector<vector<int>>& g, vector<int>& v, int cur, int pre)
    {
        if(v[cur] == 1) //key: this node is visited
        {
            return false;
        }
        
        v[cur] = 1;
        
        for( auto& node : g[cur] )
        {
            if( node != pre )
            {
                if( !dfs( g, v, node, cur ) )
                {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    //method 2: bfs
    bool validTree(int n, vector<pair<int, int>>& edges)
    {
        vector<unordered_set<int>> g(n, unordered_set<int>());
        unordered_set<int> v(n, 0);
        
        for( auto e : edges )
        {
            g[e.first].insert(e.second);
            g[e.second].insert(e.first);
        }
        
        queue<int> q;
        
        q.push(0);
        
        v.insert(0);
        
        while(!q.empty())
        {
            auto f = q.front();
            q.pop();
            
            for(auto n : g[f])
            {
                if( v.find(n) != v.end() )
                {
                    return false;
                }
                
                v.insert(n);
                q.push(n);
                
                g[n].erase(f);
            }
        }
        
        return v.size() == n;
    }
    
    //method 3: union find
    bool validTree(int n, vector<pair<int, int>>& edges)
    {
        vector<int> roots(n , -1);
        for(auto e : edges)
        {
            int x = find(roots, a.first);
            int y = find(roots, a.second);
            if(x==y)
            {
                return false;
            }
            
            roots[x] = y;
        }
        
        return edges.size() == n - 1;
    }
    
    int find(vector<int>& roots, int i)
    {
        while(roots[i] != -1)
        {
            i = roots[i];
        }
        
        return i;
    }
};

//<-->263. Ugly Number
/*
Write a program to check whether a given number is an ugly number.

Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.

Note that 1 is typically treated as an ugly number.
*/
class Solution {
public:
    bool isUgly(int num)
    {
        if(num < = 0)
        {
            return false;
        }
        
        while(n%2==0)
        {
            n /= 2;
        }
        
        while(n%3==0)
        {
            n /= 3;
        }
        
        while(n%5 == 0)
        {
            n /= 5;
        }
        
        return n == 1;
    }
};

//<--> 264. Ugly Number II
/*
Write a program to find the n-th ugly number.

Ugly numbers are positive numbers whose prime factors only

include 2, 3, 5. For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12

is the sequence of the first 10 ugly numbers.

Note that 1 is typically treated as an ugly number,

and n does not exceed 1690.
*/
class Solution {
public:
    // dynamic programming
    int nthUglyNumber(int n)
    {
        vector<int> dp(n,1);
        
        int u2 = 1, u3 = 1, u5 = 1;
        int i2 = 0, i3 = 0, i5 = 0;

		for ( int i = 1; i < n; ++i )
		{
			u2 = dp[i2] * 2;
			u3 = dp[i3] * 3;
			u5 = dp[i5] * 5;

			dp[i] = min( u2, min( u3, u5 ) );

			if ( dp[i] == u2 )
			{
				++i2;
			}

			if ( dp[i] == u3 )
			{
				++i3;
			}

			if ( dp[i] == u5 )
			{
				++i5;
			}
		}

		return dp.back();
    }
};

//<-->265. Paint House II
 /*
There are a row of n houses,

each house can be painted with one of the k colors.

The cost of painting each house with a certain color is different.

You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by a n x k cost matrix.

For example, costs[0][0] is the cost of painting house 0 with color 0;

costs[1][2]is the cost of painting house 1 with color 2, and so on... Find the minimum cost to paint all houses.

Note:
All costs are positive integers.

Follow up:
Could you solve it in O(nk) runtime?
*/
 class Solution {
public:
    // using O(nk) memory space
    int minCostII(vector<vector<int>>& costs)
    {
        if(costs.empty() || costs[0].empty())
        {
            return 0;
        }
        
        vector<vector<int>> dp = costs;
        
        int min1=-1, min2=-1;
        
        int last1=-1,last2=-1;
        
		for ( size_t i = 0; i < costs.size(); ++i )
		{
			last1 = min1;
			last2 = min2;
			min1 = -1;
			min2 = -1;

			//iterate over all colors
			for ( size_t j = 0; j < costs[i].size(); ++j )
			{
				if ( j != last1 )
				{
					dp[i][j] += last1 < 0 ? 0 : dp[i - 1][last1];
				}
				else
				{
					dp[i][j] += last2 < 0 ? 0 : dp[i - 1][last2]; //key: add second to minimum 
				}

				if ( min1 < 0 || dp[i][j] < dp[i][min1] ) // update min1: the color to get miniumum cost
				{
					min2 = min1;
					min1 = j;
				}
				else if ( min2 < 0 || dp[i][j] < dp[i][min2] )
				{
					min2 = j;
				}
			}
		}
        
        return dp.back()[min1];
    }
    
    // using O(1) memory space
	int minCostII( vector<vector<int>>& costs )
	{
		if ( costs.empty() || costs[0].empty() )
		{
			return 0;
		}

		int min1_so_far = 0;
		int min2_so_far = 0;
		int min1_color = -1;

		for ( size_t i = 0; i < costs.size(); ++i )
		{
			int cur_min1 = INT_MAX;
			int cur_min2 = cur_min1;
			int cur_min1_color = -1;

			for ( size_t j = 0; j < costs[i].size(); ++j )
			{
				int cost = costs[i][j] + (j == min1_color) ? min2_so_far : min1_so_far;

				if ( cost < cur_min1 ) //update mininum
				{
					cur_min2 = cur_min1;
					cur_min1 = cost;
					cur_min1_color = j;
				}
				else if (cost < )
				{
					cur_min2 = cost; // update second minimum.
				}
			}

			//update min1_so_far, min2_so_far and min1_color
			min1_so_far = cur_min1;
			min2_so_far = cur_min2;
			min1_color = cur_min1_color;
		}

		return min1_so_far;
	}
 };
 
//<-->266. Palindrome Permutation
/*
Given a string, determine if a permutation of

the string could form a palindrome.

For example,
"code" -> False, "aab" -> True, "carerac" -> True.

Hint:

1. Consider the palindromes of odd vs even length. What difference do you notice?
2. Count the frequency of each character.
3. If each character occurs even number of times,
then it must be a palindrome. How about character which occurs odd number of times?
*/
class Solution {
public:
    //method 1: using map;
    bool canPermutePalindrome(string s)
    {
        unordered_map<char, int> m;
        
        for( auto c : s )
        {
            if(m.find(c) == m.end())
            {
                m[c] = 0;
            }
            
            ++m[c];
        }
        
        int odd_count = 0;
        
        for( auto& p : m )
        {
            if(p.second % 2 == 1)
            {
                ++odd_count;
            }
        }
        
        return odd_count == 0 || ( ( s.size() & 1 == 1 ) && ( odd_count == 1 ) );
    }
    
    //method 2: using set
    bool canPermutePalindrome(string s)
    {
        unordered_set<char> char_set;
        
        for( auto c : s )
        {
            if( char_set.find(c) == s.end() )
            {
                char_set.insert(c);
            }
            else
            {
                char_set.erase(c);
            }
        }
        
        return ( char_set.empty() ) || ( char_set.size() == 1 )
    }
    
    //method 3: using bitset
    bool canPermutePalindrome(string s)
    {
        bitset<256> bs;
        
        for( auto c : s )
        {
            b.flip( c );
        }
        
        return b.count() < 2;
    }
};

//<-->267. Palindrome Permutation II
/*
Given a string s, return all the palindromic permutations

(without duplicates) of it. Return an empty list if no palindromic permutation could be form.

For example:

Given s = "aabb", return ["abba", "baab"].

Given s = "abc", return [].

Hint:

1. If a palindromic permutation exists, we just need to generate the first half of the string.
2. To generate all distinct permutations of a (half of) string,
use a similar approach from: Permutations II or Next Permutation.
*/
class Solution {
public:
    
    //Method 1: using swap to generate permutation
    vector<string> generatePalindromes(string s)
    {
        if(s.empty())
        {
            return {};
        }
        
        unordered_map<char, int> m;
        
        for( auto c : s )
        {
            if(m.find(c)==m.end())
            {
                m[c] = 0;
            }
            
            ++m[c];
        }
        
        string mid;
        string fr;
        
        for( auto& p : m )
        {
            if( ( p.second & 1 ) == 1 ) //odd number
            {
                mid.push_back(p.first);
                
                if(mid.size() > 1) //key: if more than one character appears odd times, it cannot be a palindrome.
                {
                    return {};
                }
            }
            
            //generate first half
            int count = p.second >> 1;
                
            for(int i = 0; i < count; ++i)
            {
                fr.push_back(p.first);
            }
        }
        
        vector<string> res;
        
        permute(mid, fr, 0, res);
        
        return res;
    }
    
    void permute( const string& mid, string& fr, int start, vector<string>& res )
    {
        if(start==fr.size())
        {
            res.emplace_back( fr + mid + string( fr.rbegin(), fr.rend() ) );
            return;
        }
        
        for(int i = start; i< fr.size(); ++i)
        {
            if(i!=start && fr[i]==fr[start])
            {
                continue;
            }
            
            swap(fr[start], fr[i]);
            permute(mid, fr, start+1, res);
            swap(fr[start], fr[i]);
        }
    }
    
    //Method 2: using different characters to generate permutation
    vector<string> generatePalindromes(string s)
    {
        if(s.empty())
        {
            return {};
        }
        
        unordered_map<char, int> m;
        
        for( auto c : s )
        {
            if(m.find(c)==m.end())
            {
                m[c] = 0;
            }
            
            ++m[c];
        }
        
        string mid;
        string fr;
        
        for( auto& p : m )
        {
            if( ( p.second & 1 ) == 1 ) //odd number
            {
                mid.push_back(p.first);
                
                if(mid.size() > 1) //key: if more than one character appears odd times, it cannot be a palindrome.
                {
                    return {};
                }
            }
            
            //generate first half
            //notice the different from method 1:
            //the count of the character is halved directly. This is very important since the generation
            //is different
            p.second >>= 1;
                
            for(int i = 0; i < p.second; ++i)
            {
                fr.push_back(p.first);
            }
        }
        
        vector<string> res;
        
        permute(mid, fr, m, "", res);
        
        return res;
    }
    
    void permute( const string& mid, const string& fr, unordered_map<char, int>& m, string out, vector<string>& res )
    {
        if( out.size() == fr.size() )
        {
            res.emplace_back( out + mid + string( out.rbegin(), out.rend() ) );
            return;
        }
        
        for( auto& p : m )
        {
            if(p.second > 0)
            {
                --p.second;
                permute(mid, fr, m, out+p.first, res);
                ++p.second; //key: need to increase back to restore the count.
            }
        }
    }

};

//<-->268. Missing Number
/*
Given an array containing n distinct numbers

taken from 0, 1, 2, ..., n, find the one that is missing from the array.

For example,
Given nums = [0, 1, 3] return 2.
*/
class Solution {
public:
    //method 1: using sum;
    int missingNumber(vector<int>& nums)
    {
        auto sum = accumulate(begin(nums), end(nums), 0);
        
        int total = n * ( n - 1 );
        total >>= 1;
        
        return total - sum;
    }
    
    //method 2: using XOR
    int missingNumber(vector<int>& nums)
    {
        int len = nums.size();
        int res = 0;
        for( int i = 0; i < len; ++i )
        {
            res ^= ( ( i+1 ) ^ nums[i] );
        }
        
        return res;
    }
    
    //method 3: using binary search
    int missingNumber(vector<int>& nums)
    {
        sort(begin(nums), end(nums));
        
        int left = 0;
        int right = nums.size(); // key: right must be the nums.size()
        
        while(left<right)
        {
            int mid = left + (right-left)/2;
            //key: only compare with mid,
            //if it larger than mid, means we need to search in first half
            if(nums[mid] > mid) 
            {
                right = mid;
            }
            else
            {
                left = mid + 1;
            }
        }
        
        return right;
    }
};

//<-->269. Alien Dictionary
/*
There is a new alien language which uses the latin alphabet.
However, the order among letters are unknown to you.
You receive a list of words from the dictionary,
where words are sorted lexicographically
by the rules of this new language. Derive the order of letters in this language.

For example,
Given the following words in dictionary,

[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]
 

The correct order is: "wertf".

Note:

You may assume all letters are in lowercase.
If the order is invalid, return an empty string.
There may be multiple valid order of letters, return any one of them is fine.
*/

class Solution {
public:
    //method 1: bfs (using set and in-count)
    string alienOrder(vector<string>& words)
    {
        unordered_set<char> char_set;
           
        for(auto & word : words)
        {
            char_set.insert(word.begin(), word.end());
        }
        
        set<pair<char,char>> table;
        
        for(size_t i = 0; i < words.size() - 1; ++i)
        {
            auto& cur = words[i];
            auto& nxt = words[i+1];
            
            bool bValid = false;
            auto min_sz = min(cur.size(), nxt.size());
            
            for(size_t j = 0; j< min_sz; ++j)
            {
                if(cur[j]!=nxt[j])
                {
                    table.emplace( cur[j], nxt[j] );
                    bValid = true;
                    break;
                }
            }
            
            if( ( !bValid ) || ( cur.size() > nxt.size() ) )
            {
                return ""
            }
        }
        
        vector<int> in_cnt(26, 0);
        for( auto& p : table )
        {
            ++in_cnt[p.second];
        }
        
        queue<char> q;
        
        for( auto& c : char_set )
        {
            if(in_cnt[c-'a'] == 0)
            {
                q.push(c);
            }
        }
        
        while(!q.empty())
        {
            auto t = q.top();
            q.pop();
            
            for(auto & p : table)
            {
                if(p.first == t)
                {
                    --in_cnt[p.second-'a'];
                    if(in_cnt[p.second-'a'] == 0)
                    {
                        q.push(p.second);
                        res += p.second;
                    }
                }
            }
        }
        
        return res.size() == char_set.size() ? res : "";
    }
    
    //method 2: dfs (recursion)
    string alienOrder(vector<string>& words)
    {
        vector<vector<int>> g(26, vector<int>(26, 0));
        
        for(auto& word : words)
        {
            for( auto& c : word)
            {
                g[c-'a'][c-'a'] = 1;
            }
        }
        
        vector<int> path(26, 0);
        
        for(size_t i = 0; i < words.size() - 1; ++i)
        {
            auto min_sz = min(words[i].size(), words[i+1].size());
            
            bool bValid = false;
            
            for(size_t j = 0; j < min_sz; ++j)
            {
                auto prev = words[i][j];
                auto next = words[i+1][j];
                if(prev != next)
                {
                    bValid = true;
                    g[prev - 'a'][next - 'a'] = 1;
                }
            }
            
            if( !bValid && ( words[i].size() > words[i+1].size() ) )
            {
                return "";
            }
        }
        
        string res = "";
        
        for(int i = 0; i<26; ++i)
        {
            if(!dfs(g, path, i, res))
            {
                return "";
            }
        }
        
        for(int i = 0; i<26; ++i)
        {
            if(g[i][i] == 1)  // this are the standalone characters appears in the first word but does not appear in any other word
            {
                res += (char)( i + 'a' );
            }
        }
        
        return string(res.rbegin(), res.rend());
    }
    
    bool dfs(vector<vector<int>>& g, vector<int>& path, int idex, string& res)
    {
        if(g[idx][idx] == 0)
        {
            return true;
        }
        
        path[idx] = 1;
        
        for(int i = 0; i<26; ++i)
        {
            if( ( i == idx ) || ( g[idx][i] == 0 ) )
            {
                continue;
            }
            
            if(path[i] == 1)
            {
                return false;
            }
            
            if(!dfs(g,path,i,res))
            {
                return false;
            }
        }
        
        path[idx] = 0;
        g[idx][idx] = 0; // key: we already have visited
        res += char(idx + '0');
    }
};

//<-->270. Closest Binary Search Tree Value
/* 
Given a non-empty binary search tree and a target value,
find the value in the BST that is closest to the target.

Note:

Given target value is a floating point.
You are guaranteed to have only one unique
value in the BST that is closest to the target.
*/
class Solution {
public:
    int closestValue(TreeNode* root, double target)
    {
        
        while(root)
        {
            
        }
    }
}

//<--> 271. Encode and Decode Strings
/*
Design an algorithm to encode a list of strings to a string. 
The encoded string is then sent over the network 
and is decoded back to the original list of strings.

Machine 1 (sender) has the function:

string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
Machine 2 (receiver) has the function:

vector<string> decode(string s) {
  //... your code
  return strs;
}
 

So Machine 1 does:

string encoded_string = encode(strs);
 

and Machine 2 does:

vector<string> strs2 = decode(encoded_string);
 

strs2 in Machine 2 should be the same as strs in Machine 1.

Implement the encode and decode methods.

Note:

1.The string may contain any possible characters out of 256 valid ascii characters. 
Your algorithm should be generalized enough to work on any possible characters.
2. Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.
3. Do not rely on any library method such as eval or serialize methods. 
You should implement your own encode/decode algorithm.
/*



//<--> 272. Closest Binary Search Tree Value II
/*
Given a non-empty binary search tree and a target value, 
find k values in the BST that are closest to the target.

Note:

Given target value is a floating point.
You may assume k is always valid, that is: k ≤ total nodes.
You are guaranteed to have only one unique set of k values in the BST that are closest to the target.
 

Follow up:
Assume that the BST is balanced, could you solve it in less than O(n) runtime (where n = total nodes)?

Hint:

1. Consider implement these two helper functions:
　　i. getPredecessor(N), which returns the next smaller node to N.
　　ii. getSuccessor(N), which returns the next larger node to N.
2. Try to assume that each node has a parent pointer, it makes the problem much easier.
3. Without parent pointer we just need to keep track of the path from the root to the current node using a stack.
4. You would need two stacks to track the path in finding predecessor and successor node separately.
*/
class Solution {
public:
	//method 1: using inorder
    vector<int> closestKValues(TreeNode* root, double target, int k)
	{
		vector<int> res;
		inorder(root, target, k, res);
		return res;
	}
	
	void inorder(TreeNode* root, double target, int k, vector<int>& res)
	{
		if(root)
		{
			inorder(root->left, target, k, res);
			
			if(res.size() < k)
			{
				res.push_back(root->val);
			}
			else if( abs( res[0] - target ) > abs( root->val - target ) )
			{
				res.erase(res.begin());
				res.push_back(root->val);
			}
			else
			{
				return;
			}
			
			inorder(root->right, target, k, res);
		}
	}
	
	//method 2: using iterative inorder
    vector<int> closestKValues(TreeNode* root, double target, int k)
	{
		stack<TreeNode*> s;
		
		auto p = root;
		
		while(p || !s.empty())
		{
			while(p)
			{
				s.push(p);
				p = p->left;
			}
			
			p = s.top();
			s.pop();
			
			if(res.size() < k)
			{
				res.push_back(p->val);
			}
			else if( abs( res[0] - target ) > abs( p->val - target ) )
			{
				res.erase(res.begin());
				res.push_back(p->val);
			}
			else
			{
				break;
			}
			
			p = p->right;
		}
		
		return res;
	}
	
	//method 3: using two stacks
	vector<int> closestKValues(TreeNode* root, double target, int k)
	{
		stack<TreeNode*> pre_s;
		stack<TreeNode*> next_s;
		
		while(root)
		{
			if(root->val <= target)
			{
				pre_s.push(root);
				root = root->right;
			}
			else
			{
				next_s.push(root);
				root = root->left;
			}
		}
		
		for(int i = 0; i<k; ++i)
		{
			if( next_s.empty() || ( !pre_s.empty() && ( next_s.top() - target ) ) )
			{
				res.push_back(pre_s.top()->val);
				getPredecessor(pre_s);
			}
			else
			{
				res.push_back(next_s.top()->val);
				getSuccessor(next_s);				
			}
		}
	}
	
	void getPredecessor(stack<TreeNode*>& pre)
	{
		auto t = pre.top();
		pre.pop();
		
		if(t->left)
		{
			pre.push(t->left);
			while(pre.top()->right)
			{
				pre.push(pre.top()->right);
			}
		}
	}
	
	void getSuccessor(stack<TreeNode*>& next)
	{
		auto t = next.top();
		next.pop();
		
		if(t->right)
		{
			pre.push(t->right);
			while(pre.top()->left)
			{
				pre.push(pre.top()->left);
			}
		}
	}
};

//<--> 273. Integer to English Words
/*
Convert a non-negative integer to its english words representation. 
Given input is guaranteed to be less than 2^31 - 1.

For example,
123 -> "One Hundred Twenty Three"
12345 -> "Twelve Thousand Three Hundred Forty Five"
1234567 -> "One Million Two Hundred Thirty Four 
Thousand Five Hundred Sixty Seven"
*/

//"One", "Two", "Three", 
//"Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", 
//"Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", 
// "Sixteen", "Seventeen", "Eighteen", "Nineteen"
// "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"
// "Thousand", "Million", "Billion"
class Solution {
public:
    string numberToWords(int num)
	{
		vector<string> v = 
		{
			"Thousand",
			"Million",
			"Billion"
		}
		vector<string> v1 = 
		{
			"",
			"One", "Two", "Three",
			"Four", "Five", "Six", 
			"Seven", "Eight", "Nine", 
			"Ten", "Eleven", "Twelve", 
			"Thirteen", "Fourteen", "Fifteen",
			"Sixteen", "Seventeen", "Eighteen", 
			"Nineteen"
		};
		
		vector<string> v2 = 
		{
			"", "",
			"Twenty", "Thirty", "Forty", 
			"Fifty", "Sixty", 
			"Seventy", "Eighty", 
			"Ninety"
		}
		
		int r = num % 100;
		auto res = convert_hundred(r, v1, v2);
		
		for(int i = 0; i<3; ++i)
		{
			num /= 1000;
			
			if(num == 0)
			{
				break;
			}
			
			r = num % 1000;
			
			res = (r!=0) ? covert_hundred(r, v1, v2) + " " + v[i] + " " + res : res;
		}
		
		while(res.back()==' ')
		{
			res.pop_back();
		}
		
		return res.empty() ? "Zero" : res;
		
	}
	
	string convert_hundred(int n, const vector<string>& v1, const vector<string>& v2)
	{
		int a = n / 100;
		int b = n - a * 100;
		int c = n % 10;
		
		string res = b < 20 ? ( v1[b] ): ( v2[b/10] + ( c!=0 ? " "+v1[c] : "" ) );
		if( a > 0 )
		{
			res =  ( v1[a] + " Hundred" ) + ( b != 0 ? " " + res : "" );
		}
		
	}
};

//<--> 274. H-Index
/*
Given an array of citations (each citation is a non-negative integer) 
of a researcher, write a function to compute the researcher's h-index.

According to the definition of h-index on Wikipedia: 
"A scientist has index h if h of his/her N papers have at least h citations each, 
and the other N − h papers have no more than h citations each."

For example, given citations = [3, 0, 6, 1, 5], which means the researcher 
has 5 papers in total and each of them had received 3, 0, 6, 1, 5 citations respectively. 
Since the researcher has 3 papers with at least 3 citations each and 
the remaining two with no more than 3 citations each, his h-index is 3.

Note: If there are several possible values for h, the maximum one is taken as the h-index.

*/
class Solution {
public:
	//method 1: sort
    int hIndex(vector<int>& citations) 
	{
		sort(begin(citations), end(citations),greater<int>);
		int len = citations.size();
		for(int i = 0; i<len; ++i)
		{
			if(i >= citations[i])
			{
				return i;
			}
		}
		
		return citations.size();
    }
	
	//method 2: O(n) memory space
	int hIndex(vector<int>& citations)
	{
		vector<int> stats(citations.size()+1, 0);
		int len = citations.size();
		
		for(int i = 0; i<len; ++i)
		{
			if( citation[i] < len )
			{
				stats[citation[i]] += 1;
			}
			else
			{
				stats[len] += 1;
			}
		}
		
		int sum = 0;
		for(int i = len; i>0; --i)
		{
			sum += stats[i];
			if(sum >= i)
			{
				return i;
			}
		}
		
		return 0;
	}
};

//<--> 275. H-Index II
/*
Follow up for H-Index: What if the citations array is sorted in ascending order? 

Could you optimize your algorithm?

Hint:

Expected runtime complexity is in O(log n) and the input is sorted
*/
class Solution {
public:
    int hIndex(vector<int>& citations)
		{
			int len = citations.size();
			
			int left = 0, right = citations.size() - 1;
			
			while(left <= right)
			{
				int mid = left + (right - left)/2;
				if(citations[mid] == len - mid)
				{
					return len - mid;
				}
				
				if(citations[mid] > len - mid)
				{
					right  = mid - 1;
				}
				else
				{
					left = mid + 1;
				}
			}
			
			return len - left;
		}
};

//<--> 276. Paint Fence
/*
There is a fence with n posts, each post can be painted with one of the k colors.

You have to paint all the posts 
such that no more than two adjacent fence posts have the same color.

Return the total number of ways you can paint the fence.

Note:
n and k are non-negative integers.
*/
// induction:
// For the last 2 posts:
// 1) if the last 2 posts have same color, the last 3rd post cannot have the same color as these 2 posts.
// therefore, the ways to paint is diff[n-2] * (k-1)
// 2) if the last 2 posts have different color, then the ways to paint is diff[n-1]* (k-1);

class Solutions
{
	public:
	     int numWays(int n, int k) 
			 {
				 int dp2 = 0;
				 int dp1 = k;
				 
				 for(int i = 2; i<=n; ++i)
				 {
					 int t = (dp1 + dp2) * (k-1);
					 dp2 = dp1;
					 dp1 = t;
				 }
				 
				 return dp1+dp2;
			 }
			 
			 return dp1 + dp2;
}

//<-->277. Find the Celebrity
/*
Suppose you are at a party with n people (labeled from 0 to n - 1) 
and among them, there may exist one celebrity. 
The definition of a celebrity is that all the other n - 1 people know him/her
but he/she does not know any of them.

Now you want to find out who the celebrity is or verify 
that there is not one. The only thing you are allowed 
to do is to ask questions like: "Hi, A. Do you know B?" 
to get information of whether A knows B. 
You need to find out the celebrity (or verify there is not one) 
by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b) which tells you whether A knows B. 
Implement a function int findCelebrity(n), your function should minimize the number of calls to knows.

Note: There will be exactly one celebrity if he/she is in the party. 
Return the celebrity's label if there is a celebrity in the party. 
If there is no celebrity, return -1.
*/
class Solution {
public:
    int findCelebrity(int n)
		{
			int res = 0;
			
			for(int i = 0; i< n; ++i)
			{
				if(knows(res,i))
				{
					res = i;
				}
			}
			
			for(int i = 0; i<n; ++i)
			{
				if( res != i && ( knows( res, i ) || !knows( i, res ) ) )
				{
					return -1;
				}
			}
			
			return res;
		}
};

//<--> 278. First Bad Version
/*
You are a product manager and currently 
leading a team to develop a new product. 
Unfortunately, the latest version of your product fails the quality check. 
Since each version is developed based on the previous version, 
all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and 
you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) 
which will return whether version is bad. 
Implement a function to find the first bad version. 
You should minimize the number of calls to the API.
*/
// Forward declaration of isBadVersion API.
bool isBadVersion(int version);

class Solution {
public:
    int firstBadVersion(int n)
		{
			int left = 1;
			int right = n;
			
			while(left < right)
			{
				int mid = left + (right - left) / 2;
				if( isBadVersion( mid ) )
				{
					right = mid;
				}
				else
				{
					left  = mid + 1;
				}
			}
			
			return mid;
    }
};

//<--> 279. Perfect Squares
/*
Given a positive integer n, 
find the least number of perfect square numbers 
(for example, 1, 4, 9, 16, ...) which sum to n.

For example, given n = 12, return 3 
because 12 = 4 + 4 + 4; 
given n = 13, return 2 because 13 = 4 + 9.
*/
class Solution {
public:
		//method 1: most efficient way
		//任意一个正整数均可表示为4个整数的平方和，
		//其实是可以表示为4个以内的平方数之和，
		// 那么就是说返回结果只有1,2,3或4其中的一个，
		// 首先我们将数字化简一下，由于一个数如果含有因子4，
		// 那么我们可以把4都去掉，并不影响结果，比如2和8,3和12等等，返回的结果都相同，
		// 还有一个可以化简的地方就是，如果一个数除以8余7的话，那么肯定是由4个完全平方数组成
    int numSquares(int n) 
		{  
			while(n % 4 == 0)
			{
				n /= 4;
			}
			
			if(n%8=7)
			{
				return 4;
			}
			
			for(int a = 0; a<=n*n; ++a)
			{
				int b = static_cast<int>( sqrt(n - a*a) );
				if( a*a + b*b == n )
				{
					if(a!=0 && b!=0)
					{
						return 2;
					}
					
					return 1;
				}
			}
			
			return 3;
    }
		
		//method 2: DP
		int numSquares(int n) 
		{
			vector<int> dp(n, 0);
			dp[0] = 1; // n= 1, only 1;
			dp[1] = 2; // n= 2, only 1 + 1;
			
			for( int i = 3; i<=n; ++i )
			{
				dp[i-1] = INT_MAX;
				
				for(int j = 0; j*j <=i; ++j)
				{
					dp[i-1] = min(dp[i-1], dp[i-j*j-1] + 1);
				}
			}
			
			return dp.back();
		}
};

//<--> 280. Wiggle Sort
/*
Given an unsorted array nums, 

reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....

For example, given nums = [3, 5, 2, 1, 6, 4], 

one possible answer is [1, 6, 2, 5, 3, 4].
*/
class Solution {
public:
    void wiggleSort(vector<int> &nums)
		{
			auto len = nums.size();
			
			for(size_t i = 0; i<len; ++i)
			{
				if( i & 1 ) //odd position.
				{
					if(nums[i] < nums[i-1])
					{
						swap(nums[i], nums[i-1]);
					}
				}
				else		//even position.
				{
					if(nums[i] > nums[i-1])
					{
						swap(nums[i], nums[i-1]);
					}
				}
			}
		}
};

//<--> 281. Zigzag Iterator 
/*
Given two 1d vectors, implement an iterator to return their elements alternately.

For example, given two 1d vectors:

v1 = [1, 2]
v2 = [3, 4, 5, 6]
By calling next repeatedly until hasNext returns false, 
the order of elements returned by next should be: [1, 3, 2, 4, 5, 6].

Follow up: What if you are given k 1d vectors? How well can your code be extended to such cases?

Clarification for the follow up question - Update (2015-09-18):
The "Zigzag" order is not clearly defined and is ambiguous for k > 2 cases. 
If "Zigzag" does not look right to you, replace "Zigzag" with "Cyclic". For example, given the following input:

[1,2,3]
[4,5,6,7]
[8,9]
It should return [1,4,8,2,5,9,3,6,7].
*/
class ZigzagIterator
{
	public:
	 ZigzagIterator(vector<int>& v1, vector<int>& v2)
	 {
			if(!v1.empty())
			{
				q.emplace(v1.begin(), v1.end());
			}
			
			if(!v2.empty())
			{
				q.emplace(v2.begin(), v2.end());
			}
	 }
	 
	 int next() 
	 {
		 auto p = q.front();
		 auto it = p.first;
		 int val = *it;
		 
		 if( (++it) != p.second)
		 {
			 q.push(it, p.second);
		 }
	 }
	 
	 bool hasNext()
	 {
		 return !q.empty();
	 }
	 
	 private:
	 
		queue<pair<vector<int>::iterator, vector<int>::iterator>>> q;
}

//<--> 282. Expression Add Operators
/*
Given a string that contains only digits 0-9 and a target value, 
return all possibilities to add binary operators (not unary) +, -, or * 
between the digits so they evaluate to the target value.

Examples: 
"123", 6 -> ["1+2+3", "1*2*3"] 
"232", 8 -> ["2*3+2", "2+3*2"]
"105", 5 -> ["1*0+5","10-5"]
"00", 0 -> ["0+0", "0-0", "0*0"]
"3456237490", 9191 -> []
*/
class Solution {
public:
    vector<string> addOperators(string num, int target) 
		{
			vector<string> result;
			dfs(num, "", target, 0, 0, res);
			return result;
    }
		
		void dfs(string num, string expr, long long target, long long last, long long cur_expr_result, vector<string>& res)
		{
			if(expr.empty() && ( target == (int)cur_expr_result ) )
			{
				res.emplace_back(expr);
				return;
			}
			//key: we must let i until equal to num.size()
			//so that substr(0,i) finally can return whole string, and substr(i) return empty
			for(size_t i = 1; i<=num.size(); ++i) 
			{
				auto left = num.substr(0,i);
				if(left.size()>1 && left[0] == '0') //ignore leading zeros
				{
					return;
				}
				
				auto right = num.substr(i);
				
				auto L = stoll(left);
				
				if( expr.empty() )
				{
					dfs(right, left, target, L, L, res);
				}
				else
				{
					//add: the last one that brought to next level will be L 
					dfs(right, expr+"+"+left, target, L, cur_expr_result + L, res);
					//sub: the last one that brought to next level will be -L 
					dfs(right, expr+"-"+left, target, -L, cur_expr_result - L, res);
					
					//multiply: the last one that brought to next level will be last*L
					// and cur_expr_result will be (cur_expr_result - last)+( last * L )
					// For example: 2+3*2, when dfs run to last = 3, now, num=2, cur_expr_result = 5
					// we must remove 3 from cur_expr_result, and then add the multiply result of 3 and 2
					dfs(right, expr+"*"+left, target, L, cur_expr_result - last + ( last * L ), res);
				}
			}
		}
};

//<--> 283. Move Zeroes
/*
Given an array nums, write a function to 

move all 0's to the end of it while maintaining the relative order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.
*/
class Solution {
public:
    void moveZeroes(vector<int>& nums) 
		{
        size_t left = 0;
        
        for(size_t i = 0; i< nums.size(); ++i)
        {
            if(nums[i] != 0)
            {
                swap(nums[left++], nums[i]);
            }
        }
    }
};

//<-->284. Peeking Iterator
/*
Given an Iterator class interface with methods: next() and hasNext(), 

design and implement a PeekingIterator that support the peek() operation -- 

it essentially peek() at the element that will be returned by the next call to next().

Here is an example. Assume that the iterator is initialized to the beginning of the list: [1, 2, 3].

Call next() gets you 1, the first element in the list.

Now you call peek() and it returns 2, the next element. Calling next() after that still return 2.

You call next() the final time and it returns 3, the last element. 

Calling hasNext() after that should return false.
*/

// Below is the interface for Iterator, which is already defined for you.
// **DO NOT** modify the interface for Iterator.
class Iterator {
    struct Data;
	Data* data;
public:
	Iterator(const vector<int>& nums);
	Iterator(const Iterator& iter);
	virtual ~Iterator();
	// Returns the next element in the iteration.
	int next();
	// Returns true if the iteration has more elements.
	bool hasNext() const;
};


class PeekingIterator : public Iterator {
public:
	PeekingIterator(const vector<int>& nums) : Iterator(nums) {
	    // Initialize any member here.
	    // **DO NOT** save a copy of nums and manipulate it directly.
	    // You should only use the Iterator interface methods.
	    _use_peek = false;
	}

    // Returns the next element in the iteration without advancing the iterator.
	int peek() {
		if(!_use_peek)
		{
			val = Iterator::next();
			_use_peek = true;
		}
		
		return val;
	}

	// hasNext() and next() should behave the same as in the Iterator interface.
	// Override them if needed.
	int next() {
		if(!_use_peek)
		{
			return Iterator::next();
		}
		
		_use_peek = false;
		return val;
		
	}

	bool hasNext() const {
		if(!_use_peek)
		{
			return true;
		}
		
		return Iterator::hasNext();
	}
	
	private:
	
	  bool _use_peek;
		int val;
};

//<--> 285. Inorder Successor in BST   
/*
Given a binary search tree and a node in it, 

find the in-order successor of that node in the BST.

Note: If the given node has no in-order successor in the tree, return null.
*/
class Solution {
public:
		//method 1: iterative
    TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) 
		{
			auto cur = root;
			stack<TreeNode*> s;
			
			bool bFound = false;
			
			while(cur || !s.empty())
			{
				while(cur)
				{
					s.push(cur);
					cur = cur->left;
				}
				
				cur = s.top();
				s.pop();
				if(!bFound && ( cur == p ) ) 
				{
					bFound = true;
				}
				else if(bFound)
				{
					return cur;
				}
				
				cur = cur->right;
			}
			return nullptr;
		}
		
		//method 2: recursive inorder
		TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) 
		{
			TreeNode* pre = nullptr;
			TreeNode* suc = nullptr;
			
			inorder(root, p, pre, suc);
			return suc;
		}
		
		void inorder(TreeNode* root, TreeNode* p, TreeNode*& pre, TreeNode*& suc)
		{
			if(root)
			{
				inorder(root->left, p, pre, suc);
				if(pre==p)
				{
					suc = root;
				}
				pre = root;
				inorder(root->right, p, pre, suc);
			}
		}
		
		//method 3: iterative using BST properties
		TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p)
		{
			TreeNode* res = nullptr;
			
			while(root)
			{
				if(root->val > p->val)
				{
					//p is in the left subtree of root, 
					//so, root could be the successor since this is inorder
					res = root;
					root = root->left;
				}
				else
				{
					root = root->right;
				}
			}
			
			return res;
		}
		
		//method 4: recursive using BST properties
		TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p)
		{
			if(!root)
			{
				return nullptr;
			}
			
			if(root->val <= p->val)
			{
				return inorderSuccessor(root->right, p);
			}
			else
			{
				auto left = inorderSuccessor(root->left, p);
				return left ? left : root;
			}
		}
}

//<--> 286. Walls and Gates
/*
You are given a m x n 2D grid initialized with these three possible values.

-1 - A wall or an obstacle.
0 - A gate.
INF - Infinity means an empty room. We use the value 2^31 - 1 = 2147483647 
to represent INF as you may assume that the distance to a gate is less than 2147483647.

Fill each empty room with the distance to its nearest gate. 
If it is impossible to reach a gate, it should be filled with INF.

For example, given the 2D grid:
INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF
After running your function, the 2D grid should be:
  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4
*/
class Solution {
public:
		//dfs
    void wallsAndGates(vector<vector<int>>& rooms)
		{
			
		}
};

//<--> 287. Find the Duplicate Number
/*
Given an array nums containing n + 1 integers 

where each integer is between 1 and n (inclusive), 

prove that at least one duplicate number must exist. 

Assume that there is only one duplicate number, find the duplicate one.

Note:
1. You must not modify the array (assume the array is read only).
2. You must use only constant, O(1) extra space.
2. Your runtime complexity should be less than O(n^2).
There is only one duplicate number in the array, 
but it could be repeated more than once.
*/
class Solution {
public:
    int findDuplicate(vector<int>& nums) 
		{
    
    }
};

//<--> 288. Unique Word Abbreviation
/*
An abbreviation of a word follows the form 
<first letter><number><last letter>. 
Below are some examples of word abbreviations:

a) it                      --> it    (no abbreviation)

     1
b) d|o|g                   --> d1g

              1    1  1
     1---5----0----5--8
c) i|nternationalizatio|n  --> i18n

              1
     1---5----0
d) l|ocalizatio|n          --> l10n
Assume you have a dictionary and given a word, 
find whether its abbreviation is unique in the dictionary. 
A word's abbreviation is unique if no other word from the dictionary has the same abbreviation.

Example: 

Given dictionary = [ "deer", "door", "cake", "card" ]

isUnique("dear") -> false
isUnique("cart") -> true
isUnique("cane") -> false
isUnique("make") -> true
*/
class ValidWordAbbr
{
	public:
    ValidWordAbbr(vector<string> &dictionary)
		{
		}
}

//<--> 289. Game of Life
/*
According to the Wikipedia's article: "The Game of Life, also known simply as Life, 

is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

Given a board with m by n cells, each cell has an initial state live (1) or dead (0).
Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) 
using the following four rules (taken from the above Wikipedia article):

1. Any live cell with fewer than two live neighbors dies, as if caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population..
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

Write a function to compute the next state (after one update) of the board given its current state.

Follow up: 
1. Could you solve it in-place? 
Remember that the board needs to be updated at the same time: 
You cannot update some cells first and then use their updated values to update other cells.
2. In this question, we represent the board using a 2D array. 
In principle, the board is infinite, 
which would cause problems when the active area encroaches the border of the array. 
How would you address these problems?
*/
// 0 : 上一轮是0，这一轮过后还是0
// 1 : 上一轮是1，这一轮过后还是1
// 2 : 上一轮是1，这一轮过后变为0
// 3 : 上一轮是0，这一轮过后变为1
class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) 
		{    
    }
};

//<--> 290. Word Pattern
/*
Given a pattern and a string str, find if str follows the same pattern.

Here follow means a full match, 

such that there is a bijection between a letter in pattern and a non-empty word in str.

Examples:
pattern = "abba", str = "dog cat cat dog" should return true.
pattern = "abba", str = "dog cat cat fish" should return false.
pattern = "aaaa", str = "dog cat cat dog" should return false.
pattern = "abba", str = "dog dog dog dog" should return false.
Notes:
You may assume pattern contains only lowercase letters, and str 
contains lowercase letters separated by a single space.
*/
class Solution {
public:
    bool wordPattern(string pattern, string str) {
        
        unordered_map<char,string> m;
        unordered_set<string> s;
        
        istringstream iss(str);
        
        string cur_s;
        size_t i = 0;
        
        while(getline(iss, cur_s, ' '))
        {
            if(i == pattern.size())
            {
                return false;
            }
            
            auto c = pattern[i];
            
            if( m.find(c)!=m.end() )
            {
                if(m[c] != cur_s )
                {
                    return false;
                }
            }
            else
            {
                if(s.find(cur_s)!=s.end())
                {
                    return false;
                }
                m.emplace(c, cur_s);
                s.emplace(cur_s);
            }
            
            ++i;
        }
        
        
        return i == pattern.size();
        

    }
};

//<--> 291. Word Pattern II
/*
Given a pattern and a string str, find if str follows the same pattern.

Here follow means a full match, 
such that there is a bijection between a letter in pattern 
and a non-empty substring in str.

Examples:

pattern = "abab", str = "redblueredblue" should return true.
pattern = "aaaa", str = "asdasdasdasd" should return true.
pattern = "aabb", str = "xyzabcxzyabc" should return false.
 

Notes:
You may assume both pattern and str contains only lowercase letters.
*/
class Solution {
public:
    bool wordPatternMatch(string pattern, string str)
		{
			unordered_map<char,string> m;
			
			for(auto& c : pattern)
			{
				if(m.find(c)==m.end())
				{
					m.emplace(c, "");
				}
			}
			
			return dfs(str, 0, pattern, 0, m);
		}
		
		bool dfs(string s, size_t si, string p, size_t pi, unordered_map<char, string>& m)
		{
			if( pi == p.size() && si == s.size() )
			{
				return true;
			}
			
			if(pi == p.size() || si == s.size())
			{
				return false;
			}
			
			char c = pattern[pi];
			
			for(size_t i = si; i< s.size(); ++i)
			{
				auto sub = s.substr(si, i-si+1);
				
				if(m[c]==sub)
				{
					if(dfs(s, i+1, p, pi+1, m))
					{
						return true;
					}
				}
				
				if(m[c].empty())
				{
					auto iter = find_if(begin(m), end(m), [&sub](const pair<char,string>& p){return p.second==sub});
					
					if(iter == m.end())
					{
						m[c] = sub;
						if(dfs(s, i+1, p, pi+1, m))
						{
							return true;
						}
					}
				}
			} //end for
			
			return false;
		}
};

//<--> 292. Nim Game
/*
You are playing the following Nim Game with your friend: 

There is a heap of stones on the table, each time one of you take turns to remove 1 to 3 stones. 

The one who removes the last stone will be the winner. You will take the first turn to remove the stones.

Both of you are very clever and have optimal strategies for the game. 

Write a function to determine whether you can win the game 

given the number of stones in the heap.

For example, if there are 4 stones in the heap,

then you will never win the game: no matter 1, 2, or 3 stones you remove, 

the last stone will always be removed by your friend.
*/

//只要是4的倍数个，我们一定会输，所以对4取余即可
class Solution {
public:
		bool canWinNim(int n)
		{
			return n%4 != 0;
    }
};

//<--> 293. Flip Game
/*
You are playing the following Flip Game with your friend: 

Given a string that contains only these two characters: + and -, 

you and your friend take turns to flip twoconsecutive "++" into "--". 

The game ends when a person can no longer make a move and therefore the other person will be the winner.

Write a function to compute all possible states of the string after one valid move.

For example, given s = "++++", after one move, it may become one of the following states:

[
  "--++",
  "+--+",
  "++--"
]
 
If there is no valid move, return an empty list [].
*/

class Solution {
public:
    vector<string> generatePossibleNextMoves(string s)
		{
			vector<string> v;
			
			for(size_t i = 0; i<s.size()-1; ++i)
			{
				if(s[i] == '+' && s[i+1] == '+')
				{
					v.push_back(s);
					v[v.size()-1][i] = '-';
					v[v.size()-1][i+1] = '-';
				}
			}
			
			return v;
		}
		
		
};

//<--> 293. Flip Game II
/*
You are playing the following Flip Game with your friend: 

Given a string that contains only these two characters: + and -, 

you and your friend take turns to flip twoconsecutive "++" into "--". 

The game ends when a person can no longer make a move 

and therefore the other person will be the winner.

Write a function to determine if the starting player can guarantee a win.

For example, given s = "++++", return true. 

The starting player can guarantee a win by flipping the middle "++" to become "+--+".

Follow up:
Derive your algorithm's runtime complexity.
*/
class Solution {
public:
    bool canWin(string s)
		{
			for(size_t i = 0; i<s.size()-1; ++i)
			{
				if(s[i]=='+' && s[i+1]=='+')
				{
					auto tmp = s;
					tmp[i] = '-';
					tmp[i+1] = '-';
					
					if(!canWin(tmp))
					{
						return true;
					}
				}
			}
			
			return false;
		}
};

//<--> 295. Find Median from Data Stream
/*
Median is the middle value in an ordered integer list. 

If the size of the list is even, there is no middle value. 

So the median is the mean of the two middle value.

Examples: 
[2,3,4] , the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far.
For example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
*/

class MedianFinder {
public:
    /** initialize your data structure here. */
    MedianFinder() 
		{
    }
    
    void addNum(int num)
		{
    }
    
    double findMedian()
		{
			
    }
		
		private:
		
		
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
 
 //<--> 296. Best Meeting Point
/*
 A group of two or more people wants to meet and 
 
 minimize the total travel distance. 
 
 You are given a 2D grid of values 0 or 1, 
 
 where each 1 marks the home of someone in the group. 
 
 The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.

For example, given three people living at (0,0), (0,4), and (2,2):

1 - 0 - 0 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0

The point (0,2) is an ideal meeting point, as the total travel distance of 2+2+2=6 is minimal. 

So return 6.

Hint:

Try to solve it in one dimension first. How can this solution apply to the two dimension case?
*/

class Solution {
public:
    int minTotalDistance(vector<vector<int>>& grid)
		{
			int rows = grid.size();
			int cols = grid[0].size();
			
			vector<int> home_rows;
			vector<int> home_cols;
			
			for(int i = 0; i<rows; ++i)
			{
				for(int j = 0; j<cols; ++j)
				{
					if(grid[i][j]==1)
					{
						home_rows.push_back(i);
						home_cols.push_back(j);
					}
				}
			}
			
			sort(begin(home_rows), end(home_rows));
			sort(begin(home_cols), end(home_cols));
			
			int i = 0;
			int j = home_rows.size() -1 ;
			
			int min_dist = 0;
			
			while(i < j)
			{
				min_dist += ( home_rows[j] - home_rows[i] );
				min_dist += ( home_cols[j] - home_cols[i] );
				
				--j;
				++i;
			}
			
			return min_dist;
		}
};

//<--> 297. Serialize and Deserialize Binary Tree
/*
Serialization is the process of converting a data structure 

or object into a sequence of bits so that it can be stored in a file or memory buffer, 

or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. 

There is no restriction on how your serialization/deserialization algorithm should work. 

You just need to ensure that a binary tree can be serialized to a string and 

this string can be deserialized to the original tree structure.

For example, you may serialize the following tree

    1
   / \
  2   3
     / \
    4   5
as "[1,2,3,null,null,4,5]", just the same as how LeetCode OJ serializes a binary tree. 

You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Note: Do not use class member/global/static variables to store states. 

Your serialize and deserialize algorithms should be stateless.
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
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));

//<--> 298. Binary Tree Longest Consecutive Sequence
/*
Given a binary tree, find the length of the longest consecutive sequence path.

The path refers to any sequence of nodes from some starting node 
to any node in the tree along the parent-child connections. 
The longest consecutive path need to be from parent to child (cannot be the reverse).

For example,

   1
    \
     3
    / \
   2   4
        \
         5
Longest consecutive sequence path is 3-4-5, so return 3.

   2
    \
     3
    / 
   2    
  / 
 1
Longest consecutive sequence path is 2-3, not 3-2-1, so return 2.
*/
class Solution {
public:
	//method 1: recursive.
    int longestConsecutive(TreeNode* root)
	{
		if(!root)
		{
			return 0;
		}
		
		int res = 0;
		dfs(root, 1, res);
		
		return res;
	}
	
	void dfs(TreeNode* root, int len, int &res)
	{
		res = max(len, res);
		
		
		if( root->left )
		{
			if( root->left->val == ( root->val + 1) )
			{
				dfs(root->left, len+1, res);
			}
			else
			{
				dfs(root->left, 1, res);
			}
		}
		
		if( root->right )
		{
			if( root->right->val == ( root->val + 1) )
			{
				dfs(root->right, len+1, res);
			}
			else
			{
				dfs(root->right, 1, res);
			}
		}
	}
	
	//method 2: iterative.
    int longestConsecutive(TreeNode* root)
	{
		if(!root)
		{
			return 0;
		}
		
		int res = 0;
		
		queue<TreeNode*> q;
		q.push(root);
		
		while(!q.empty())
		{
			auto t = q.front();
			q.pop();
			
			while( (t->left && t->left->val == t->val + 1) || (t->right && t->right->val == t->val + 1) )
			{
				if(t->left && t->left->val == t->val + 1)
				{
					if(t->right)
					{
						q.push(t->right);
					}
					t = t->left;
				}
				else if(t->right && t->right->val == t->val + 1)
				{
					if(t->left)
					{
						q.push(t->left);
					}
					t = t->right;
				}
				
				++len;
			} //end inner while
			
			res = max(len, res);
			
			if(t->left)
			{
				q.push(t->left);
			}
			
			if(t->right)
			{
				q.push(t->right);
			}
		}
		
		return res;
	}
};

//<--> 299. Bulls and Cows
/*
You are playing the following Bulls and Cows game with your friend: 

You write down a number and ask your friend to guess what the number is. 

Each time your friend makes a guess, you provide a hint that indicates 

how many digits in said guess match your secret number exactly in both digit and position 

(called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). 

Your friend will use successive guesses and hints to eventually derive the secret number.

For example:

Secret number:  "1807"
Friend's guess: "7810"
Hint: 1 bull and 3 cows. (The bull is 8, the cows are 0, 1 and 7.)
Write a function to return a hint according to the secret number and friend's guess, 

use A to indicate the bulls and B to indicate the cows. 

In the above example, your function should return "1A3B".

Please note that both secret number and friend's guess may contain duplicate digits, for example:

Secret number:  "1123"
Friend's guess: "0111"
In this case, the 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow, 
and your function should return "1A1B".
You may assume that the secret number and your friend's guess only contain digits, 
and their lengths are always equal.
*/
class Solution {
public:
	// two passes
	string getHint(string secret, string guess)
	{
		int len = secret.size();

		int m[10] = { 0 };

		int cows = 0;
		int bulls = 0;

		for (int i = 0; i < len; ++i)
		{
			if (secret[i] == guess[i])
			{
				++bulls;
			}
			else
			{
				++m[secret[i]];
			}
		}

		for (int i = 0; i < len; ++i)
		{
			if ((secret[i] != guess[i]) && (m[guess[i]] != 0))
			{
				++cows;
				--m[guess[i]];
			}
		}

		return to_string(bulls) + "A" + to_string(cows) + "B";
	}

	// one passes
	// 在处理不是bulls的位置时，
	// 我们看如果secret当前位置数字的映射值小于0，
	// 则表示其在guess中出现过，cows自增1，然后映射值加1，
	// 如果guess当前位置的数字的映射值大于0，则表示其在secret中出现过，cows自增1，然后映射值减1
	string getHint(string secret, string guess)
	{
		int len = secret.size();

		int m[10] = { 0 };

		int cows = 0;
		int bulls = 0;

		for (int i = 0; i < len; ++i)
		{
			if (secret[i] == guess[i])
			{
				++bulls;
			}
			else
			{
				if (m[secret[i]-'0'] < 0)
				{
					++cows;
				}

				++m[secret[i] - '0'];
				
				if (m[guess[i] - '0'] > 0)
				{
					++cows;
				}

				--m[guess[i] - '0'];
			}
		}

		return to_string(bulls) + "A" + to_string(cows) + "B";
	}
};

//<--> 300. Longest Increasing Subsequence
/*
iven an unsorted array of integers, 

find the length of longest increasing subsequence.

For example,
Given [10, 9, 2, 5, 3, 7, 101, 18],
The longest increasing subsequence is [2, 3, 7, 101], 

therefore the length is 4. Note that there may be more than one LIS combination, 

it is only necessary for you to return the length.

Your algorithm should run in O(n2) complexity.

Follow up: Could you improve it to O(n log n) time complexity?
*/
class Solution {
public:
	int lengthOfLIS(vector<int>& nums)
	{
		if (nums.empty())
		{
			return 0;
		}

		vector<int> dp(nums.size(), 0);

		dp[0] = 1;

		int res = 1;

		int len = nums.size();

		for (int i = 1; i < len; ++i)
		{
			dp[i] = 0;

			for (int j = 0; j < i; ++j)
			{
				if (nums[j] < nums[i])
				{
					dp[i] = max(dp[i], dp[j] + 1);
				}
			}

			res = max(res, dp[i]);
		}

		return res;
	}

	//O(nlogn)
	int lengthOfLIS(vector<int>& nums)
	{
		vector<int> dp;
		int len = nums.size();
		
		for (int i = 0; i < len; ++i)
		{
			int left = 0;
			int right = dp.size();

			while (left < right)
			{
				int mid = left + (right - left) / 2;
				if (dp[mid] < nums[i])
				{
					left = mid + 1;
				}
				else
				{
					right = mid;
				}
			}

			if (right == dp.size())
			{
				dp.push_back(nums[i]);
			}
			else
			{
				dp[right] = nums[i];
			}
		}

		return dp.size();
	}
};

//<--> 301. Remove Invalid Parentheses
/*
Remove the minimum number of invalid parentheses in order to make 

the input string valid. Return all possible results.

Note: The input string may contain letters other than the parentheses ( and ).

Examples:
"()())()" -> ["()()()", "(())()"]
"(a)())()" -> ["(a)()()", "(a())()"]
")(" -> [""]
*/
class Solution {
public:
	//DFS method
	vector<string> removeInvalidParentheses(string s) 
	{
		if (s.empty())
		{
			return{""};
		}

		int min_del = INT_MAX;
		
		unordered_set<string> str_set;

		dfs(s, 0, "", 0, 0, min_del, 0, str_set);

		return vector<string>(str_set.begin(), str_set.end());
	}

	void dfs(string str, int start, string cur, int left, int right, int& min_del, int del, unordered_set<string>& str_set)
	{
		if (start == str.size())
		{
			if (left != right)
			{
				return;
			}

			if (del == min_del)
			{
				str_set.insert(cur);
			}
			else if (del < min_del)
			{
				str_set.clear();
				str_set.insert(cur);
			}

			min_del = min(del, min_del);
			return;
		}

		if (left < right) //key: if use left!=right, no any string will be generate. if use left > right, then invalid string will be included
		{
			return;
		}

		if (str[start] == '(')
		{
			dfs(str, start + 1, cur + "(", left + 1, right, min_del, del, str_set);
			//not use this left parenthesis, therefore, del will be increment
			dfs(str, start + 1, cur, left, right, min_del, del + 1, str_set);
		}
		else if (str[start] == ')')
		{
			dfs(str, start + 1, cur + ")", left, right + 1, min_del, del, str_set);
			//not use this right parenthesis, therefore, del will be increment
			dfs(str, start + 1, cur, left, right, min_del, del + 1, str_set);
		}
		else
		{
			dfs(str, start + 1, cur + str[start], left, right, min_del, del, str_set);
		}
	}

	//BFS
	//The idea is very much: add each valid string into the queue, and remove the parentheis each time.
	//if the string is valid, we will not remove parentheis from this string
	bool is_valid( const string& s )
	{
		int count = 0;

		for ( const auto& c : s )
		{
			if ( c == '(' )
			{
				++count;
			}
			else if ( c == ')' )
			{
				if ( count == 0 )
				{
					return false;
				}

				--count;
			}
		}

		return true;
	}

	vector<string> removeInvalidParentheses( string s )
	{
		vector<string> res;
		unordered_map<string, int> visited;

		queue<string> q;

		q.push( s );

		bool bFound = false;

		while ( !q.empty() )
		{
			s = q.front();
			q.pop();

			if ( is_valid( s ) )
			{
				res.emplace_back( s.c_str() );
				bFound = true;
			}

			if ( bFound )
			{
				continue;
			}

			for ( size_t i = 0; i < s.size(); ++i )
			{
				if ( s[i] != '(' && s[i] != ')' )
				{
					continue;
				}

				string t = s.substr( 0, i ) + s.substr( i + 1 );

				if ( visited.find( t ) == visited.end() )
				{
					res.emplace_back( t.c_str() );
					visited.emplace( t, 1 );
				}
			}
		}

		return res;
	}
};

//<--> 302. Smallest Rectangle Enclosing Black Pixels
/*
An image is represented by a binary matrix with 0 as a white pixel and 1 as a black pixel. 

The black pixels are connected, i.e., there is only one black region. 

Pixels are connected horizontally and vertically. 

Given the location (x, y) of one of the black pixels, 

return the area of the smallest (axis-aligned) rectangle that encloses all black pixels.

For example, given the following image:

[
"0010",
"0110",
"0100"
]
and x = 0, y = 2,

Return 6.
*/
class Solution {
public:
	//DFS: easy way to do
	int minArea( vector<vector<char>>& image, int x, int y )
	{
	}
};

//<--> 303. Range Sum Query - Immutable
/*
Given an integer array nums, 

find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
Note:
You may assume that the array does not change.
There are many calls to sumRange function.
*/
class NumArray {
public:
	NumArray( vector<int> nums ) 
	{
		if ( !nums.empty() )
		{
			sums.push_back( nums[0] );
			for ( size_t i = 1; i<nums.size(); ++i )
			{
				sums.push_back( sums.back() + nums[i] );
			}
		}
	}

	int sumRange( int i, int j ) {

		if ( sums.empty() )
		{
			return 0;
		}

		if ( i == 0 )
		{
			return sums[j];
		}
		else
		{
			return sums[j] - sums[i - 1];
		}
	}

private:

	vector<int> sums;
};

/**
* Your NumArray object will be instantiated and called as such:
* NumArray obj = new NumArray(nums);
* int param_1 = obj.sumRange(i,j);
*/

//<--> 304. Range Sum Query 2D - Immutable
/*
Given a 2D matrix matrix, find the sum of the elements 
inside the rectangle defined by its upper left corner (row1, col1) 
and lower right corner (row2, col2).

Example:
Given matrix = [
[3, 0, 1, 4, 2],
[5, 6, 3, 2, 1],
[1, 2, 0, 1, 5],
[4, 1, 0, 1, 7],
[1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
sumRegion(1, 1, 2, 2) -> 11
sumRegion(1, 2, 2, 4) -> 12

Note:
1. You may assume that the matrix does not change.
2. There are many calls to sumRegion function.
3/ You may assume that row1 <= row2 and col1 <= col2.
*/

/**
* Your NumMatrix object will be instantiated and called as such:
* NumMatrix obj = new NumMatrix(matrix);
* int param_1 = obj.sumRegion(row1,col1,row2,col2);
*/

class NumMatrix {
public:
	NumMatrix( vector<vector<int>> matrix ) 
	{
		if ( matrix.empty() || matrix[0].empty() )
		{
			return;
		}

		auto rows = matrix.size();
		auto cols = matrix[0].size();

		for ( size_t i = 1; i <= rows; ++i )
		{
			for ( size_t j = 1; j <= cols; ++j )
			{
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1] - dp[i - 1][j - 1] + matrix[i - 1][j - 1];
			}
		}
	}

	int sumRegion( int row1, int col1, int row2, int col2 )
	{
		return dp[row2 + 1][col2 + 1] - dp[row2 + 1][col1] - dp[row1][col2 + 1] + dp[row1][col1];
	}

private:
	vector<vector<int>> dp;
};



//<--> 305. Number of Islands II
/*
A 2d grid map of m rows and n columns is initially filled with water. 

We may perform an addLand operation which turns the water at position (row, col) 

into a land. Given a list of positions to operate, 

count the number of islands after each addLand operation. 

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 

You may assume all four edges of the grid are all surrounded by water.

Example:

Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]].
Initially, the 2d grid grid is filled with water. (Assume 0 represents water and 1 represents land).

0 0 0
0 0 0
0 0 0
Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.

1 0 0
0 0 0   Number of islands = 1
0 0 0
Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.

1 1 0
0 0 0   Number of islands = 1
0 0 0
Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land.

1 1 0
0 0 1   Number of islands = 2
0 0 0
Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land.

1 1 0
0 0 1   Number of islands = 3
0 1 0
We return the result as an array: [1, 1, 2, 3]

Challenge:

Can you do it in time complexity O(k log mn), where k is the length of the positions?
*/
class Solution {
public:
	vector<int> numIslands2( int m, int n, vector<pair<int, int>>& positions )
	{
		if ( m <= 0 || n <= 0 )
		{
			return{};
		}

		vector<int> roots( m*n, -1 ); //key: this vector will return the same island that current point will belong to.

		int count = 0;

		int dx[] = { 1, -1, 0, 0 };
		int dy[] = { 0, 0, 1, -1 };

		vector<int> res;

		for ( auto & p : positions )
		{
			int id = p.first*n + p.second;
			roots[id] = id;
			++count;

			for ( int i = 0; i < 4; ++i )
			{
				int x = p.first + dx[i];
				int y = p.second + dy[i];

				int cur_id = x*n + y;

				if ( x < 0 || x >= m || y < 0 || y >= n || roots[cur_id] == -1 )
				{
					continue;
				}

				int new_id = find_root( roots, cur_id );

				//key: since its adjacent is belongs to an island already marked, 
				//we should mark current point to this island.
				if ( id != new_id )  
				{
					roots[id] = new_id;
					id = new_id;
					--count;
				}
			}

			res.push_back( count );
		}

		return res;
	}

	int find_root( vector<int>& roots, int id )
	{
		while ( id != roots[id] )
		{
			roots[id] = roots[roots[id]];
			id = roots[id];
		}

		return id;
	}
};

//<--> 306. Additive Number
/*
Additive number is a string whose digits can form additive sequence.

A valid additive sequence should contain at least three numbers. 
Except for the first two numbers, each subsequent number in the sequence must be the sum of the preceding two.

For example:
"112358" is an additive number because the digits can form an additive sequence: 1, 1, 2, 3, 5, 8.

1 + 1 = 2, 1 + 2 = 3, 2 + 3 = 5, 3 + 5 = 8

"199100199" is also an additive number, the additive sequence is: 1, 99, 100, 199.

1 + 99 = 100, 99 + 100 = 199

Note: Numbers in the additive sequence cannot have leading zeros, so sequence 1, 2, 03 or 1, 02, 3 is invalid.

Given a string containing only digits '0'-'9', write a function to determine if it's an additive number.

Follow up:
How would you handle overflow for very large input integers?
*/
class Solution {
public:
	//this is done by myself
	bool isAdditiveNumber( string num ) 
	{
		if (sum.empty())
		{
			return false;
		}

		dfs(num, 0, "", "");
	}

	bool dfs( const string& n, size_t start, string s1, string s2 )
	{
		if ( !s1.empty() && !s2.empty() )
		{
			auto sum = get_sum( s1, s2 );

			if ( start + sum.size() <= n.size() )
			{
				if ( n.substr( start, sum.size() ) == sum )
				{
					//if start+sum.size() is less than total size, there still have more string
					if ( start + sum.size() < n.size() )
					{
						//s1 will be replaced with s2, and s2 will be replaced with sum.
						return dfs( n, start + sum.size(), s2, sum );
					}
					else
					{
						return true;
					}

				}
			}

			return false;
		}

		for ( size_t i = start; i < n.size(); ++i )
		{
			if ( !s1.empty() )
			{
				s1 = n.substr( start, i - start + 1 );

				//remove the case s1 has leading zeros
				if ( s1.size()> 1 && s1[0] == '0' )
				{
					return false;
				}

				if ( dfs( n, i + 1, s1, s2 ) )
				{
					return true;
				}

				s1 = ""; //key: we have to set s1 to empty if last dfs return false so that next loop will set s1 to a new string.
			}
			else if ( !s2.empty() )
			{
				s2 = n.substr( start, i - start + 1 );

				//remove the case s2 has leading zeros
				if ( s2.size()> 1 && s2[0] == '0' )
				{
					return false;
				}

				if ( dfs( n, i + 1, s1, s2 ) )
				{
					return true;
				}

				s2 = ""; //key: we have to set s2 to empty if last dfs return false so that next loop will set s1 to a new string.
			}
		}

		return false;
	}

	string get_sum( const string& s1, const string& s2 )
	{
		int len1 = s1.size() - 1;
		int len2 = s2.size() - 1;

		int sum = 0, carry = 0;

		while ( len1 >= 0 && len2 >= 0 )
		{
			sum = (len1 >= 0 ? s1[len1--] - '0' : 0) + (len2 >= 0 ? s2[len2--] - '0' : 0) + carry;
			carry = sum / 10;
			sum -= carry * 10;

			res.push_back( sum + '0' );
		}

		if ( carry != 0 )
		{
			res.push_back( carry + '0' );
		}

		return string( res.rbegin(), res.rend() );
	}
};

//<--> 307. Range Sum Query - Mutable
/*
Given an integer array nums, 
find the sum of the elements between indices i and j (i ≤ j), inclusive.

The update(i, val) function modifies nums 
by updating the element at index i to val.
Example:
Given nums = [1, 3, 5]

sumRange(0, 2) -> 9
update(1, 2)
sumRange(0, 2) -> 8
Note:
The array is only modifiable by the update function.
You may assume the number of calls to update and sumRange function is distributed evenly.
*/

/**
* Your NumArray object will be instantiated and called as such:
* NumArray obj = new NumArray(nums);
* obj.update(i,val);
* int param_2 = obj.sumRange(i,j);
*/

/*
这道题我们要使用一种新的数据结构，
叫做树状数组Binary Indexed Tree，
这是一种查询和修改复杂度均为O(logn)的数据结构。
这个树状数组比较有意思，所有的奇数位置的数字和原数组对应位置的相同，
偶数位置是原数组若干位置之和，假如原数组A(a1, a2, a3, a4 ...)，和其对应的树状数组C(c1, c2, c3, c4 ...)有如下关系：
C1 = A1
C2 = A1 + A2
C3 = A3
C4 = A1 + A2 + A3 + A4
C5 = A5
C6 = A5 + A6
C7 = A7
C8 = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8

那么是如何确定某个位置到底是有几个数组成的呢，
原来是根据坐标的最低位Low Bit来决定的，
所谓的最低位，就是二进制数的最右边的一个1开始，加厚后面的0(如果有的话)组成的数字，
例如1到8的最低位如下面所示：

坐标          二进制          最低位

1               0001          1

2               0010          2

3               0011          1

4               0100          4

5               0101          1

6               0110          2

7               0111          1

8               1000          8

...

最低位的计算方法有两种，一种是x&(x^(x–1))，另一种是利用补码特性x&-x。
*/
class NumArray {
public:
	NumArray( vector<int> nums ) 
	{
		bits.resize(nums.size() + 1, 0);
		N.resize(nums.size() + 1, 0);

		int len = nums.size();

		for (int i = 0; i < len; ++i)
		{
			update(i, nums[i]);
		}
	}

	void update( int i, int val )
	{
		int diff = val - N[i + 1];
		int len = N.size();

		for (int j = i + 1; j < len; j += (j&-j))
		{
			BIT[j] += diff;
		}

		N[i + 1] = val;
	}

	int sumRange( int i, int j )
	{
		return getSum(j + 1) - getSum(i);
	}

	int getSum(int i)
	{
		int res = 0;

		for (int j = i; j > 0; j -= (j&-j))
		{
			res += BIT[j];
		}

		return res;
	}

private:
	vector<int> BIT;
	vector<int> N;
};

//<--> 308. Range Sum Query 2D - Mutable
/*
Given a 2D matrix matrix, find the sum of the elements
inside the rectangle defined by its upper left corner (row1, col1)
and lower right corner (row2, col2).

Example:
Given matrix = [
[3, 0, 1, 4, 2],
[5, 6, 3, 2, 1],
[1, 2, 0, 1, 5],
[4, 1, 0, 1, 7],
[1, 0, 3, 0, 5]
]

sumRegion(2, 1, 4, 3) -> 8
update(3, 2, 2)
sumRegion(2, 1, 4, 3) -> 10
Note:
1. The matrix is only modifiable by the update function.
2. You may assume the number of calls to update and sumRegion function is distributed evenly.
3. You may assume that row1 <= row2 and col1 <= col2.
*/
/*Binary Index Tree Method: 2D*/
class NumMatrix {
public:
	NumMatrix( vector<vector<int>> &matrix )
	{
		if ( matrix.empty() || matrix[0].empty() )
		{
			return;
		}

		BIT.resize( matrix.size() + 1, vector<int>( matrix[0].size() + 1, 0 ) );
		N.resize( matrix.size() + 1, vector<int>( matrix[0].size() + 1, 0 ) );

		int rows = matrix.size();
		int cols = matrix[0].size();

		for ( int i = 0; i < rows; ++i )
		{
			for ( int j = 0; j < cols; ++j )
			{
				update( i, j, matrix[i][j] );
			}
		}
	}

	void update( int row, int col, int val )
	{
		int diff = val - N[row + 1][col + 1];
		int rows = N.size();
		int cols = N[0].size();

		for ( int i = row + 1; i < rows; i += (i&-i) )
		{
			for ( int j = col + 1; j < cols; j += (j&-j) )
			{
				BIT[i][j] += diff;
			}
		}

		N[row + 1][col + 1] = val;
	}

	int sumRegion( int row1, int col1, int row2, int col2 )
	{
		return getSum( row2 + 1, col2 + 1 ) - getSum( row2 + 1, col1 ) - getSum( row1, col2 + 1 ) + getSum( row1, col1 );
	}

	int getSum( int row, int col )
	{
		int sum = 0;

		for ( int i = row; i > 0; i -= (i&-i) )
		{
			for ( int j = col; j > 0; j -= (j&-j) )
			{
				sum += BIT[i][j];
			}
		}

		return sum;
	}

private:
	vector<vector<int>> BIT;
	vector<vector<int>> N;
};

//<--> 309. Best Time to Buy and Sell Stock with Cooldown
/*
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit.

You may complete as many transactions as you like
(i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

prices = [1, 2, 3, 0, 2]
maxProfit = 3
transactions = [buy, sell, cooldown, buy, sell]
*/

/*
The series of problems are typical dp.

The natural states for this problem is the 3 possible transactions:

buy, sell, rest. Here rest means no transaction on that day (aka cooldown).

Then the transaction sequences can end with any of these three states.

For each of them we make an array, buy[n], sell[n] and rest[n].

buy[i] means before day i what is the maxProfit for any sequence end with buy.

sell[i] means before day i what is the maxProfit for any sequence end with sell.

rest[i] means before day i what is the maxProfit for any sequence end with rest.

Then we want to deduce the transition functions for buy sell and rest. By definition we have:

buy[i]  = max(rest[i-1] - price, buy[i-1])

sell[i] = max(buy[i-1] + price, sell[i-1])

rest[i] = max(sell[i-1], buy[i-1], rest[i-1])

Where price is the price of day i. All of these are very straightforward. They simply represents :

(1) We have to `rest` before we `buy` and
(2) we have to `buy` before we `sell`

One tricky point is how do you make sure you sell before you buy,

since from the equations it seems that [buy, rest, buy] is entirely possible.

Well, the answer lies within the fact that buy[i] <= rest[i]

which means rest[i] = max(sell[i-1], rest[i-1]).

That made sure [buy, rest, buy] is never occurred.

A further observation is that rest[i] <= sell[i] is also true therefore

rest[i] = sell[i-1]

Substitute this in to buy[i] we now have 2 functions instead of 3:

buy[i] = max(sell[i-2]-price, buy[i-1])

sell[i] = max(buy[i-1]+price, sell[i-1])

This is better than 3, but

we can do even better

Since states of day i relies only on i-1 and i-2

we can reduce the O(n) space to O(1). And here we are at our final solution:
*/
class Solution {
public:
	int maxProfit( vector<int>& prices )
	{
		int buy = INT_MIN;
		int pre_buy = 0;
		int sell = 0;
		int pre_sell = 0;

		for ( auto p : prices )
		{
			pre_buy = buy;
			buy = max( pre_sell - p, pre_buy );

			pre_sell = sell;
			sell = max( pre_buy + p, pre_sell );
		}

		return sell;
	}
};

//<--> 310. Minimum Height Trees
/*
For a undirected graph with tree characteristics, 

we can choose any node as the root. 

The result graph is then a rooted tree. 

Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). 

Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. 

You will be given the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. 

Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

Example 1:

Given n = 4, edges = [[1, 0], [1, 2], [1, 3]]

0
|
1
/ \
2   3
return [1]

Example 2:

Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

0  1  2
\ | /
3
|
4
|
5
return [3, 4]

Show Hint
Note:

(1) According to the definition of tree on Wikipedia: 

“a tree is an undirected graph in which any two vertices are connected by exactly one path. 

In other words, any connected graph without simple cycles is a tree.”

(2) The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.
*/
class Solution {
public:
	vector<int> findMinHeightTrees(int n, vector<pair<int, int>>& edges) 
	{
		vector<unordered_set<int>> graph( n );

		for ( auto& p : edges )
		{
			graph[p.first].insert( p.second );
			graph[p.second].insert( p.first );
		}

		queue<int> q;

		for ( int i = 0; i < n; ++i )
		{
			if ( graph[i].size() == 1 )
			{
				q.push( i );
			}
		}

		while ( n > 2 )
		{
			int cur_sz = q.size();

			n -= cur_sz;

			for ( int i = 0; i < cur_sz; ++i )
			{
				auto t = q.front();
				q.pop();

				for ( auto v : graph[t] )
				{
					graph[v].erase( t ); //key: remove t from v
					if ( graph[v].size() == 1 )
					{
						q.push( v );
					}
				}
			}
		}

		vector<int> res;
		while ( !q.empty() )
		{
			res.push_back( q.front() );
			q.pop();
		}

		return res;
	}
};

//<--> 311. Sparse Matrix Multiplication
/*
Given two sparse matrices A and B, return the result of AB.

You may assume that A's column number is equal to B's row number.

Example:

A = [
[ 1, 0, 0],
[-1, 0, 3]
]

B = [
[ 7, 0, 0 ],
[ 0, 0, 0 ],
[ 0, 0, 1 ]
]


		|  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB =	| -1 0 3 | x | 0 0 0 | = | -7 0 3 |
					 | 0 0 1 |
*/
class Solution {
public:
	vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B)
	{
		int rows = A.size();
		int cols = A[0].size();

		vector<vector<int>> res( rows, vector<int>( cols, 0 ) );

		for ( int i = 0; i < rows; ++i )
		{
			for ( int j = 0; j < cols; ++j )
			{
				if ( A[i][j] != 0 )
				{
					for ( int k = 0; k < B[0].size(); ++k )
					{
						if ( B[j][k] != 0 )
						{
							res[i][k] += A[i][j] * B[j][k];
						}
					}
				}
			}
		}
	}
};

//<--> 312. Burst Balloons
/*
Given n balloons, indexed from 0 to n-1. 

Each balloon is painted with a number on it represented by array nums. 

You are asked to burst all the balloons. 

If the you burst balloon i you will get nums[left] * nums[i] * nums[right] coins.

Here left and right are adjacent indices of i. After the burst, the left and right then becomes adjacent.

Find the maximum coins you can collect by bursting the balloons wisely.

Note:
(1) You may imagine nums[-1] = nums[n] = 1. They are not real therefore you can not burst them.
(2) 0 <= n <= 500, 0 <= nums[i] <= 100

Example:

Given [3, 1, 5, 8]

Return 167

nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167
*/

class Solution {
public:
	int maxCoins(vector<int>& nums)
	{
		int len = nums.size();

		nums.insert( nums.begin(), 1 );
		nums.push_back( 1 );

		vector<vector<int>> dp( nums.size(), vector<int>( nums.size(), 0 ) );
		for ( int L = 1; L <= len; ++L )
		{
			for ( int left = 1; left <= len - L + 1; ++left )
			{
				int right = left + L - 1;

				for ( int k = left; k <= right; ++k )
				{
					dp[left][right] = max( dp[left][right],
						nums[left - 1] * nums[k] * nums[right + 1] + dp[left][k - 1] + dp[k + 1][right] );
				}
			}
		}

		return dp[1][n];
	}
};

//<--> 313. Super Ugly Number
/*
Write a program to find the nth super ugly number.

Super ugly numbers are positive numbers whose all prime 

factors are in the given prime list primes of size k. 

For example, [1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32] 

is the sequence of the first 12 super ugly numbers given primes = [2, 7, 13, 19] of size 4.

Note:
(1) 1 is a super ugly number for any given primes.
(2) The given numbers in primes are in ascending order.
(3) 0 < k <= 100, 0 < n <= 106, 0 < primes[i] < 1000.
(4) The nth super ugly number is guaranteed to fit in a 32-bit signed integer.
*/
class Solution {
public:
	int nthSuperUglyNumber( int n, vector<int>& primes )
	{
		vector<int> idx(primes.size(), 0);
		vector<int> dp(n, 1);

		for (int i = 1; i < n; ++i)
		{
			dp[i] = INT_MAX;

			for (size_t k = 0; k < primes.size(); ++k)
			{
				dp[i] = min(dp[i], dp[idx[k]] * primes[k]);
			}

			for (size_t k = 0; k < primes.size(); ++k)
			{
				if (dp[i] == dp[idx[k]] * primes[k])
				{
					++idx[k];
				}
			}
		}

		return dp.back();
	}
};

//<--> 	314. Binary Tree Vertical Order Traversal
/*
Given a binary tree, return the vertical order traversal of its nodes' values. 

(ie, from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.

Examples:
Given binary tree [3,9,20,null,null,15,7],
3
/ \
9  20
/  \
15   7
return its vertical order traversal as:
[
[9],
[3,15],
[20],
[7]
]
Given binary tree [3,9,20,4,5,2,7],
_3_
/   \
9    20
/ \   / \
4   5 2   7
return its vertical order traversal as:
[
[4],
[9],
[3,5,2],
[20],
[7]
]
*/
/*
我们可以把根节点给个序号0，然后开始层序遍历，
凡是左子节点则序号减1，右子节点序号加1，
这样我们可以通过序号来把相同列的节点值放到一起，
我们用一个map来建立序号和其对应的节点值的映射，
用map的另一个好处是其自动排序功能可以让我们的列从左到右，
由于层序遍历需要用到queue，
我们此时queue里不能只存节点，
而是要存序号和节点组成的pair，
这样我们每次取出就可以操作序号，
而且排入队中的节点也赋上其正确的序号
*/
class Solution {
public:
	vector<vector<int>> verticalOrder( TreeNode* root )
	{
		if (!root)
		{
			return {};
		}

		map<int, vector<int>> m;

		queue<pair<int, TreeNode*>> q;

		q.emplace(0, root);

		while (!q.empty())
		{
			auto p = q.front();
			q.pop();

			if (m.find(p.first) == m.end())
			{
				m.emplace(p.first, vector<int>());
			}

			m[p.first].push_back(p.second->val);

			if (p.second->left)
			{
				q.emplace(p.first - 1, p.second->left);
			}

			if (p.second->right)
			{
				q.emplace(p.first + 1, p.second->right);
			}
		}

		vector<vector<int>> res;
		for (auto& p : m)
		{
			res.emplace_back(p.second);
		}

		return res;
	}
};

//<--> 315. Count of Smaller Numbers After Self
/*
You are given an integer array nums and you have to return a new counts array. 

The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

Example:

Given nums = [5, 2, 6, 1]

To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
Return the array [2, 1, 1, 0].
*/
class Solution {
public:
	//method 1: using binary search
	vector<int> countSmaller(vector<int>& nums)
	{
		if ( nums.empty() )
		{
			return{};
		}

		vector<int> aux;
		vector<int> res( nums.size(), 0 );

		int len = nums.size();

		for ( int i = len - 1; i >= 0; --i )
		{
			int left = 0, right = aux.size();

			while ( left < right )
			{
				int mid = left + (right - left) / 2;
				if ( aux[mid] >= nums[i] )
				{
					right = mid;
				}
				else
				{
					left = mid + 1;
				}
			}

			res[i] = right;

			aux.insert( aux.begin() + right, nums[i] );
		}

		return res;
	}

	//method 2: using std function lower_bound
	vector<int> countSmaller( vector<int>& nums )
	{
		if ( nums.empty() )
		{
			return{};
		}

		vector<int> aux;
		vector<int> res( nums.size(), 0 );

		int len = nums.size();

		for ( int i = len - 1; i >= 0; --i )
		{
			auto it = lower_bound( begin( aux ), end( aux ), nums[i] );
			int d = distance( begin( aux ), it );

			res[i] = d;

			aux.insert( it, nums[i] );
		}

		return res;
	}
};

//<--> 316. Remove Duplicate Letters
/*
Given a string which contains only lowercase letters, 
remove duplicate letters so that every letter appear once and only once. 
You must make sure your result is the smallest in lexicographical order among all possible results.

Example:
Given "bcabc"
Return "abc"

Given "cbacdcbc"
Return "acdb"
*/
class Solution {
public:
	string removeDuplicateLetters( string s ) 
	{
		int m[26] = { 0 };
		int visit[26] = { 0 };

		for ( auto c : s )
		{
			++m[c - 'a'];
		}

		string res;

		for ( auto c : s )
		{
			--m[c - 'a'];
			if ( visit[c - 'a'] == 1 )
			{
				continue;
			}

			//key: (m[res.back() - 'a'] != 0 means this character will be met afterwards, so we will remove now.
			while ( !res.empty() && (c<res.back()) && (m[res.back() - 'a'] != 0) ) 
			{
				visit[res.back() - 'a'] = 0;
				res.pop_back();
			}

			res.push_back( c );
			visit[c - 'a'] = 1;
		}

		return res;
	}
};

//<-> 317. Shortest Distance from All Buildings
/*
You want to build a house on an empty land which reaches all buildings 

in the shortest amount of distance. 

You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:

Each 0 marks an empty land which you can pass by freely.
Each 1 marks a building which you cannot pass through.
Each 2 marks an obstacle which you cannot pass through.
For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):

1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0
The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal. So return 7.

Note:
There will be at least one building. If it is not possible to build such house according to the above rules, return -1.
*/
class Solution {
public:
	int shortestDistance( vector<vector<int>>& grid )
	{
	}
}
