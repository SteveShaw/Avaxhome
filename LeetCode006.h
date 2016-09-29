#pragma once

#include <string>

class LeetCode006
{
public:
	std::string Solution( std::string s, std::size_t row )
	{
		auto pStr = s.c_str();
		const auto len = s.size();

		std::string out;
		int step[2];
		step[0] = 2 * row - 2;
		step[1] = 0;
		int idx = 0;
		
		for ( std::size_t start = 0; start < len; start += step[0] )
		{
			out += pStr[start];
		}

		step[0] -= 2;
		step[1] += 2;

		for ( std::size_t row_i = 1; row_i < row-1; ++row_i )
		{
			auto start = row_i;
			
			idx = 0;
			while ( start < len )
			{
				out += pStr[start];
				start += step[idx%2];
				++idx;
			}
			step[0] -= 2;
			step[1] += 2;
		}

		step[1] += 2;

		for ( std::size_t start = row - 1; start < len; start += step[1] )
		{
			out += pStr[start];
		}

		return out;
	}
		
};

class LeetCode007
{
public:
	int myAtoi( std::string str ) 
	{
		const char* pStr = str.c_str();
		const auto len = str.size();

		int k = 0;

		bool bStart = false;

		int result = 0;

		while ( k < len )
		{
			if ( pStr[k] == ' ' )
			{
				if ( bStart )
				{
					break;
				}
				else
				{
					++k;
					continue;
				}
			}

			if ( IsDigit( pStr[k] ) )
			{
				if ( !bStart )
				{
					bStart = true;
				}
				
				result *= 10;
				result += (pStr[k] - '0');
				++k;
			}
			else
			{
				break;
			}
		}

		return result;
	}

	bool IsDigit( const char& c )
	{
		return ((c >= '0') && (c <= '9'));
	}


};


class LeetCode407
{
	using IntVec = std::vector<int>;
	using IntMat = std::vector<IntVec>;

	struct LowSquare
	{
		std::size_t row;
		std::size_t col;
		int num;
		int sum;
		int heightToReach;
	};

public:
	int trapRainWater( IntMat& heightMap )
	{
		rows = heightMap.size();
		cols = heightMap[0].size();

		marks = new int[rows*cols];
		visits = new unsigned char[rows*cols];

		memset( marks, 0, sizeof( int )*rows*cols );
		memset( visits, 0, sizeof( int )*rows*cols );

		int total = 0;


		for ( std::size_t r = 1; r < rows - 1; ++r)
		{
			for ( std::size_t c = 1; c < cols - 1; ++c )
			{
				if ( GetVisit( r, c ) > 0 )
				{
					continue;
				}

				int sum = 0;
				int num = 0;
				int min_h4N = 0;
				bool touch_boundary = false;
				min_reach_height = 1000000;
				bFoundLow = false;

				TrapWater( heightMap, r, c, num, sum, touch_boundary );

				if ( bFoundLow )
				{
					total += (num * min_reach_height - sum);
				}
			}
		}

		delete []marks;
		delete []visits;

		return total;
	}

private:
	int *marks;
	unsigned char *visits;
	//std::vector<int> paths;
	std::vector<LowSquare> lows;
	std::size_t rows;
	std::size_t cols;
	int min_reach_height;
	bool bFoundLow;
	//bool to_boundary;


	void SetMark( std::size_t row, std::size_t col, int mark )
	{
		const auto pos = row*cols + col;
		marks[pos] = mark;
	}

	int GetMark( std::size_t row, std::size_t col )
	{
		return marks[row*cols + col];
	}

	void SetVisit( std::size_t row, std::size_t col )
	{
		visits[row*cols + col] = 1;
	}

	unsigned char GetVisit( std::size_t row, std::size_t col )
	{
		return visits[row*cols + col];
	}

	/*
	{1,4,3,1,2,2},
	{3,2,2,4,2,4},
	{2,3,2,1,2,5},
	{2,3,3,4,6,6}

	[1,4,3,1,3,2],
	[5,3,7,2,4,3],
	[3,2,1,3,2,4],
	[2,3,3,2,3,1],
	[5,6,4,4,4,3]

	[1,4,3,1,3,2],
	[5,3,7,4,4,3],
	[3,2,1,3,5,4],
	[2,1,1,2,3,1],
	[5,6,4,4,4,3]
	*/

	void TrapWater( IntMat& hmat, std::size_t row, std::size_t col, int &num, int &sum, bool& touch_boundary )
	{
		if ( GetVisit( row, col ) != 0 )
		{
			return;
		}

		SetVisit( row, col );

		if ( row == 0 || row == rows - 1 || col == 0 || col == cols - 1 )
		{
			//This is the boundary
			//min_reach_height = hmat[row][col];
			touch_boundary = true;
			return;
		}

		num += 1;
		sum += hmat[row][col];

		auto t = hmat[row - 1][col];
		auto visit_t = GetVisit( row - 1, col );
		auto b = hmat[row + 1][col];
		auto visit_b = GetVisit( row + 1, col );
		auto l = hmat[row][col - 1];
		auto visit_l = GetVisit( row, col - 1 );
		auto r = hmat[row][col + 1];
		auto visit_r = GetVisit( row, col + 1 );

		auto c = hmat[row][col];


		//if ( GetVisit( row, col ) == 0 )
		//{
		//	SetVisit( row, col );

		//	sum += c;
		//	num += 1;
		//}
		//else
		//{
		//	return;
		//	//if ( c < t && c < b && c < l && c < r )
		//	//{
		//	//	auto mark = GetMark( row, col );
		//	//	if ( lows[mark].heightToReach < min_h4N )
		//	//	{
		//	//		lows[mark].heightToReach = min_h4N;
		//	//	}

		//	//	lows[mark].sum += sum;
		//	//	lows[mark].num += num;
		//	//}

		//	//return;
		//}

		unsigned char move[4] = { 0 };

		if ( t > c &&  visit_t == 0 )
		{
			min_reach_height = min_reach_height > t ? t : min_reach_height;
		}

		if ( b > c && visit_b == 0 )
		{
			min_reach_height = min_reach_height > b ? b : min_reach_height;
		}

		if ( l > c && visit_l == 0  )
		{
			min_reach_height = min_reach_height > l ? l : min_reach_height;
		}

		if ( r > c && visit_r == 0 )
		{
			min_reach_height = min_reach_height > r ? r : min_reach_height;
		}

		if ( t < min_reach_height && visit_t == 0 )
		{
			TrapWater( hmat, row - 1, col, num, sum, touch_boundary );
		}

		if ( b < min_reach_height && visit_b == 0 )
		{
			TrapWater( hmat, row - 1, col, num, sum, touch_boundary );
		}


		if ( b <= c )
		{
			move[1] = 1;
		}
		else
		{
			if ( min_reach_height > b )
			{
				min_reach_height = b;
			}
		}

		if ( l <= c )
		{
			bMove = true;
			if ( GetVisit( row,  col - 1 ) == 0 )
			{
				TrapWater( hmat, row, col - 1, num, sum, touch_boundary );
			}
		}
		else
		{
			min_reach_height = min_reach_height <= l ? min_reach_height : l;
		}

		if ( r <= c )
		{
			bMove = true;
			if ( GetVisit( row, col + 1 ) == 0 )
			{
				TrapWater( hmat, row, col + 1, num, sum, touch_boundary );
			}
		}
		else
		{
			min_reach_height = min_reach_height <= r ? min_reach_height : r;
		}

		if ( !bMove )
		{
			bFoundLow = true;
		}

		//if ( !bMove )
		//{
		//	min_reach_height 

		//	lows.emplace_back();
		//	auto& low = lows.back();
		//	low.row = row;
		//	low.col = col;
		//	low.sum = sum;
		//	low.num = num;
		//	low.heightToReach = ;
		//	SetMark( row, col, lows.size() - 1 );
		//	//lows.push_back( 0 ); //nums
		//	//lows.push_back( 0 ); //sums;
		//}
	}

};
