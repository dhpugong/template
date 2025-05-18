---
title: STL && 各种函数、容器
mathjax: true
math: true
categories: # 分类
  - c++板子  # 只能由一个
tags: # 标签
  - STL  # 能有多个
  - 算法  # 一个标签一行
  - c++
---

# Fragmented knowledge points+STL

<!-- more -->

## STL各种容器常数

均为大约值

- `vector`：1
- `queue`：1
- `map`：13
- `unoredred_map`：8

## 数组最值/求和

```cpp
#include <numeric>
#include<algorithm>

int sum = std::accumulate(a + 1, a + n + 1, 0);    // 0 是初始值，表示累加的起点
int ma = *max_element(a + 1, a + n + 1);           //多个最大值，返回第一个的指针
int mi = *min_element(a + 1, a + n + 1);
```

## bitset()

- 相当于一个二进制的数组，并且可以直接用 `01` 串赋值
- 时间复杂度一般为为 $O(\frac{N}w)$ ，$N$ 为 `bitset` 长度，$w$ 为计算机字长，其中在 32 位系统上， $w$ = 32，64位系统上 $w$ = 64 。

```cpp
bitset<4>a1;//长度为4，默认以0填充
bitset<8>a2(12);//长度为8，将12以二进制保存，前面用0补充

string s = "100101";
bitset<10>a3(s);//长度为10，前面用０补充

//c++11下，char可以赋值给bitset
char s2[] = "10101";
bitset<13>a4(s2);//长度为13，前面用０补充

cout<<a1<<endl;//0000
cout<<a2<<endl;//00001100
cout<<a3<<endl;//0000100101
cout<<a4<<endl;//0000000010101
```

- 如果超出了 `bitset` 定义的范围：

```cpp
bitset<2>bitset1(12);//12的二进制为1100（长度为４），但bitset1的size=2，只取后面部分，即00

string s="100101";
bitset<4> bitset2(s);//s的size=6，而bitset的size=4，只取前面部分，即1001
```

- 位运算操作

```cpp
bitset<4> foo (string("1001"));//这种赋值方式就可以直接用，没有限制
bitset<4> bar (string("0011"));
cout << (foo^=bar) << endl;       // 1010 (foo对bar按位异或后赋值给foo)
cout << (foo&=bar) << endl;       // 0010 (按位与后赋值给foo)
```

- 单一元素访问和修改

```cpp
bitset<4>a1("1011");//这个赋值方法只能在c++11里用
bitset<4>a1(string("1011"));

cout<<a1[0]<<endl;//1
cout<<a1[1]<<endl;//1
cout<<a1[2]<<endl;//0
cout<<a1[3]<<endl;//1
//注意！这两种赋值方式都是反序赋值的
//可以直接输出a1来输出正序

//bitset支持单点修改
a1[0]=0;
```

- 各种函数

```cpp
bitset<8>foo(string("10011011"));

cout<<foo.count()<<endl;//5　　（count函数用来求bitset中1的位数，foo中共有５个１
cout<<foo.size()<<endl;//8　　（size函数用来求bitset的大小，一共有８位

cout<<foo.test(0)<< endl;//true　　（test函数用来查下标处的元素是０还是１，并返回false或true，此处foo[0]为１，返回true
cout<<foo.test(2)<<endl;//false　　（同理，foo[2]为０，返回false

cout<<foo.any()<<endl;//true　　（any函数检查bitset中是否有１
cout<<foo.none()<<endl;//false　　（none函数检查bitset中是否没有１
cout<<foo.all()<<endl;//false　　（all函数检查bitset中是全部为１
```

## .compare()

```cpp
/*
从3开始，包含4个字符
1: >
0: ==
-1: <
*/
a.compare(3,4,b,3,4);
```

## get()

$2$ 进制第 $k$ 位上 $1$ 的个数

```cpp
// 1 ~ n 中第 k 位上 1 的个数, 0 <= n, 0 <= k <= 62
long long get(long long n, long long k){
    return (n) / (1LL << k + 1) * (1LL << k) + max((n) % (1LL << k + 1) + 1 - (1LL << k), 0LL);
}
```

## getnext()

返回下一位 $2$ 进制下相同 $1$ 个数的数

$g(1)=2$，$g(2)=4$，$g(3)=5$

```cpp
int cal(int x) {
    int low1 = x & -x;
    int y = x + low1;
    int low2 = y & -y;
    return y | (low2 / low1 >> 1) - 1;
}
```

## iota()

`iota`是一个算法函数，定义在头文件`<numeric>`中，用于生成一个连续递增的序列。具体来说，`iota`函数接受三个参数：两个迭代器（`first`和`last`），它们定义了要填充的序列的范围（注意，`last`是范围之外的第一个位置，即不包括`last`），以及一个初始值（`value`）。`iota`函数会从`first`开始，逐个将递增的值赋给范围内的元素，直到`last`（不包括`last`）。

```cpp
vector<int> a(5);
iota(a.begin(),a.end(),0);
```

## log()

`C++` 内置对数函数只有以 $e$ 为底和以 $10$ 为底的，如果想要以 $m$ 为底的对数可以借助如下公式：

$ loga(n)/loga(m) = logm(n) $ 。

```cpp
double res = log(n)/log(m);    //res = logm(n)
```

## lowbit()

找一个数 $2$ 进制下的最低位的 $1$。

```cpp
int lowbit(int x){
	return x & -x ;
}
```

```cpp
#define lowbit(x) ((x)&(-x))
```

 ## lower_bound/upper_bound()

### 用法

`lower_bound` 返回第一个大于等于x的元素的位置的迭代器（或指针）。

```cpp
int i=lower_bound(a+1,a+n+1,x)-a;
```

`upper_bound` 返回第一个    大于   $x$ 的元素的位置的迭代器（或指针）。

在有序 `vector` 中查找小于等于 $x$ 的最大整数（假设一定存在)

```cpp
 int y=*--upper_bound(a.begin(),a.end(),x);
```

### 比较函数

若数组降序排列，可写上比较函数 `greater<type>()`

```cpp
lower_bound(begin, end, a, greater<int>()) // 返回数组[begin, end)之间第一个小于或等于a的地址，找不到返回end
upper_bound(begin, end, a, greater<int>()) // 返回数组[begin, end)之间第一个小于a的地址，找不到返回end
```

**自定义比较函数**

`lower_bound/upper_bound` **默认操作**都是 **<** 。

`lower_bound` 返回第一个使 `cmp(element,value)` 为 $false$ 的元素位置（`element` 为序列的元素，`value` 为比较参数）。`upper_bound` 少用。

**查找第一个小于x的元素**

```cpp
// v从小到大排序
bool cmp(int e,int val){
	return e>=val;             // >  查找第一个<=x的元素
}
int t=lower_bound(v.begin(),v.end(),x,cmp)-v.begin()+1;
cout<<t;
```

**查找第一个不能被val整除的元素**

```cpp
bool cmp(int e,int val){
	return e%val==0;
}
```

## map

利用 `map` 可以简单实现 `hash`，离散化等操作。

### map

`map`：基于红黑树，元素有序存储；增删查改时间复杂度为 $O(logN)$。

利用 $map$ 排序特性可以查找 $key=x$ 的前一个元素和后一个元素，利用迭代器返回，不存在返回 `mp.end()`；

```cpp
map<int, int> mp;
mp[1] = 10;
mp[5] = 50;
mp[3] = 30;
auto l = mp.find(3);
auto r = l;
l--;r++;
cout << l->first << "\n";
cout << r->first << "\n";
```

### unordered_map

`unordered_map：`基于散列表，元素无序存储；大多数情况下其复杂度接近于 $O(1)$。

```cpp
using ULL = unsigned long long;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const ULL RANDOM = rng();
struct Hsh {
    ULL operator ()(const ULL& x)const {
        return x ^ RANDOM;
    }
};
unordered_map<ULL, ULL, Hsh> mp;
```

## memset()

可以初始化为 $-1，0，0x3f$

```cpp
//因为一个 int 是四个字节，所以 memset 会把每个字节初始化为你给的值

memset(a, 1, sizeof(int)*10) 
00000001 00000001 00000001 00000001   //每个字节初始化为1

memset(a, 2, sizeof(int)*10) 
00000010 00000010 00000010 00000010  每个字节初始化为2
```

## next_permutation

全排列

`next＿permutation` 的作用是在给定范围内找到下一个排列，如果有下一个排列，则返回 `true`，否则返回 `false` 。如果函数找到下一个排列，则函数会将给定范围内的元素按照字典序进行重排，并返回 `true` 。如果没有下一个排列，则函数返回 `false`，并将给定范围的元素排列成第一个排列。

```cpp
next_permutation(v.begin(), v.end())
```

## round()

四舍五入函数：

```cpp
#include<cmath>
int t=round(x);// 将浮点数四舍五入为整数
double t = round(x*100)/100.0  //保留浮点数的前两位小数
    
double my_round(double x) {        //手写round()函数   ##小数点13位以后会出现精度问题
    if (x < 0) return (int)(x - 0.5);
    return (int)(x + 0.5);
}
```

## sort()

is_sorted()

```cpp
//判断 [first, last) 区域内的数据是否符合 std::less<T> 排序规则，即是否为升序序列
bool is_sorted (ForwardIterator first, ForwardIterator last);
//判断 a 内的数据是否符合 cmp 排序规则  
bool is_sorted (a.begin(), a.end(), cmp);
```

## strstol()

```cpp
long int strtol(const char *str, char **endptr, int base)
```

`strtol()` 会将 $str$ 指向的字符串，根据参数 $base$，按权转化为 $long int$ ，然后返回这个值。

`base` 必须介于 $2$ 和 $36$（包含）之间，或者是特殊值 $0$；

`str`中不符合 $base$ 的部分存储于 `*endptr` 中。

```cpp
char str[100];
char *endptr;
cin>>str;
int x=strtol(str,&endptr,2);
cout<<x<<" "<<endptr;
```

```
①如果base为0，且字符串不是以0x(或者0X)开头，则按十进制进行转化。
②如果base为0或者16，并且字符串以0x（或者0X）开头，那么，x（或者X）被忽略，字符串按16进制转化。
③如果base不等于0和16，并且字符串以0x(或者0X)开头，那么x被视为非法字符。
④对于nptr指向的字符串，其开头和结尾处的空格被忽视，字符串中间的空格被视为非法字符。
```

## itoa()

将 $10$ 进制的 `value` 转化成 `radix` 进制

```cpp
char*itoa(int value,char*string,int radix);
```

## substr()

复制字符串

```cpp
string substr (size_t pos = 0, size_t len = npos) const;
```

## swap

`swap` 交换数组是 $O(n)$ ，（开 `c++11` 后）交换 $STL$ 是 $O(1)$ 。

## vector

[vector中间插入元素](https://blog.csdn.net/weixin_44205193/article/details/121522516#:~:text=本文详细介绍了C%2B%2B)

## 日历

```cpp
int dm[] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };

struct date {
	int y, m, d;
	int pf(int yy, int mm) {
		if (mm != 2) return 0;
		if ((yy % 4 == 0 && yy % 100 != 0) || yy % 400 == 0) return 1;
		return 0;
	}
	// 加x天
	date addd(int x) {
		date a = *this;
		a.d += x;
		while (a.d > dm[a.m] + pf(a.y, a.m)) {
			a.d -= dm[a.m] + pf(a.y, a.m);
			if (++a.m > 12) { a.m = 1; a.y++; }
		}
		return a;
	}
	// 加x个月
	date addm(int x) {
		date a = *this;
		a.m += x;
		while (a.m > 12) { a.y++; a.m -= 12; }
		return a;
	}
	// 判断日期是否合法
	bool pd() {
		if (d > dm[m] + pf(y, m)) return false;
		return true;
	}
	bool operator>(const date& b)const {
		if (y != b.y) return y > b.y;
		if (m != b.m) return m > b.m;
		return d > b.d;
	}
	bool operator<(const date& b)const {
		if (y != b.y) return y < b.y;
		if (m != b.m) return m < b.m;
		return d < b.d;
	}
	bool operator==(const date& b)const {
		if (y == b.y && m == b.m && d == b.d) return true;
		return false;
	}
};
```

```cpp
int dayofweek(int y, int m, int d) /* 0 = Sunday */
{
    int t[] = { 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 };
    y -= m < 3;
    return (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
}
```

## 小波树

时间复杂度都是 $logn$ 。

所有的下标都是从 $0$ 开始。

区间第 $k$ 小（ $k$ 从 $0$ 开始，$k=1$ 代表第$ 2$ 小）。

区间某个数出现的频率（如果 $val$ 不在数据集中会返回错误的值）。

区间小于等于某个数的个数。

```cpp
struct BitRank {
    // block 管理一行一行的bit
    std::vector<unsigned long long> block;
    std::vector<unsigned int> count;
    BitRank() {}
    // 位向量长度
    void resize(const unsigned int num) {
        block.resize(((num + 1) >> 6) + 1, 0);
        count.resize(block.size(), 0);
    }
    // 设置i位bit
    void set(const unsigned int i, const unsigned long long val) {
        block[i >> 6] |= (val << (i & 63));
    }
    void build() {
        for (unsigned int i = 1; i < block.size(); i++) {
            count[i] = count[i - 1] + __builtin_popcountll(block[i - 1]);
        }
    }
    // [0, i) 1的个数
    unsigned int rank1(const unsigned int i) const {
        return count[i >> 6] +
            __builtin_popcountll(block[i >> 6] & ((1ULL << (i & 63)) - 1ULL));
    }
    // [i, j) 1的个数
    unsigned int rank1(const unsigned int i, const unsigned int j) const {
        return rank1(j) - rank1(i);
    }
    // [0, i) 0的个数
    unsigned int rank0(const unsigned int i) const { return i - rank1(i); }
    // [i, j) 0的个数
    unsigned int rank0(const unsigned int i, const unsigned int j) const {
        return rank0(j) - rank0(i);
    }
};


class WaveletMatrix {
private:
    unsigned int height;
    std::vector<BitRank> B;
    std::vector<int> pos;

public:
    WaveletMatrix() {}
    WaveletMatrix(std::vector<int> vec)
        : WaveletMatrix(vec, *std::max_element(vec.begin(), vec.end()) + 1) {}
    // sigma: 字母表大小(字符串的话)，数字序列的话是数的种类
    WaveletMatrix(std::vector<int> vec, const unsigned int sigma) {
        init(vec, sigma);
    }
    void init(std::vector<int>& vec, const unsigned int sigma) {
        height = (sigma == 1) ? 1 : (64 - __builtin_clzll(sigma - 1));
        B.resize(height), pos.resize(height);
        for (unsigned int i = 0; i < height; ++i) {
            B[i].resize(vec.size());
            for (unsigned int j = 0; j < vec.size(); ++j) {
                B[i].set(j, get(vec[j], height - i - 1));
            }
            B[i].build();
            auto it = stable_partition(vec.begin(), vec.end(), [&](int c) {
                return !get(c, height - i - 1);
            });
            pos[i] = it - vec.begin();
        }
    }

    int get(const int val, const int i) { return val >> i & 1; }
    
    // [l, r) 中val出现的频率
    int rank(const int val, const int l, const int r) {
        return rank(val, r) - rank(val, l);
    } 
    // [0, i) 中val出现的频率
    int rank(int val, int i) {
        int p = 0;
        for (unsigned int j = 0; j < height; ++j) {
            if (get(val, height - j - 1)) {
                p = pos[j] + B[j].rank1(p);
                i = pos[j] + B[j].rank1(i);
            } else {
                p = B[j].rank0(p);
                i = B[j].rank0(i);
            }
        }
        return i - p;
    }
    // [l, r) 中k小
    int quantile(int k, int l, int r) {
        int res = 0;
        for (unsigned int i = 0; i < height; ++i) {
            const int j = B[i].rank0(l, r);
            if (j > k) {
                l = B[i].rank0(l);
                r = B[i].rank0(r);
            } else {
                l = pos[i] + B[i].rank1(l);
                r = pos[i] + B[i].rank1(r);
                k -= j;
                res |= (1 << (height - i - 1));
            }
        }
        return res;
    }
    int rangefreq(const int i, const int j, const int a, const int b, const int l,
                  const int r, const int x) {
        if (i == j || r <= a || b <= l) return 0;
        const int mid = (l + r) >> 1;
        if (a <= l && r <= b) {
            return j - i;
        } else {
            const int left =
                rangefreq(B[x].rank0(i), B[x].rank0(j), a, b, l, mid, x + 1);
            const int right = rangefreq(pos[x] + B[x].rank1(i),
                                        pos[x] + B[x].rank1(j), a, b, mid, r, x + 1);
            return left + right;
        }
    }
    // [l,r) 在[a, b) 值域的数字个数
    int rangefreq(const int l, const int r, const int a, const int b) {
        return rangefreq(l, r, a, b, 0, 1 << height, 0);
    }
    int rangemin(const int i, const int j, const int a, const int b, const int l,
                 const int r, const int x, const int val) {
        if (i == j || r <= a || b <= l) return -1;
        if (r - l == 1) return val;
        const int mid = (l + r) >> 1;
        const int res =
            rangemin(B[x].rank0(i), B[x].rank0(j), a, b, l, mid, x + 1, val);
        if (res < 0)
            return rangemin(pos[x] + B[x].rank1(i), pos[x] + B[x].rank1(j), a, b, mid,
                            r, x + 1, val + (1 << (height - x - 1)));
        else
            return res;
    }
    // [l,r) 在[a,b) 值域内存在的最小值是什么，不存在返回-1
    int rangemin(int l, int r, int a, int b) {
        return rangemin(l, r, a, b, 0, 1 << height, 0, 0);
    }
};
```

### 简化版（不保证对）

内存，时间都增大为160%

```cpp
struct BitRank {
    // block 管理一行一行的bit
    vector<long long> block;
    vector<int> count;
    BitRank() {}
    // 位向量长度
    void resize(int n) {
        block.resize(((n + 1) >> 6) + 1, 0);
        count.resize(block.size(), 0);
    }
    // 设置i位bit
    void set(int i, long long val) {
        block[i >> 6] |= (val << (i & 63));
    }
    void build() {
        for (int i = 1; i < block.size(); i++) {
            count[i] = count[i - 1] + __builtin_popcountll(block[i - 1]);
        }
    }
    // [0, i) 1的个数
     int rank1(int i)  {
        return count[i >> 6] +
            __builtin_popcountll(block[i >> 6] & ((1ULL << (i & 63)) - 1ULL));
    }
    // [i, j) 1的个数
     int rank1(int i, int j)  {
        return rank1(j) - rank1(i);
    }
    // [0, i) 0的个数
     int rank0(int i)  { return i - rank1(i); }
    // [i, j) 0的个数
     int rank0(int i, int j)  {
        return rank0(j) - rank0(i);
    }
};

class WM {
private:
    int height;
    vector<BitRank> B;
    vector<int> pos;

public:
    WM() {}
    WM(vector<int> vec)
        : WM(vec, *max_element(vec.begin(), vec.end()) + 1) {}
    // sigma: 字母表大小(字符串的话)，数字序列的话是数的种类
    WM(vector<int> vec, int sigma) {
        init(vec, sigma);
    }
    void init(vector<int>& vec, int sigma) {
        height = (sigma == 1) ? 1 : (64 - __builtin_clzll(sigma - 1));
        B.resize(height), pos.resize(height);
        for ( int i = 0; i < height; ++i) {
            B[i].resize(vec.size());
            for ( int j = 0; j < vec.size(); ++j) {
                B[i].set(j, get(vec[j], height - i - 1));
            }
            B[i].build();
            auto it = stable_partition(vec.begin(), vec.end(), [&](int c) {
                return !get(c, height - i - 1);
            });
            pos[i] = it - vec.begin();
        }
    }

    int get(int val, int i) { return val >> i & 1; }
    
    // [l, r) 中val出现的频率
    int rank(int val, int l, int r) {
        return rank(val, r) - rank(val, l);
    } 
    // [0, i) 中val出现的频率
    int rank(int val, int i) {
        int p = 0;
        for ( int j = 0; j < height; ++j) {
            if (get(val, height - j - 1)) {
                p = pos[j] + B[j].rank1(p);
                i = pos[j] + B[j].rank1(i);
            } else {
                p = B[j].rank0(p);
                i = B[j].rank0(i);
            }
        }
        return i - p;
    }
    // [l, r) 中k小
    int quantile(int k, int l, int r) {
        int res = 0;
        for ( int i = 0; i < height; ++i) {
             int j = B[i].rank0(l, r);
            if (j > k) {
                l = B[i].rank0(l);
                r = B[i].rank0(r);
            } else {
                l = pos[i] + B[i].rank1(l);
                r = pos[i] + B[i].rank1(r);
                k -= j;
                res |= (1 << (height - i - 1));
            }
        }
        return res;
    }
    int rangefreq(int i, int j, int a, int b, int l, int r, int x) {
        if (i == j || r <= a || b <= l) return 0;
         int mid = (l + r) >> 1;
        if (a <= l && r <= b) {
            return j - i;
        } else {
             int left =
                rangefreq(B[x].rank0(i), B[x].rank0(j), a, b, l, mid, x + 1);
             int right = rangefreq(pos[x] + B[x].rank1(i),
                                        pos[x] + B[x].rank1(j), a, b, mid, r, x + 1);
            return left + right;
        }
    }
    // [l,r) 在[a, b) 值域的数字个数
    int rangefreq( int l,  int r,  int a,  int b) {
        return rangefreq(l, r, a, b, 0, 1 << height, 0);
    }
    int rangemin( int i,  int j,  int a,  int b,  int l, int r,  int x,  int val) {
        if (i == j || r <= a || b <= l) return -1;
        if (r - l == 1) return val;
         int mid = (l + r) >> 1;
         int res =
            rangemin(B[x].rank0(i), B[x].rank0(j), a, b, l, mid, x + 1, val);
        if (res < 0)
            return rangemin(pos[x] + B[x].rank1(i), pos[x] + B[x].rank1(j), a, b, mid,
                            r, x + 1, val + (1 << (height - x - 1)));
        else
            return res;
    }
    // [l,r) 在[a,b) 值域内存在的最小值是什么，不存在返回-1
    int rangemin(int l, int r, int a, int b) {
        return rangemin(l, r, a, b, 0, 1 << height, 0, 0);
    }  
};
```

```cpp
struct BitRank {
    // block 管理一行一行的bit
    vector<long long> block;
    vector<int> count;
    BitRank() {}
    // 位向量长度
    void resize(int n) {
        block.resize(((n + 1) >> 6) + 1, 0);
        count.resize(block.size(), 0);
    }
    // 设置i位bit
    void set(int i, long long val) {
        block[i >> 6] |= (val << (i & 63));
    }
    void build() {
        for (int i = 1; i < block.size(); i++) {
            count[i] = count[i - 1] + __builtin_popcountll(block[i - 1]);
        }
    }
    // [0, i) 1的个数
    int r1(int i) {
        return count[i >> 6] +
            __builtin_popcountll(block[i >> 6] & ((1ULL << (i & 63)) - 1ULL));
    }
    // [i, j) 1的个数
    int r1(int i, int j) {
        return r1(j) - r1(i);
    }
    // [0, i) 0的个数
    int r0(int i) { return i - r1(i); }
    // [i, j) 0的个数
    int r0(int i, int j) {
        return r0(j) - r0(i);
    }
};

class WM {
private:
    int h;
    vector<BitRank> B;
    vector<int> pos;

public:
    WM() {}
    WM(vector<int> v)
        : WM(v, *max_element(v.begin(), v.end()) + 1) {
    }
    // ans: 字母表大小(字符串的话)，数字序列的话是数的种类
    WM(vector<int> v, int ans) {
        init(v, ans);
    }
    void init(vector<int>& v, int ans) {
        h = (ans == 1) ? 1 : (64 - __builtin_clzll(ans - 1));
        B.resize(h), pos.resize(h);
        for (int i = 0; i < h; ++i) {
            B[i].resize(v.size());
            for (int j = 0; j < v.size(); ++j) {
                B[i].set(j, get(v[j], h - i - 1));
            }
            B[i].build();
            auto it = stable_partition(v.begin(), v.end(), [&](int c) {
                return !get(c, h - i - 1);
                                       });
            pos[i] = it - v.begin();
        }
    }

    int get(int val, int i) { return val >> i & 1; }

    // [l, r) 中val出现的频率
    int rank(int val, int l, int r) {
        return rank(val, r) - rank(val, l);
    }
    // [0, i) 中val出现的频率
    int rank(int val, int i) {
        int p = 0;
        for (int j = 0; j < h; ++j) {
            if (get(val, h - j - 1)) {
                p = pos[j] + B[j].r1(p);
                i = pos[j] + B[j].r1(i);
            }
            else {
                p = B[j].r0(p);
                i = B[j].r0(i);
            }
        }
        return i - p;
    }
    // [l, r) 中k小
    int quantile(int k, int l, int r) {
        int res = 0;
        for (int i = 0; i < h; ++i) {
            int j = B[i].r0(l, r);
            if (j > k) {
                l = B[i].r0(l);
                r = B[i].r0(r);
            }
            else {
                l = pos[i] + B[i].r1(l);
                r = pos[i] + B[i].r1(r);
                k -= j;
                res |= (1 << (h - i - 1));
            }
        }
        return res;
    }
    int rangefreq(int i, int j, int a, int b, int l, int r, int x) {
        if (i == j || r <= a || b <= l) return 0;
        int mid = (l + r) >> 1;
        if (a <= l && r <= b) {
            return j - i;
        }
        else {
            int left = rangefreq(B[x].r0(i), B[x].r0(j), a, b, l, mid, x + 1);
            int right = rangefreq(pos[x] + B[x].r1(i), pos[x] + B[x].r1(j), a, b, mid, r, x + 1);
            return left + right;
        }
    }
    // [l,r) 在[a, b) 值域的数字个数
    int rangefreq(int l, int r, int a, int b) {
        return rangefreq(l, r, a, b, 0, 1 << h, 0);
    }
    int rangemin(int i, int j, int a, int b, int l, int r, int x, int val) {
        if (i == j || r <= a || b <= l) return -1;
        if (r - l == 1) return val;
        int mid = (l + r) >> 1;
        int res = rangemin(B[x].r0(i), B[x].r0(j), a, b, l, mid, x + 1, val);
        if (res < 0)
            return rangemin(pos[x] + B[x].r1(i), pos[x] + B[x].r1(j), a, b, mid, r, x + 1, val + (1 << (h - x - 1)));
        else
            return res;
    }
    // [l,r) 在[a,b) 值域内存在的最小值是什么，不存在返回-1
    int rangemin(int l, int r, int a, int b) {
        return rangemin(l, r, a, b, 0, 1 << h, 0, 0);
    }
};
```

## 进制转换

对于 $n≥5$，它们的 $(n−2)$ 进制表示都是 $12$ 。

```cpp
int check1(char c){
	if(c>='0'&&c<='9') return c-'0';
	else return c-'A'+10;
}

char check2(int c){
	if(c>=0&&c<=9) return c+'0';
	else return c-10+'A';
}

//将base进制的str转化成10进制 
int trans1(int base,string str){
	int ans=0,p=1;
	for(int i=str.size()-1;i>=0;i--){
		ans+=check1(str[i])*p;
		p=p*base; 
	}
	return ans;
}

//将10进制的x转化为base进制
string trans2(int base,int x){
	string str;
	while(x){
		str.push_back(check2(x%base));
		x=x/base;
	}
	reverse(str.begin(),str.end());
    if (str.size() == 0) return "0";
	return str;
}

// 将base1进制的s转化为base2进制
string trans(int base1,string s,int base2){
	return trans2(base2,trans1(base1,s));
}
```

## 取模类

```cpp
class mint {
    static const int mod = 998244353;
public:
    typedef long long LL;
    LL x;
    mint() : x(0) {}
    mint(LL _x) : x((_x% mod + mod) % mod) {}
    mint& operator = (LL b) { return *this = mint(b); }

    friend bool operator < (mint a, mint b) { return a.x < b.x; }
    friend bool operator > (mint a, mint b) { return a.x > b.x; }
    friend bool operator <= (mint a, mint b) { return a.x <= b.x; }
    friend bool operator >= (mint a, mint b) { return a.x >= b.x; }
    friend bool operator == (mint a, mint b) { return a.x == b.x; }

    friend mint operator + (mint a, mint b) { return mint((a.x + b.x) % mod); }
    friend mint& operator += (mint& a, mint b) { return a = a + b; }
    friend mint operator - (mint a, mint b) { return mint(((a.x - b.x) % mod + mod) % mod); }
    friend mint& operator -= (mint& a, mint b) { return a = a - b; }
    friend mint operator * (mint a, mint b) { return mint(a.x * b.x % mod); }
    friend mint& operator *= (mint& a, mint b) { return a = a * b; }
    mint inv() {
        LL k = mod - 2, res = 1, cnt = x;
        while (k) {
            if (k & 1) res = (res * cnt) % mod;
            cnt = cnt * cnt % mod;
            k >>= 1;
        }
        return mint(res);
    }
    friend mint operator / (mint a, mint b) { return a * b.inv(); }
    friend mint& operator /= (mint& a, mint b) { return a = a / b; }
};
```

