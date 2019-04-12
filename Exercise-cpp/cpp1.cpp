//编写函数模板，实现整型数据，实型和字符串的交换
#include <iostream>
template<class T>
void change(T &i,T &j)
{
	T temp;
	temp = i;
	i = j;
	j = temp;
 } 
 int main(){
 	int a,b;
 	cout << "输入两个整数a,b:";
 	cin >> c >> d;
 	change(c,d);
 	cout << "\n交换后为a,b:" << a << "" << b;
 	cout << "\n输入两个实数c,d:";
 	cin >> c >> d;
 	change(c,d);
 	cout << "\交换后为c,d:" << c << "" << d;
	cin.get();
	char *s1 = "",char *s2 = "";
	cout << "\n输入第一个字符串s1:";
	char str1[20],str2[20];
	s1 = str1;s2 = str2;
	cin.getline(s1,20);
	cout << "\n输入第二个字符串s2:";
	cin.getline(s2,20);
	change(s1,s2);
	cout << "\交换后为s1,s2:" << s1 << "" << s2 << endl; 
 }
 
 
 
 
 
 
 
