//��д����ģ�壬ʵ���������ݣ�ʵ�ͺ��ַ����Ľ���
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
 	cout << "������������a,b:";
 	cin >> c >> d;
 	change(c,d);
 	cout << "\n������Ϊa,b:" << a << "" << b;
 	cout << "\n��������ʵ��c,d:";
 	cin >> c >> d;
 	change(c,d);
 	cout << "\������Ϊc,d:" << c << "" << d;
	cin.get();
	char *s1 = "",char *s2 = "";
	cout << "\n�����һ���ַ���s1:";
	char str1[20],str2[20];
	s1 = str1;s2 = str2;
	cin.getline(s1,20);
	cout << "\n����ڶ����ַ���s2:";
	cin.getline(s2,20);
	change(s1,s2);
	cout << "\������Ϊs1,s2:" << s1 << "" << s2 << endl; 
 }
 
 
 
 
 
 
 
