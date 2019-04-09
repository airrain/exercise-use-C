//统计一行字符串中单词的个数，作为函数值返回

int fun(char *s) {
	int i ,j = 0
	           for(i = 0; s[i] != '\0'; i++)
		           if(s[i] != '' && (s[i +1] == '' || s[i + 1] == '\0'))
			           j++;
	return j;
}

//找出成绩最低的学生成绩，并返回

fun(STU[],STU *s) {
	int i;
	*s = a[0]
	for(i = 0;i < N;i++)
		if(s->s > a[i].s)
			*s = a[i];	
}


//数组右上半角元素乘以m
void fun(int a[][N],int m)
{
	int i,j;
	for(j = 0;j < N;j++)
		for(i = 0;i <= j;i++)
			a[i][j] = a[i][j] * m;
			
 } 

//求出数组周边的元素平均值并作为函数返回值返回给主函数中的s
double fun(int w[][N])
{
	int i,j,k = 0;
	double sum = 0.0;
	for(i = 0;i < N;i++)
		for(j = 0;j < N;j++)
			if(i == 0 || i == N - 1 || j == 0 || j == N -1)
			{
				sum += w[i][j];
				k++;
			 } 
		return sum/k;
 } 
 
 //保留两位小数
 float fun(float h)
 {
 	int tmp = (int)(h * 1000 + 5)/10;
 	return (float)tmp100.0;
  } 

//求出二维数组周边元素之和
 int fun(int a[M][N])
 {
 	int i,j,sum = 0;
 	for(i = 0;i < M;i++)
 		for(j = 0;j < N;j++)
 			if(i == 0 || i == M -1 ||j == 0 || j == N - 1)
 				sum += a[i][j];
 	return sum;
  } 

//将矩阵的行列进行转换
void fun(int array[3][3])
{
	int i,j;
	for(i = 0;i < 3;i++)
		for(j = i + 1;j < 3;j++)
		{
			t = array[i][j]
			array[i][j] = array[j][i];
			array[j][i] = t;
			
		}
 } 
 
 //求出小于lim的所有素数并存入数组
 int fun(int lim,int aa[MAX])
 {
 	int i,j,k = 0;
 	for(i = 2;i <= lim;i++)
 	{
 		for(j = 2;j < i;j++)
 			if(i % j == 0) break;
 		if(j > i)
 			aa[k++] = i;
 			
	 }
	 return k;
 }
 
 //从传入的num个字符串中找到最长的一个
char *fun(char (*a)[81],int num,char *max)
{
	int i = 0;
	max = a[0];
	for(i = 0;i < num;i++)
	
		if(strlen(max) < strlen(a[i]))
		max = a[i];
	return max;
	
 } 
 
 //删除字符串中的所有空格
 void fun(char *str)
 {
 	int i = 0;
 	char *p = str;
 	while(*p)
 	{
 		if(*p != '')
 		{
 			str[i++] = *p;
		 }
		 p++;
	 }
	 str[i] = '\0';
  } 
 
 //判断字符串是否为回文
 int fun(char *str)
 {
 	int i,n = 0;fg = 1;
 	char *p = str;
 	while(*p){
 		n++;p++;
	 }
	 for(i = 0;i < n/2;i++)
	 	if(str[i] == str[n-1-i]);
		else{
			fg = 0;break;
		}	
		return fg;
  } 
 
 
 //将二维数组中的数据放到一维数组中
 void fun(int (*s)[10],int *b,int *n,int nn)
 {
 	int i,j,k = 0;
 	for(i = 0;i < mm;i++)
 		for(j = 0;j < nn;j++)
 			b[k++] = s[i][j];
 	*n = k;
  } 
 
 //将s所指字符串中下标为偶数同时ASCII值为奇数的字符删除
 void fun(char *s,char t[])
 {
 	int i ,j = 0;
 	for(i = 0;i < strlem(s);i++)
 		if(!(i%2) == 0 && (s[i]%2))
 			t[j++] = s[i];
 	t[j] = 0;	 
 }
 
 //将ss所指字符串中所有下标为奇数位置的字母转换为大写
 void fun (char *ss)
 {
 	int i;
 	for(i = 0;ss[i] != '\0';i++)
 		if(i%2 == 1 && ss[i] >= 'a' && ss[i] <= 'z')
 			ss[i] == ss[i] - 32;
 }
 
//将a,b中地两个正整数合并成一个新的整数放在c中
void fun(int a,int b,long *c)
{
	*c = (a%10)*1000 + (b%10)*100 + (a/10)*10 + (b/10);
 } 
 
 //编写函数，计算级数和
 double fun(double x,int n){
 	int i;
 	double s = 1.0;s1 = 1.0;
 	for(i = 1;i <= n;i++){
 		s1 = s1 * i;
		 s = s + pow(x,i)/s1; 
	 }
	 return s;
 } 
 
 //求小于形参n同时能被3与7整除的所有自然数只和的平方根
 double fun(int n){
 	double sum = 0.0;
 	int i;
 	for(i = 21;i <= n;i++){
 		if((i%3) == 0 && (i%7 == 0))
 			sum += i;
 			return sqrt(sum);
	 }
 } 
 
 //移动字符串中的内容，把第1~m个字符，平移到字符串的最后，把第m+1到最后的字符移到字符串的前部。
 void fun(char *w,int m){
 	int i,j;
 	char t;
 	for(i = 1;i <= m;i++){
 		t = w[0];
 		for(j = 1;w[j] != '\0';j++){
 			w[j-1] = w[j];
		 }
		 w[j-1] = t;
	 }
 } 
 
//将一组得分，去掉一个最高分和一个最低分，然后求平均值
 
 double fun(double a[],int n){
 	double sum = 0,max,min;
 	int i;
 	max = min = a[0];
 	for(i = 0;i < n;i++){
 		sum += a[i];
 		if(max < a[i]) max = a[i];
 		if(min > a[i]) min = a[i];
	 }
	 sum = sum - max - min;
	 return (sum/(n-2));
 }
 
 //将m行n列的二维数组中的字符数据，按列的的顺序依次放到一个字符串中
void fun(char(*s)[N],char *b){
	int i,j,k = 0;
	for(i = 0;i < N;i++)
		b[k++] = a[j][i];
	b[k] = '\0'; 
} 

 //将s所值字符串中ASCII值为奇数的字符删除，剩余字符形成的新串放在t所指数组中 
 void fun(char *s,char t[])
 {
 	int i,j = 0,n;
 	n = strlen(s);
 	for(i = 0;i < n;i++)
 	{
 		if(s[i] % 2 == 0)
 		{
 			t[j] = s[i];
 			j++;
		 }
		 t[j] = '\0';
	 }
 }
//删除一位数组中华的所有相同的数，使得只剩一个
int fun(int a[],int n){
	int i,j = 1;
	for(i = i;i < n;i++)
		if(a[j - 1] != a[i])
			a[j++] = a[i];
	return j;
}
 
 //将一个数字字符串转换为一个整数
 long fun(char *p){
 	long n = 0;
 	int flag = 1;
 	if(*p == '-')
	 {
	 	p++;
	 	flag = -1;
	 }
	 else if(*p == '+')
	 	p++;
	 while(*p != '\0')
	 {
	 	n = n * 10 + *p - '0'
	 	p++;
	 }
	 return n*flag;
	 
 } 
 
 //求Fibonacci数列中大于t的最小的数，结果由函数返回
 int fun(int t){
 	int f0 = 0,f1 = 1,f;
 	do{
 		f = f0 + f1;
 		f0 = f1;
 		f1 = f;
	 }
	 while(f < t);
	 return f;
 }
 
 //计算并输出给定整数n的所有因子
 int fun(int n){
 	int s = 0,i;
 	for(i = 2;i <= n;i++)
		if(n % i == 0)
			s += i;
	return s;  
 } 





