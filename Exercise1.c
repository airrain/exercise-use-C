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
 
 
 
 
 
 
 
 
 
 
 
  
 
 
 
 
  





