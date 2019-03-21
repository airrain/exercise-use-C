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









