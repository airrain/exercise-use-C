//ͳ��һ���ַ����е��ʵĸ�������Ϊ����ֵ����

int fun(char *s) {
	int i ,j = 0
	           for(i = 0; s[i] != '\0'; i++)
		           if(s[i] != '' && (s[i +1] == '' || s[i + 1] == '\0'))
			           j++;
	return j;
}

//�ҳ��ɼ���͵�ѧ���ɼ���������

fun(STU[],STU *s) {
	int i;
	*s = a[0]
	for(i = 0;i < N;i++)
		if(s->s > a[i].s)
			*s = a[i];	
}


//�������ϰ��Ԫ�س���m
void fun(int a[][N],int m)
{
	int i,j;
	for(j = 0;j < N;j++)
		for(i = 0;i <= j;i++)
			a[i][j] = a[i][j] * m;
			
 } 

//��������ܱߵ�Ԫ��ƽ��ֵ����Ϊ��������ֵ���ظ��������е�s
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
 
 //������λС��
 float fun(float h)
 {
 	int tmp = (int)(h * 1000 + 5)/10;
 	return (float)tmp100.0;
  } 

//�����ά�����ܱ�Ԫ��֮��
 int fun(int a[M][N])
 {
 	int i,j,sum = 0;
 	for(i = 0;i < M;i++)
 		for(j = 0;j < N;j++)
 			if(i == 0 || i == M -1 ||j == 0 || j == N - 1)
 				sum += a[i][j];
 	return sum;
  } 

//����������н���ת��
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





