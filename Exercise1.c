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
 
 //���С��lim��������������������
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
 
 //�Ӵ����num���ַ������ҵ����һ��
char *fun(char (*a)[81],int num,char *max)
{
	int i = 0;
	max = a[0];
	for(i = 0;i < num;i++)
	
		if(strlen(max) < strlen(a[i]))
		max = a[i];
	return max;
	
 } 
 
 //ɾ���ַ����е����пո�
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
 
 //�ж��ַ����Ƿ�Ϊ����
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
 
 
 //����ά�����е����ݷŵ�һά������
 void fun(int (*s)[10],int *b,int *n,int nn)
 {
 	int i,j,k = 0;
 	for(i = 0;i < mm;i++)
 		for(j = 0;j < nn;j++)
 			b[k++] = s[i][j];
 	*n = k;
  } 
 
 //��s��ָ�ַ������±�Ϊż��ͬʱASCIIֵΪ�������ַ�ɾ��
 void fun(char *s,char t[])
 {
 	int i ,j = 0;
 	for(i = 0;i < strlem(s);i++)
 		if(!(i%2) == 0 && (s[i]%2))
 			t[j++] = s[i];
 	t[j] = 0;	 
 }
 
 //��ss��ָ�ַ����������±�Ϊ����λ�õ���ĸת��Ϊ��д
 void fun (char *ss)
 {
 	int i;
 	for(i = 0;ss[i] != '\0';i++)
 		if(i%2 == 1 && ss[i] >= 'a' && ss[i] <= 'z')
 			ss[i] == ss[i] - 32;
 }
 
//��a,b�е������������ϲ���һ���µ���������c��
void fun(int a,int b,long *c)
{
	*c = (a%10)*1000 + (b%10)*100 + (a/10)*10 + (b/10);
 } 
 
 //��д���������㼶����
 double fun(double x,int n){
 	int i;
 	double s = 1.0;s1 = 1.0;
 	for(i = 1;i <= n;i++){
 		s1 = s1 * i;
		 s = s + pow(x,i)/s1; 
	 }
	 return s;
 } 
 
 //��С���β�nͬʱ�ܱ�3��7������������Ȼ��ֻ�͵�ƽ����
 double fun(int n){
 	double sum = 0.0;
 	int i;
 	for(i = 21;i <= n;i++){
 		if((i%3) == 0 && (i%7 == 0))
 			sum += i;
 			return sqrt(sum);
	 }
 } 
 
 //�ƶ��ַ����е����ݣ��ѵ�1~m���ַ���ƽ�Ƶ��ַ�������󣬰ѵ�m+1�������ַ��Ƶ��ַ�����ǰ����
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
 
//��һ��÷֣�ȥ��һ����߷ֺ�һ����ͷ֣�Ȼ����ƽ��ֵ
 
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
 
 //��m��n�еĶ�ά�����е��ַ����ݣ����еĵ�˳�����ηŵ�һ���ַ�����
void fun(char(*s)[N],char *b){
	int i,j,k = 0;
	for(i = 0;i < N;i++)
		b[k++] = a[j][i];
	b[k] = '\0'; 
} 

 //��s��ֵ�ַ�����ASCIIֵΪ�������ַ�ɾ����ʣ���ַ��γɵ��´�����t��ָ������ 
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
//ɾ��һλ�����л���������ͬ������ʹ��ֻʣһ��
int fun(int a[],int n){
	int i,j = 1;
	for(i = i;i < n;i++)
		if(a[j - 1] != a[i])
			a[j++] = a[i];
	return j;
}
 
 //��һ�������ַ���ת��Ϊһ������
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
 
 //��Fibonacci�����д���t����С����������ɺ�������
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
 
 //���㲢�����������n����������
 int fun(int n){
 	int s = 0,i;
 	for(i = 2;i <= n;i++)
		if(n % i == 0)
			s += i;
	return s;  
 } 





