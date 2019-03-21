//统计一行字符串中单词的个数，作为函数值返回

int fun(char *s)
{
	int i ,j = 0
	for(i = 0;s[i] != '\0';i++)
		if(s[i] != '' && (s[i +1] == '' || s[i + 1] == '\0'))
			j++;
	return j;
 } 
